"""Instance-aware Reweighting (IAR), Eqs. (6)--(12) in the paper."""
from __future__ import annotations

import math

import numpy as np

from .pga import aggregate_generic_evidence, assign_observation, select_proposal


def frame_completeness(mask: np.ndarray) -> dict[str, float]:
    """Prediction-only parent visibility from scale and boundary margin."""
    yy, xx = np.nonzero(mask)
    if len(xx) == 0:
        return {"area": 0.0, "margin": 0.0, "scale": 0.0, "completeness": 0.0,
                "center_x": 0.0, "center_y": 0.0}
    height, width = mask.shape
    area = float(mask.mean())
    margin = max(
        0.0,
        float(
            min(
                xx.min() / width,
                (width - 1 - xx.max()) / width,
                yy.min() / height,
                (height - 1 - yy.max()) / height,
            )
        ),
    )
    scale = min(1.0, area / 0.03) * min(1.0, 0.60 / max(area, 1e-9))
    return {
        "area": area,
        "margin": margin,
        "scale": scale,
        "completeness": scale * math.sqrt(0.01 + margin),
        "center_x": float(np.median(xx) / width),
        "center_y": float(np.median(yy) / height),
    }


def observation_quality(detection_confidence: float, completeness: float) -> float:
    """Eq. (8): q_t = c_t v_t."""
    return float(detection_confidence) * float(completeness)


def nearest_track_quality(
    observation: dict, track_detections: list[dict], temporal_window_s: float
) -> tuple[float | None, float | None]:
    """Align an original grounding frame with the selected physical track."""
    same_video = [
        detection
        for detection in track_detections
        if str(detection["video_id"]) == str(observation["video_id"])
    ]
    if not same_video:
        return None, None
    timestamp = float(observation["frame_id"])
    distances = [abs(float(row["frame_id"]) - timestamp) for row in same_video]
    delta = min(distances)
    if delta > temporal_window_s + 1e-9:
        return None, delta
    tied = [row for row, dt in zip(same_video, distances) if abs(dt - delta) <= 1e-9]
    return max(float(row["quality"]) for row in tied), delta


def sigmoid_quality_weights(
    qualities: np.ndarray, beta: float = 0.20, gamma: float = 0.02
) -> np.ndarray:
    """Eqs. (9)--(10), with per-query max normalization."""
    if qualities.size == 0:
        return np.zeros(0, dtype=np.float64)
    maximum = float(qualities.max())
    normalized = qualities / maximum if maximum > 0 else np.ones_like(qualities)
    logits = np.clip((normalized - beta) / gamma, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-logits))


def predict(
    *,
    observations: list[dict],
    number_of_proposals: int,
    relational: bool,
    selected_track_detections: list[dict],
    generic_support_ratio: float = 1.30,
    relational_support_ratio: float = 1.10,
    temporal_window_s: float = 0.90,
    beta: float = 0.20,
    gamma: float = 0.02,
) -> dict:
    """Run PGA and, for relational queries, IAR over the same observations."""
    generic_votes = aggregate_generic_evidence(
        observations, number_of_proposals, generic_support_ratio
    )
    generic_prediction = select_proposal(generic_votes)
    if not relational:
        return {
            "selected_proposal": generic_prediction,
            "generic_prediction": generic_prediction,
            "route": "PGA",
            "generic_evidence": generic_votes.tolist(),
            "instance_aware_evidence": generic_votes.tolist(),
            "aligned_observations": [],
        }

    aligned: list[tuple[int, float, dict, float]] = []
    for observation in observations:
        quality, delta = nearest_track_quality(
            observation, selected_track_detections, temporal_window_s
        )
        if quality is None:
            continue
        proposal = assign_observation(observation, relational_support_ratio)
        if 0 <= proposal < number_of_proposals:
            aligned.append((proposal, quality, observation, float(delta)))

    # Paper fallback: if A* is empty, use unweighted proposal aggregation.
    if not aligned:
        return {
            "selected_proposal": generic_prediction,
            "generic_prediction": generic_prediction,
            "route": "PGA (IAR fallback: no aligned observation)",
            "generic_evidence": generic_votes.tolist(),
            "instance_aware_evidence": np.zeros(number_of_proposals).tolist(),
            "aligned_observations": [],
        }

    qualities = np.asarray([row[1] for row in aligned], dtype=np.float64)
    weights = sigmoid_quality_weights(qualities, beta, gamma)
    weighted_votes = np.zeros(number_of_proposals, dtype=np.float64)
    diagnostics = []
    for (proposal, quality, observation, delta), weight in zip(aligned, weights):
        weighted_votes[proposal] += float(weight)
        diagnostics.append(
            {
                "video_id": str(observation["video_id"]),
                "frame_id": str(observation["frame_id"]),
                "assigned_proposal": proposal,
                "quality": float(quality),
                "weight": float(weight),
                "temporal_distance": delta,
            }
        )
    return {
        "selected_proposal": select_proposal(weighted_votes),
        "generic_prediction": generic_prediction,
        "route": "PGA + IAR",
        "generic_evidence": generic_votes.tolist(),
        "instance_aware_evidence": weighted_votes.tolist(),
        "aligned_observations": diagnostics,
    }


def recover_paper_track(rebuilt: dict, frozen: dict) -> list[dict]:
    """Compatibility helper for the frozen validation checkpoint only.

    OWLv2 labels were added to the local cache after the paper experiment.
    Four saved identity frames recover the same prediction-only physical track
    in the extended timeline. Fresh inference must use ``chosen_detections``.
    """
    if not rebuilt or not frozen or frozen.get("chosen_track") is None:
        return []
    historical = next(
        (row for row in frozen.get("tracks", [])
         if int(row["track"]) == int(frozen["chosen_track"])),
        None,
    )
    if historical is None:
        return []
    anchors = frozen.get("selected", [])[:4]

    def match(track: dict) -> tuple[int, int, float, int]:
        detections = track.get("detections", [])
        exact = near = 0
        total_distance = 0.0
        for anchor in anchors:
            same = [row for row in detections
                    if str(row["video_id"]) == str(anchor["video_id"])]
            if not same:
                total_distance += 1e6
                continue
            delta = min(abs(float(row["frame_id"]) - float(anchor["frame_id"]))
                        for row in same)
            exact += int(delta <= 1e-6)
            near += int(delta <= 0.30)
            total_distance += delta
        return exact, near, total_distance, abs(len(detections) - int(historical["n"]))

    candidates = rebuilt.get("track_detections", [])
    if not candidates:
        return []
    selected = min(
        candidates,
        key=lambda row: (-match(row)[0], -match(row)[1], match(row)[2],
                         match(row)[3], int(row["track"])),
    )
    exact, near, _, _ = match(selected)
    return list(selected.get("detections", [])) if exact or near else []

