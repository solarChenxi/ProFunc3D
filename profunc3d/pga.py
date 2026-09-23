"""Proposal-grounded Association (PGA), Eqs. (2)--(5) in the paper."""
from __future__ import annotations

from typing import Iterable

import numpy as np


def eligible_proposals(distances: np.ndarray, max_distance_px: float) -> np.ndarray:
    """Return visible proposals close enough to a sparse grounding point."""
    indices = np.flatnonzero(np.isfinite(distances) & (distances <= max_distance_px))
    if indices.size == 0:
        return indices
    return indices[np.argsort(distances[indices], kind="mergesort")]


def local_surface_support(
    projection: np.ndarray,
    proposal_locations: np.ndarray,
    point_x: float,
    point_y: float,
    radius_px: float,
) -> float:
    """Compute D_i^t in Eq. (3) over visible projected proposal points."""
    if proposal_locations.size == 0:
        return 0.0
    visible = projection[proposal_locations, 2] == 1
    count = int(visible.sum())
    if count == 0:
        return 0.0
    yy = projection[proposal_locations[visible], 0].astype(np.float64)
    xx = projection[proposal_locations[visible], 1].astype(np.float64)
    inside = np.hypot(xx - point_x, yy - point_y) <= radius_px
    return float(inside.sum() / count)


def assign_observation(record: dict, support_ratio: float) -> int:
    """Resolve an observation to one proposal using distance and local support.

    The released checkpoint uses the deterministic overlap case: surface
    support resolves two proposals only when their nearest projected distances
    are numerically equal. ``support_ratio`` is the minimum D2/D1 ratio.
    """
    first = int(record["nearest_proposal"])
    second = int(record.get("second_proposal", -1))
    if (
        second >= 0
        and record.get("second_distance") is not None
        and record.get("second_support") is not None
        and float(record["second_distance"]) - float(record["nearest_distance"]) <= 1e-9
        and float(record["second_support"])
        / (float(record["nearest_support"]) + 1e-6)
        > support_ratio
    ):
        return second
    return first


def aggregate_generic_evidence(
    observations: Iterable[dict], number_of_proposals: int, support_ratio: float
) -> np.ndarray:
    """Compute generic proposal evidence V_i^g in Eq. (4)."""
    votes = np.zeros(number_of_proposals, dtype=np.float64)
    for observation in observations:
        proposal = assign_observation(observation, support_ratio)
        if 0 <= proposal < number_of_proposals:
            votes[proposal] += 1.0
    return votes


def select_proposal(votes: np.ndarray) -> int:
    """Eq. (5)/(12); returns -1 when no proposal receives evidence."""
    return int(np.argmax(votes)) if votes.size and float(votes.max()) > 0 else -1

