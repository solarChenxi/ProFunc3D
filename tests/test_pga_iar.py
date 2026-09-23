import numpy as np

from profunc3d.iar import frame_completeness, predict, sigmoid_quality_weights
from profunc3d.pga import aggregate_generic_evidence


def observation(frame, proposal, second=-1):
    return {
        "video_id": "video-1",
        "frame_id": str(frame),
        "nearest_proposal": proposal,
        "nearest_distance": 2.0,
        "nearest_support": 0.1,
        "second_proposal": second,
        "second_distance": None,
        "second_support": None,
    }


def test_pga_vote_aggregation():
    votes = aggregate_generic_evidence(
        [observation(0.0, 2), observation(1.0, 2), observation(2.0, 1)], 4, 1.30
    )
    assert votes.tolist() == [0.0, 1.0, 2.0, 0.0]


def test_iar_suppresses_non_track_observation():
    observations = [observation(1.0, 0), observation(2.0, 1), observation(4.0, 1)]
    track = [{"video_id": "video-1", "frame_id": "1.0", "quality": 0.8}]
    result = predict(
        observations=observations,
        number_of_proposals=2,
        relational=True,
        selected_track_detections=track,
        temporal_window_s=0.1,
    )
    assert result["generic_prediction"] == 1
    assert result["selected_proposal"] == 0
    assert result["route"] == "PGA + IAR"


def test_empty_alignment_falls_back_to_pga():
    observations = [observation(1.0, 1)]
    result = predict(
        observations=observations,
        number_of_proposals=2,
        relational=True,
        selected_track_detections=[],
    )
    assert result["selected_proposal"] == 1
    assert "fallback" in result["route"]


def test_quality_and_completeness_are_finite():
    mask = np.zeros((100, 100), dtype=bool)
    mask[20:80, 20:80] = True
    stats = frame_completeness(mask)
    assert stats["completeness"] > 0
    weights = sigmoid_quality_weights(np.asarray([0.1, 1.0]))
    assert np.isfinite(weights).all()
    assert weights[1] > weights[0]

