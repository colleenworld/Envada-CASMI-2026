import numpy as np
import pytest

from casmi26.retrieval.aggregation import (
    aggregate_max_by_candidate,
    merge_candidate_scores,
)


def test_aggregate_max_by_candidate():
    keys = np.array(
        ["A", "A", "B", "B", "C"],
    )

    spectrum_scores = np.array(
        [0.2, 0.8, 0.7, 0.4, 0.3],
        dtype=np.float32,
    )

    candidate_keys, scores = aggregate_max_by_candidate(
        spectrum_scores,
        keys,
    )

    result = dict(zip(candidate_keys, scores, strict=True))

    assert result["A"] == pytest.approx(0.8)
    assert result["B"] == pytest.approx(0.7)
    assert result["C"] == pytest.approx(0.3)


def test_aggregate_rejects_different_lengths():
    with pytest.raises(
        ValueError,
        match="same length",
    ):
        aggregate_max_by_candidate(
            np.array([0.1, 0.2]),
            np.array(["A"]),
        )


def test_merge_candidate_scores():
    existing = {
        "A": 0.5,
        "B": 0.7,
    }

    merge_candidate_scores(
        existing,
        np.array(["A", "B", "C"]),
        np.array(
            [0.8, 0.6, 0.4],
            dtype=np.float32,
        ),
    )

    assert existing["A"] == pytest.approx(0.8)
    assert existing["B"] == pytest.approx(0.7)
    assert existing["C"] == pytest.approx(0.4)


def test_merge_rejects_different_lengths():
    with pytest.raises(
        ValueError,
        match="same length",
    ):
        merge_candidate_scores(
            {},
            np.array(["A", "B"]),
            np.array([0.5]),
        )