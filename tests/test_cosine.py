import numpy as np
import pytest

from casmi26.retrieval.cosine import cosine_search


def test_returns_most_similar_candidate_first():
    query = np.array([1.0, 0.0], dtype=np.float32)

    candidates = np.array(
        [
            [0.0, 1.0],
            [1.0, 0.0],
            [0.8, 0.6],
        ],
        dtype=np.float32,
    )

    results = cosine_search(query, candidates, k=3)

    assert [r.index for r in results] == [1, 2, 0]


def test_returns_scores():
    query = np.array([1.0, 0.0], dtype=np.float32)

    candidates = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )

    results = cosine_search(query, candidates, k=2)

    assert results[0].score == pytest.approx(1.0)
    assert results[1].score == pytest.approx(0.0)


def test_k_larger_than_candidates():
    query = np.array([1.0, 0.0], dtype=np.float32)

    candidates = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )

    results = cosine_search(query, candidates, k=25)

    assert len(results) == 2


def test_empty_candidates():
    query = np.array([1.0, 0.0], dtype=np.float32)
    candidates = np.empty((0, 2), dtype=np.float32)

    assert cosine_search(query, candidates) == []


def test_rejects_dimension_mismatch():
    query = np.array([1.0, 0.0], dtype=np.float32)
    candidates = np.zeros((3, 4), dtype=np.float32)

    with pytest.raises(
        ValueError,
        match="same vector dimension",
    ):
        cosine_search(query, candidates)