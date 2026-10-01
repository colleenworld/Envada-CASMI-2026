import numpy as np
import pytest

from casmi26.retrieval.chunked import (
    iter_chunks,
    search_chunks,
)


def test_iter_chunks():
    vectors = np.arange(
        20,
        dtype=np.float32,
    ).reshape(5, 4)

    keys = np.array(
        ["A", "B", "C", "D", "E"],
    )

    chunks = list(
        iter_chunks(
            vectors,
            keys,
            chunk_size=2,
        )
    )

    assert len(chunks) == 3
    assert len(chunks[0][0]) == 2
    assert len(chunks[1][0]) == 2
    assert len(chunks[2][0]) == 1


def test_search_chunks_returns_best_score_per_molecule():
    query = np.array(
        [1.0, 0.0],
        dtype=np.float32,
    )

    vectors = np.array(
        [
            [0.2, 0.0],
            [0.8, 0.0],
            [0.5, 0.0],
            [0.9, 0.0],
        ],
        dtype=np.float32,
    )

    keys = np.array(
        ["A", "B", "A", "C"],
    )

    result = search_chunks(
        query,
        iter_chunks(
            vectors,
            keys,
            chunk_size=2,
        ),
    )

    assert result["A"] == pytest.approx(0.5)
    assert result["B"] == pytest.approx(0.8)
    assert result["C"] == pytest.approx(0.9)


def test_same_molecule_across_chunks_uses_maximum():
    query = np.array(
        [1.0, 0.0],
        dtype=np.float32,
    )

    vectors = np.array(
        [
            [0.3, 0.0],
            [0.2, 0.0],
            [0.9, 0.0],
        ],
        dtype=np.float32,
    )

    keys = np.array(
        ["A", "B", "A"],
    )

    result = search_chunks(
        query,
        iter_chunks(
            vectors,
            keys,
            chunk_size=2,
        ),
    )

    assert result["A"] == pytest.approx(0.9)


def test_chunk_size_larger_than_input():
    vectors = np.zeros(
        (3, 2),
        dtype=np.float32,
    )

    keys = np.array(["A", "B", "C"])

    chunks = list(
        iter_chunks(
            vectors,
            keys,
            chunk_size=100,
        )
    )

    assert len(chunks) == 1
    assert len(chunks[0][0]) == 3


def test_rejects_invalid_chunk_size():
    with pytest.raises(
        ValueError,
        match="chunk_size",
    ):
        list(
            iter_chunks(
                np.zeros((2, 2), dtype=np.float32),
                np.array(["A", "B"]),
                chunk_size=0,
            )
        )


def test_rejects_different_lengths():
    with pytest.raises(
        ValueError,
        match="same length",
    ):
        list(
            iter_chunks(
                np.zeros((2, 2), dtype=np.float32),
                np.array(["A"]),
                chunk_size=1,
            )
        )