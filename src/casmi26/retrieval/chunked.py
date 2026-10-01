from collections.abc import Iterator

import numpy as np
from numpy.typing import NDArray

from casmi26.retrieval.aggregation import (
    aggregate_max_by_candidate,
    merge_candidate_scores,
)


def iter_chunks(
    vectors: NDArray[np.float32],
    candidate_keys: NDArray[np.str_],
    *,
    chunk_size: int,
) -> Iterator[
    tuple[
        NDArray[np.float32],
        NDArray[np.str_],
    ]
]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    if len(vectors) != len(candidate_keys):
        raise ValueError(
            "vectors and candidate_keys must have the same length"
        )

    for start in range(0, len(vectors), chunk_size):
        end = min(
            start + chunk_size,
            len(vectors),
        )

        yield (
            vectors[start:end],
            candidate_keys[start:end],
        )


def search_chunks(
    query: NDArray[np.float32],
    chunks: Iterator[
        tuple[
            NDArray[np.float32],
            NDArray[np.str_],
        ]
    ],
) -> dict[str, float]:
    """
    Search candidate-vector chunks and return the maximum cosine
    similarity observed for each candidate molecule.

    Query and candidate vectors must already be L2-normalized.
    """
    if query.ndim != 1:
        raise ValueError("query must be one-dimensional")

    result: dict[str, float] = {}

    for candidate_vectors, candidate_keys in chunks:
        if candidate_vectors.ndim != 2:
            raise ValueError(
                "candidate vectors must be two-dimensional"
            )

        if candidate_vectors.shape[1] != query.shape[0]:
            raise ValueError(
                "query and candidates must have the same vector dimension"
            )

        spectrum_scores = candidate_vectors @ query

        keys, scores = aggregate_max_by_candidate(
            spectrum_scores,
            candidate_keys,
        )

        merge_candidate_scores(
            result,
            keys,
            scores,
        )

    return result