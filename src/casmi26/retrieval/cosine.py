from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class SearchResult:
    index: int
    score: float


def cosine_search(
    query: NDArray[np.float32],
    candidates: NDArray[np.float32],
    *,
    k: int = 25,
) -> list[SearchResult]:
    """
    Find the k candidate vectors with highest cosine similarity.

    Both query and candidate vectors are expected to already be
    L2-normalized.
    """
    if query.ndim != 1:
        raise ValueError("query must be one-dimensional")

    if candidates.ndim != 2:
        raise ValueError("candidates must be two-dimensional")

    if candidates.shape[1] != query.shape[0]:
        raise ValueError(
            "query and candidates must have the same vector dimension"
        )

    if k <= 0:
        raise ValueError("k must be positive")

    if len(candidates) == 0:
        return []

    scores = candidates @ query

    actual_k = min(k, len(scores))

    if actual_k == len(scores):
        indices = np.argsort(scores)[::-1]
    else:
        partition = np.argpartition(
            scores,
            -actual_k,
        )[-actual_k:]

        indices = partition[
            np.argsort(scores[partition])[::-1]
        ]

    return [
        SearchResult(
            index=int(index),
            score=float(scores[index]),
        )
        for index in indices
    ]