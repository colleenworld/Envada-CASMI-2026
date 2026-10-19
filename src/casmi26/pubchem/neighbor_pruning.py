from __future__ import annotations

import numpy as np
from rdkit import DataStructs


def score_candidates(
    candidate_fps: list,
    neighbors: list[
        tuple[
            str,
            float,
        ]
    ],
    training_fps: dict,
) -> tuple[
    np.ndarray,
    int,
]:
    """
    Score candidate fingerprints against structurally similar training
    neighbors, weighted by spectral similarity.

    The score for each candidate is the maximum weighted Tanimoto
    similarity across usable training neighbors.
    """
    scores = np.zeros(
        len(candidate_fps),
        dtype=np.float32,
    )

    usable_neighbors = 0

    for (
        neighbor_key,
        spectral_score,
    ) in neighbors:
        neighbor_fp = (
            training_fps.get(
                neighbor_key
            )
        )

        if neighbor_fp is None:
            continue

        usable_neighbors += 1

        similarities = np.asarray(
            DataStructs.BulkTanimotoSimilarity(
                neighbor_fp,
                candidate_fps,
            ),
            dtype=np.float32,
        )

        weighted = (
            similarities
            * max(
                float(
                    spectral_score
                ),
                0.0,
            )
        )

        np.maximum(
            scores,
            weighted,
            out=scores,
        )

    return (
        scores,
        usable_neighbors,
    )
