import numpy as np
from numpy.typing import NDArray


def aggregate_max_by_candidate(
    spectrum_scores: NDArray[np.floating],
    candidate_keys: NDArray[np.str_],
) -> tuple[NDArray[np.str_], NDArray[np.float32]]:
    """
    Aggregate spectrum-level similarity scores into molecule-level scores.

    A molecule's score is the maximum score among all of its reference
    spectra.
    """
    if spectrum_scores.ndim != 1:
        raise ValueError("spectrum_scores must be one-dimensional")

    if candidate_keys.ndim != 1:
        raise ValueError("candidate_keys must be one-dimensional")

    if len(spectrum_scores) != len(candidate_keys):
        raise ValueError(
            "spectrum_scores and candidate_keys must have the same length"
        )

    unique_keys, groups = np.unique(
        candidate_keys,
        return_inverse=True,
    )

    scores = np.full(
        len(unique_keys),
        -np.inf,
        dtype=np.float32,
    )

    np.maximum.at(
        scores,
        groups,
        spectrum_scores,
    )

    return unique_keys, scores


def merge_candidate_scores(
    existing: dict[str, float],
    candidate_keys: NDArray[np.str_],
    scores: NDArray[np.floating],
) -> None:
    """
    Merge candidate scores using maximum similarity.

    Mutates existing in place.
    """
    if len(candidate_keys) != len(scores):
        raise ValueError(
            "candidate_keys and scores must have the same length"
        )

    for key, score in zip(candidate_keys, scores, strict=True):
        key = str(key)
        value = float(score)

        existing[key] = max(
            existing.get(key, -np.inf),
            value,
        )