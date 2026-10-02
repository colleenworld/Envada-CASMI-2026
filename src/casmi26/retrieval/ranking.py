from collections.abc import Iterable

import numpy as np

def candidate_scores(
    candidate_ids: Iterable[int] | np.ndarray,
    scores: np.ndarray,
) -> np.ndarray:
    """
    Extract scores for a candidate subset from a dense candidate
    score vector.

    Candidate IDs are positions in the full score vector.

    Candidates with no usable retrieval score retain the value
    stored in scores, normally -inf.
    """
    candidate_ids = np.asarray(
        (
            candidate_ids
            if isinstance(
                candidate_ids,
                np.ndarray,
            )
            else list(
                candidate_ids
            )
        ),
        dtype=np.int64,
    ).reshape(-1)

    scores = np.asarray(
        scores,
        dtype=np.float32,
    ).reshape(-1)

    if len(candidate_ids) == 0:
        return np.empty(
            0,
            dtype=np.float32,
        )

    if np.any(
        candidate_ids < 0
    ):
        raise ValueError(
            "candidate IDs must be non-negative"
        )

    if np.any(
        candidate_ids >= len(scores)
    ):
        raise ValueError(
            "candidate ID exceeds score vector"
        )

    return scores[
        candidate_ids
    ]

def candidate_membership(
    candidate_ids: np.ndarray,
    members: Iterable[int] | np.ndarray,
) -> np.ndarray:
    """
    Return a boolean array indicating whether each candidate ID
    belongs to the supplied candidate set.
    """
    candidate_ids = np.asarray(
        candidate_ids,
        dtype=np.int64,
    ).reshape(-1)

    members_array = np.asarray(
        (
            members
            if isinstance(members, np.ndarray)
            else list(members)
        ),
        dtype=np.int64,
    ).reshape(-1)

    if len(candidate_ids) == 0:
        return np.empty(
            0,
            dtype=bool,
        )

    if len(members_array) == 0:
        return np.zeros(
            len(candidate_ids),
            dtype=bool,
        )

    return np.isin(
        candidate_ids,
        members_array,
    )


def candidate_spectral_ranks(
    candidate_ids: np.ndarray,
    spectral_candidate_ids: Iterable[int] | np.ndarray,
) -> np.ndarray:
    """
    Return each candidate's 1-based spectral rank.

    Candidates absent from the spectral candidate list receive NaN.

    The order of spectral_candidate_ids is significant: the first
    candidate has rank 1, the second rank 2, etc.
    """
    candidate_ids = np.asarray(
        candidate_ids,
        dtype=np.int64,
    ).reshape(-1)

    spectral_ids = np.asarray(
        (
            spectral_candidate_ids
            if isinstance(
                spectral_candidate_ids,
                np.ndarray,
            )
            else list(
                spectral_candidate_ids
            )
        ),
        dtype=np.int64,
    ).reshape(-1)

    ranks = np.full(
        len(candidate_ids),
        np.nan,
        dtype=np.float64,
    )

    if (
        len(candidate_ids) == 0
        or len(spectral_ids) == 0
    ):
        return ranks

    rank_by_candidate = {
        int(candidate_id): rank
        for rank, candidate_id
        in enumerate(
            spectral_ids,
            start=1,
        )
    }

    for index, candidate_id in enumerate(
        candidate_ids
    ):
        rank = rank_by_candidate.get(
            int(candidate_id)
        )

        if rank is not None:
            ranks[index] = rank

    return ranks


def build_candidate_features(
    hybrid_candidate_ids: Iterable[int] | np.ndarray,
    same_candidate_ids: Iterable[int] | np.ndarray,
    opposite_candidate_ids: Iterable[int] | np.ndarray,
    spectral_candidate_ids: Iterable[int] | np.ndarray,
    scores: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """
    Build ranking features for one query structure.

    Candidate order follows hybrid_candidate_ids.
    """
    hybrid_ids = np.asarray(
        (
            hybrid_candidate_ids
            if isinstance(
                hybrid_candidate_ids,
                np.ndarray,
            )
            else list(
                hybrid_candidate_ids
            )
        ),
        dtype=np.int64,
    ).reshape(-1)

    features = {
        "candidate_id": hybrid_ids,
        "in_same_mass": candidate_membership(
            hybrid_ids,
            same_candidate_ids,
        ),
        "in_opposite_mass": candidate_membership(
            hybrid_ids,
            opposite_candidate_ids,
        ),
        "in_spectral_top25": candidate_membership(
            hybrid_ids,
            spectral_candidate_ids,
        ),
        "spectral_rank": candidate_spectral_ranks(
            hybrid_ids,
            spectral_candidate_ids,
        ),
    }

    if scores is not None:
        features[
            "same_polarity_cosine"
        ] = candidate_scores(
            hybrid_ids,
            scores,
        )

    return features