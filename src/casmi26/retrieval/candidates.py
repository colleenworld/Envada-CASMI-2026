from collections.abc import Iterable

import numpy as np

from casmi26.retrieval.mass_index import (
    NeutralMassIndex,
    search_neutral_masses,
)

def top_k_candidate_ids(
    scores: np.ndarray,
    k: int,
) -> np.ndarray:
    """
    Return candidate structure IDs for the k highest finite scores.

    Candidate IDs correspond to positions in the score array.
    Results are ordered from highest to lowest score.

    Ties are resolved by candidate ID to make the result
    deterministic.
    """
    scores = np.asarray(
        scores,
        dtype=np.float64,
    ).reshape(-1)

    if k < 0:
        raise ValueError(
            "k must be non-negative"
        )

    if (
        k == 0
        or len(scores) == 0
    ):
        return np.empty(
            0,
            dtype=np.int64,
        )

    candidate_ids = np.flatnonzero(
        np.isfinite(scores)
    ).astype(
        np.int64,
        copy=False,
    )

    if len(candidate_ids) == 0:
        return np.empty(
            0,
            dtype=np.int64,
        )

    candidate_scores = scores[
        candidate_ids
    ]

    # np.lexsort uses the final key as the primary key.
    # Primary: descending score.
    # Secondary: ascending candidate ID.
    order = np.lexsort(
        (
            candidate_ids,
            -candidate_scores,
        )
    )

    return candidate_ids[
        order[:k]
    ]

def opposite_polarity_mass_candidates(
    index: NeutralMassIndex,
    neutral_masses: Iterable[float] | np.ndarray,
    modes: Iterable[str] | np.ndarray,
    reference_candidate_ids: np.ndarray,
    tolerance_da: float,
) -> np.ndarray:
    """
    Return unique candidate structure IDs whose reference spectra
    have a compatible neutral mass in the opposite ionization mode
    from at least one query spectrum.
    """
    masses = np.asarray(
        list(neutral_masses)
        if not isinstance(neutral_masses, np.ndarray)
        else neutral_masses,
        dtype=np.float64,
    ).reshape(-1)

    query_modes = np.asarray(
        list(modes)
        if not isinstance(modes, np.ndarray)
        else modes,
        dtype=object,
    ).reshape(-1)

    opposite_modes = np.array(
        [
            (
                "negative"
                if str(mode) == "positive"
                else "positive"
                if str(mode) == "negative"
                else str(mode)
            )
            for mode in query_modes
        ],
        dtype=object,
    )

    reference_rows = search_neutral_masses(
        index=index,
        neutral_masses=masses,
        modes=opposite_modes,
        tolerance_da=tolerance_da,
    )

    return reference_rows_to_candidate_ids(
        reference_rows,
        reference_candidate_ids,
    )

def same_polarity_mass_candidates(
    index: NeutralMassIndex,
    neutral_masses: Iterable[float] | np.ndarray,
    modes: Iterable[str] | np.ndarray,
    reference_candidate_ids: np.ndarray,
    tolerance_da: float,
) -> np.ndarray:
    """
    Return unique candidate structure IDs whose reference spectra
    have a compatible neutral mass in the same ionization mode as
    at least one query spectrum.
    """
    masses = np.asarray(
        list(neutral_masses)
        if not isinstance(neutral_masses, np.ndarray)
        else neutral_masses,
        dtype=np.float64,
    ).reshape(-1)

    query_modes = np.asarray(
        list(modes)
        if not isinstance(modes, np.ndarray)
        else modes,
        dtype=object,
    ).reshape(-1)

    reference_rows = search_neutral_masses(
        index=index,
        neutral_masses=masses,
        modes=query_modes,
        tolerance_da=tolerance_da,
    )

    return reference_rows_to_candidate_ids(
        reference_rows,
        reference_candidate_ids,
    )

def reference_rows_to_candidate_ids(
    reference_rows: Iterable[int] | np.ndarray,
    reference_candidate_ids: np.ndarray,
) -> np.ndarray:
    """
    Convert reference-spectrum row IDs into unique candidate
    structure IDs.

    Multiple reference spectra may belong to the same structure,
    so the returned candidate IDs are deduplicated and sorted.
    """
    rows = np.asarray(
        list(reference_rows)
        if not isinstance(reference_rows, np.ndarray)
        else reference_rows,
        dtype=np.int64,
    ).reshape(-1)

    if len(rows) == 0:
        return np.empty(
            0,
            dtype=np.int64,
        )

    return np.unique(
        reference_candidate_ids[rows]
    )

def merge_candidate_ids(
    *candidate_sets: Iterable[int] | np.ndarray,
) -> np.ndarray:
    """
    Return the sorted union of candidate structure IDs.

    Candidate sources may contain duplicate IDs. Empty candidate
    sources are allowed.
    """
    arrays: list[np.ndarray] = []

    for candidates in candidate_sets:
        array = np.asarray(
            list(candidates)
            if not isinstance(candidates, np.ndarray)
            else candidates,
            dtype=np.int64,
        ).reshape(-1)

        if len(array):
            arrays.append(array)

    if not arrays:
        return np.empty(
            0,
            dtype=np.int64,
        )

    return np.unique(
        np.concatenate(arrays)
    )