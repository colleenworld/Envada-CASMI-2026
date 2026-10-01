import numpy as np
from scipy import sparse
import pandas as pd

def search_structures_indexed_blockwise(
    query_vectors: sparse.csr_matrix,
    query_structure_ids: Sequence[int],
    query_modes: Sequence[str],
    reference_vectors: sparse.csr_matrix,
    reference_candidate_ids: Sequence[int],
    reference_modes: Sequence[str],
    query_structure_count: int,
    candidate_count: int,
    block_size: int = 25_000,
) -> np.ndarray:
    """
    Search multiple query structures against a reference index using
    integer structure/candidate IDs.

    Returns a dense score matrix with shape:

        (query_structure_count, candidate_count)

    Each cell contains the maximum cosine similarity across all
    same-polarity spectrum pairs for that query/candidate structure.

    Query and reference vectors must already be L2-normalized.
    """
    query_structure_ids = np.asarray(
        query_structure_ids,
        dtype=np.int64,
    )

    query_modes = np.asarray(
        query_modes,
        dtype=object,
    )

    reference_candidate_ids = np.asarray(
        reference_candidate_ids,
        dtype=np.int64,
    )

    reference_modes = np.asarray(
        reference_modes,
        dtype=object,
    )

    if query_vectors.shape[0] != len(
        query_structure_ids
    ):
        raise ValueError(
            "query_structure_ids length does not match "
            "number of query vectors"
        )

    if query_vectors.shape[0] != len(
        query_modes
    ):
        raise ValueError(
            "query_modes length does not match "
            "number of query vectors"
        )

    if reference_vectors.shape[0] != len(
        reference_candidate_ids
    ):
        raise ValueError(
            "reference_candidate_ids length does not match "
            "number of reference vectors"
        )

    if reference_vectors.shape[0] != len(
        reference_modes
    ):
        raise ValueError(
            "reference_modes length does not match "
            "number of reference vectors"
        )

    if query_vectors.shape[1] != reference_vectors.shape[1]:
        raise ValueError(
            "query and reference dimensions differ"
        )

    if query_structure_count <= 0:
        raise ValueError(
            "query_structure_count must be positive"
        )

    if candidate_count <= 0:
        raise ValueError(
            "candidate_count must be positive"
        )

    if len(query_structure_ids):
        if (
            query_structure_ids.min() < 0
            or query_structure_ids.max()
            >= query_structure_count
        ):
            raise ValueError(
                "query_structure_ids contains an invalid ID"
            )

    if len(reference_candidate_ids):
        if (
            reference_candidate_ids.min() < 0
            or reference_candidate_ids.max()
            >= candidate_count
        ):
            raise ValueError(
                "reference_candidate_ids contains an invalid ID"
            )

    scores = np.full(
        (
            query_structure_count,
            candidate_count,
        ),
        -np.inf,
        dtype=np.float32,
    )

    for mode in np.unique(query_modes):
        query_mode_indices = np.flatnonzero(
            query_modes == mode
        )

        reference_mode_indices = np.flatnonzero(
            reference_modes == mode
        )

        if (
            len(query_mode_indices) == 0
            or len(reference_mode_indices) == 0
        ):
            continue

        mode_query_vectors = query_vectors[
            query_mode_indices
        ]

        mode_query_structure_ids = (
            query_structure_ids[
                query_mode_indices
            ]
        )

        # A query structure can have multiple spectra of the same
        # polarity. Work out which rows belong to each structure once.
        unique_query_ids = np.unique(
            mode_query_structure_ids
        )

        for block_start in range(
            0,
            len(reference_mode_indices),
            block_size,
        ):
            block_stop = min(
                block_start + block_size,
                len(reference_mode_indices),
            )

            block_indices = reference_mode_indices[
                block_start:block_stop
            ]

            block_vectors = reference_vectors[
                block_indices
            ]

            block_candidate_ids = (
                reference_candidate_ids[
                    block_indices
                ]
            )

            similarities = (
                mode_query_vectors
                @ block_vectors.T
            ).toarray()

            for query_id in unique_query_ids:
                query_spectrum_mask = (
                    mode_query_structure_ids
                    == query_id
                )

                # Max over all spectra belonging to this query
                # structure.
                spectrum_scores = similarities[
                    query_spectrum_mask
                ].max(axis=0)

                # Aggregate all reference spectra belonging to the
                # same candidate directly into the global score row.
                np.maximum.at(
                    scores[query_id],
                    block_candidate_ids,
                    spectrum_scores,
                )

    return scores

def cosine_similarity_block(
    queries: sparse.csr_matrix,
    candidates: sparse.csr_matrix,
) -> np.ndarray:
    """
    Calculate cosine similarities between already L2-normalized
    sparse query and candidate spectra.

    Returns a dense matrix shaped:

        (number of queries, number of candidates)
    """
    if queries.shape[1] != candidates.shape[1]:
        raise ValueError(
            "query and candidate dimensions differ"
        )

    similarities = (
        queries @ candidates.T
    )

    return similarities.toarray()


def aggregate_query_structure_scores(
    similarities: np.ndarray,
    query_keys: np.ndarray,
    candidate_keys: np.ndarray,
) -> dict[str, dict[str, float]]:
    """
    Aggregate spectrum-level similarities to structure-level scores.

    For every query structure/candidate structure pair, retain the
    maximum similarity across all corresponding spectrum pairs.
    """
    if similarities.shape != (
        len(query_keys),
        len(candidate_keys),
    ):
        raise ValueError(
            "similarity matrix dimensions do not "
            "match query/candidate keys"
        )

    results: dict[
        str,
        dict[str, float],
    ] = {}

    unique_query_keys = np.unique(
        query_keys
    )

    for query_key in unique_query_keys:
        query_mask = (
            query_keys == query_key
        )

        # Maximum across spectra belonging to this query structure.
        candidate_spectrum_scores = (
            similarities[
                query_mask
            ].max(axis=0)
        )

        candidate_results = {}

        for candidate_key in np.unique(
            candidate_keys
        ):
            candidate_mask = (
                candidate_keys
                == candidate_key
            )

            candidate_results[
                str(candidate_key)
            ] = float(
                candidate_spectrum_scores[
                    candidate_mask
                ].max()
            )

        results[
            str(query_key)
        ] = candidate_results

    return results

from collections.abc import Sequence

def search_structure_blockwise(
    query_vectors: sparse.csr_matrix,
    query_modes: Sequence[str],
    reference_vectors: sparse.csr_matrix,
    reference_metadata: pd.DataFrame,
    block_size: int = 25_000,
) -> dict[str, float]:
    """
    Search one query structure against a reference spectral index.

    Query spectra are compared only with reference spectra having the
    same ionization mode.

    The score for a candidate structure is the maximum cosine
    similarity across every valid query/reference spectrum pair.

    Assumes query and reference vectors are already L2-normalized.
    """
    if query_vectors.shape[0] != len(query_modes):
        raise ValueError(
            "query_modes length does not match "
            "number of query vectors"
        )

    if reference_vectors.shape[0] != len(
        reference_metadata
    ):
        raise ValueError(
            "reference metadata length does not match "
            "number of reference vectors"
        )

    if query_vectors.shape[1] != reference_vectors.shape[1]:
        raise ValueError(
            "query and reference dimensions differ"
        )

    if "inchikey14" not in reference_metadata.columns:
        raise ValueError(
            "reference metadata must contain inchikey14"
        )

    if (
        "ionization_mode"
        not in reference_metadata.columns
    ):
        raise ValueError(
            "reference metadata must contain "
            "ionization_mode"
        )

    query_modes = np.asarray(
        query_modes,
        dtype=object,
    )

    candidate_scores: dict[str, float] = {}

    for start in range(
        0,
        len(reference_metadata),
        block_size,
    ):
        stop = min(
            start + block_size,
            len(reference_metadata),
        )

        block_metadata = (
            reference_metadata.iloc[start:stop]
        )

        block_vectors = (
            reference_vectors[start:stop]
        )

        # Work polarity by polarity so that invalid cross-polarity
        # comparisons are never scored.
        for mode in np.unique(query_modes):
            query_mask = (
                query_modes == mode
            )

            reference_mask = (
                block_metadata[
                    "ionization_mode"
                ].to_numpy()
                == mode
            )

            if not np.any(reference_mask):
                continue

            mode_queries = query_vectors[
                query_mask
            ]

            mode_references = block_vectors[
                reference_mask
            ]

            similarities = (
                mode_queries
                @ mode_references.T
            ).toarray()

            # Maximum over all query spectra from this structure.
            spectrum_scores = similarities.max(
                axis=0
            )

            candidate_keys = (
                block_metadata.loc[
                    reference_mask,
                    "inchikey14",
                ]
                .astype(str)
                .to_numpy()
            )

            # Aggregate spectra belonging to the same candidate
            # structure within this block.
            unique_keys, inverse = np.unique(
                candidate_keys,
                return_inverse=True,
            )

            block_scores = np.full(
                len(unique_keys),
                -np.inf,
                dtype=np.float32,
            )

            np.maximum.at(
                block_scores,
                inverse,
                spectrum_scores,
            )

            # Merge candidate maxima across reference blocks.
            for key, score in zip(
                unique_keys,
                block_scores,
                strict=True,
            ):
                key = str(key)
                score = float(score)

                previous = candidate_scores.get(
                    key
                )

                if (
                    previous is None
                    or score > previous
                ):
                    candidate_scores[key] = score

    return candidate_scores

def search_structures_blockwise(
    query_vectors: sparse.csr_matrix,
    query_structure_keys: Sequence[str],
    query_modes: Sequence[str],
    reference_vectors: sparse.csr_matrix,
    reference_metadata: pd.DataFrame,
    block_size: int = 25_000,
) -> dict[str, dict[str, float]]:
    """
    Search spectra from multiple query structures against the
    reference index.

    Comparisons are restricted to the same ionization mode.

    For every query-structure / candidate-structure pair, retain
    the maximum cosine similarity across all valid spectrum pairs.

    Query and reference vectors must already be L2-normalized.
    """
    query_structure_keys = np.asarray(
        query_structure_keys,
        dtype=object,
    )

    query_modes = np.asarray(
        query_modes,
        dtype=object,
    )

    if query_vectors.shape[0] != len(
        query_structure_keys
    ):
        raise ValueError(
            "query_structure_keys length does not "
            "match number of query vectors"
        )

    if query_vectors.shape[0] != len(
        query_modes
    ):
        raise ValueError(
            "query_modes length does not match "
            "number of query vectors"
        )

    if reference_vectors.shape[0] != len(
        reference_metadata
    ):
        raise ValueError(
            "reference metadata length does not match "
            "number of reference vectors"
        )

    if (
        query_vectors.shape[1]
        != reference_vectors.shape[1]
    ):
        raise ValueError(
            "query and reference dimensions differ"
        )

    required_columns = {
        "inchikey14",
        "ionization_mode",
    }

    missing_columns = (
        required_columns
        - set(reference_metadata.columns)
    )

    if missing_columns:
        raise ValueError(
            "reference metadata missing columns: "
            + ", ".join(
                sorted(missing_columns)
            )
        )

    unique_query_keys = np.unique(
        query_structure_keys
    )

    results: dict[
        str,
        dict[str, float],
    ] = {
        str(key): {}
        for key in unique_query_keys
    }

    # Work polarity-by-polarity. This avoids repeatedly checking
    # every reference spectrum's polarity for every query
    # structure.
    for mode in np.unique(
        query_modes
    ):
        mode_query_mask = (
            query_modes == mode
        )

        mode_query_vectors = (
            query_vectors[
                mode_query_mask
            ]
        )

        mode_query_keys = (
            query_structure_keys[
                mode_query_mask
            ]
        )

        reference_mode_mask = (
            reference_metadata[
                "ionization_mode"
            ]
            .astype(str)
            .to_numpy()
            == str(mode)
        )

        reference_mode_indices = (
            np.flatnonzero(
                reference_mode_mask
            )
        )

        for block_start in range(
            0,
            len(reference_mode_indices),
            block_size,
        ):
            block_stop = min(
                block_start + block_size,
                len(reference_mode_indices),
            )

            block_indices = (
                reference_mode_indices[
                    block_start:block_stop
                ]
            )

            block_vectors = (
                reference_vectors[
                    block_indices
                ]
            )

            block_keys = (
                reference_metadata.iloc[
                    block_indices
                ][
                    "inchikey14"
                ]
                .astype(str)
                .to_numpy()
            )

            similarities = (
                mode_query_vectors
                @ block_vectors.T
            ).toarray()

            # Aggregate independently for each query structure.
            for query_key in np.unique(
                mode_query_keys
            ):
                query_mask = (
                    mode_query_keys
                    == query_key
                )

                # Max over all query spectra belonging to this
                # structure.
                spectrum_scores = (
                    similarities[
                        query_mask
                    ].max(axis=0)
                )

                unique_candidate_keys, inverse = (
                    np.unique(
                        block_keys,
                        return_inverse=True,
                    )
                )

                block_scores = np.full(
                    len(
                        unique_candidate_keys
                    ),
                    -np.inf,
                    dtype=np.float32,
                )

                np.maximum.at(
                    block_scores,
                    inverse,
                    spectrum_scores,
                )

                query_results = results[
                    str(query_key)
                ]

                for candidate_key, score in zip(
                    unique_candidate_keys,
                    block_scores,
                    strict=True,
                ):
                    candidate_key = str(
                        candidate_key
                    )

                    score = float(
                        score
                    )

                    previous = (
                        query_results.get(
                            candidate_key
                        )
                    )

                    if (
                        previous is None
                        or score > previous
                    ):
                        query_results[
                            candidate_key
                        ] = score

    return results