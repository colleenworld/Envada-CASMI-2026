from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from casmi26.chemistry.adducts import neutral_mass
from casmi26.retrieval.index import load_retrieval_index


INDEX_DIR = Path(
    "data/processed/retrieval_index"
)

MANIFEST_PATH = Path(
    "data/processed/splits/retrieval_dev.parquet"
)

BASELINE2_PATH = Path(
    "data/processed/results/"
    "retrieval_dev_baseline2_frozen.parquet"
)

TOLERANCE_DA = 0.01


def infer_neutral_masses(
    precursor_mzs: np.ndarray,
    adducts: np.ndarray,
) -> np.ndarray:
    result = np.full(
        len(precursor_mzs),
        np.nan,
        dtype=np.float64,
    )

    for index, (
        precursor_mz,
        adduct,
    ) in enumerate(
        zip(
            precursor_mzs,
            adducts,
            strict=True,
        )
    ):
        if pd.isna(
            precursor_mz
        ):
            continue

        mass = neutral_mass(
            float(precursor_mz),
            str(adduct),
        )

        if mass is not None:
            result[index] = mass

    return result


def build_reference_mask(
    metadata: pd.DataFrame,
    manifest: pd.DataFrame,
) -> np.ndarray:
    """
    Reproduce the frozen retrieval-dev reference pool.
    """
    query_pairs = set(
        zip(
            manifest[
                "inchikey14"
            ].astype(str),
            manifest[
                "query_library"
            ].astype(str),
            strict=True,
        )
    )

    return np.fromiter(
        (
            (
                str(key),
                str(library),
            )
            not in query_pairs
            for key, library in zip(
                metadata[
                    "inchikey14"
                ],
                metadata[
                    "ingest_lib"
                ],
                strict=True,
            )
        ),
        dtype=bool,
        count=len(metadata),
    )


def cosine_scores(
    query_vectors: sparse.csr_matrix,
    reference_vectors: sparse.csr_matrix,
) -> np.ndarray:
    """
    Maximum cosine similarity for each reference spectrum over
    all supplied query spectra.

    Retrieval vectors are already L2-normalized, so sparse matrix
    multiplication gives cosine similarity.
    """
    if query_vectors.shape[0] == 0:
        return np.empty(
            reference_vectors.shape[0],
            dtype=np.float32,
        )

    if reference_vectors.shape[0] == 0:
        return np.empty(
            0,
            dtype=np.float32,
        )

    similarities = (
        query_vectors
        @ reference_vectors.T
    )

    if sparse.issparse(
        similarities
    ):
        similarities = (
            similarities.toarray()
        )

    similarities = np.asarray(
        similarities,
        dtype=np.float32,
    )

    return np.max(
        similarities,
        axis=0,
    )


def rank_truth(
    query_vectors: sparse.csr_matrix,
    query_modes: np.ndarray,
    query_masses: np.ndarray,
    reference_vectors: sparse.csr_matrix,
    reference_modes: np.ndarray,
    reference_masses: np.ndarray,
    reference_candidate_ids: np.ndarray,
    truth_candidate_id: int,
    candidate_count: int,
) -> tuple[
    int | None,
    int,
    bool,
]:
    """
    Reproduce Baseline 2 for one query structure.

    Returns:
        truth rank
        number of mass-filtered reference rows
        whether fallback was used
    """
    usable_query_mass = (
        np.isfinite(
            query_masses
        )
    )

    known_query_mass_count = int(
        usable_query_mass.sum()
    )

    selected_reference_rows: set[
        int
    ] = set()

    if known_query_mass_count > 0:
        for (
            query_mode,
            query_mass,
        ) in zip(
            query_modes[
                usable_query_mass
            ],
            query_masses[
                usable_query_mass
            ],
            strict=True,
        ):
            matches = np.flatnonzero(
                (
                    reference_modes
                    == query_mode
                )
                & np.isfinite(
                    reference_masses
                )
                & (
                    np.abs(
                        reference_masses
                        - query_mass
                    )
                    <= TOLERANCE_DA
                )
            )

            selected_reference_rows.update(
                int(row)
                for row in matches
            )

    mass_filtered_row_count = len(
        selected_reference_rows
    )

    used_fallback = (
        known_query_mass_count == 0
        or mass_filtered_row_count == 0
    )

    if used_fallback:
        # Baseline 2 fallback:
        # full reference pool, while the actual cosine search
        # remains same-polarity.
        selected_rows = np.arange(
            len(reference_modes),
            dtype=np.int64,
        )
    else:
        selected_rows = np.asarray(
            sorted(
                selected_reference_rows
            ),
            dtype=np.int64,
        )

    candidate_scores = np.full(
        candidate_count,
        -np.inf,
        dtype=np.float32,
    )

    # ---------------------------------------------------------
    # Same-polarity search.
    #
    # A query structure can have spectra in more than one mode,
    # so handle each mode separately and merge candidate maxima.
    # ---------------------------------------------------------

    for mode in np.unique(
        query_modes
    ):
        query_rows = np.flatnonzero(
            query_modes
            == mode
        )

        mode_reference_rows = (
            selected_rows[
                reference_modes[
                    selected_rows
                ]
                == mode
            ]
        )

        if (
            len(query_rows) == 0
            or len(
                mode_reference_rows
            ) == 0
        ):
            continue

        spectrum_scores = cosine_scores(
            query_vectors[
                query_rows
            ],
            reference_vectors[
                mode_reference_rows
            ],
        )

        candidate_ids = (
            reference_candidate_ids[
                mode_reference_rows
            ]
        )

        np.maximum.at(
            candidate_scores,
            candidate_ids,
            spectrum_scores,
        )

    truth_score = candidate_scores[
        truth_candidate_id
    ]

    if not np.isfinite(
        truth_score
    ):
        return (
            None,
            mass_filtered_row_count,
            used_fallback,
        )

    # This is deliberately the exact Baseline 2 rank definition:
    #
    #     1 + number of candidates scoring strictly higher
    #
    # Do NOT introduce candidate-ID tie breaking here.
    rank = 1 + int(
        np.sum(
            candidate_scores
            > truth_score
        )
    )

    return (
        rank,
        mass_filtered_row_count,
        used_fallback,
    )


def main() -> None:
    print(
        "Loading retrieval index..."
    )

    metadata, vectors = (
        load_retrieval_index(
            INDEX_DIR
        )
    )

    manifest = pd.read_parquet(
        MANIFEST_PATH
    )

    baseline2 = pd.read_parquet(
        BASELINE2_PATH
    )

    # ---------------------------------------------------------
    # Verify manifest / frozen result alignment.
    # ---------------------------------------------------------

    if len(manifest) != 2_000:
        raise RuntimeError(
            "Expected 2,000 retrieval-dev structures."
        )

    if len(baseline2) != 2_000:
        raise RuntimeError(
            "Expected 2,000 frozen Baseline 2 rows."
        )

    # ---------------------------------------------------------
    # Rebuild frozen reference pool.
    # ---------------------------------------------------------

    reference_mask = build_reference_mask(
        metadata,
        manifest,
    )

    reference_indices = np.flatnonzero(
        reference_mask
    )

    removed = (
        len(metadata)
        - len(reference_indices)
    )

    if removed != 19_812:
        raise RuntimeError(
            "Reference pool does not match "
            "the frozen retrieval benchmark."
        )

    reference_metadata = (
        metadata.iloc[
            reference_indices
        ]
        .reset_index(
            drop=True
        )
    )

    reference_vectors = vectors[
        reference_indices
    ]

    reference_keys = (
        reference_metadata[
            "inchikey14"
        ]
        .astype(str)
        .to_numpy()
    )

    (
        candidate_keys,
        reference_candidate_ids,
    ) = np.unique(
        reference_keys,
        return_inverse=True,
    )

    reference_candidate_ids = (
        reference_candidate_ids.astype(
            np.int64,
            copy=False,
        )
    )

    candidate_count = len(
        candidate_keys
    )

    candidate_id_by_key = {
        key: index
        for index, key in enumerate(
            candidate_keys
        )
    }

    reference_modes = (
        reference_metadata[
            "ionization_mode"
        ]
        .astype(str)
        .to_numpy()
    )

    reference_masses = (
        infer_neutral_masses(
            reference_metadata[
                "precursor_mz"
            ].to_numpy(),
            reference_metadata[
                "adduct"
            ].to_numpy(),
        )
    )

    # ---------------------------------------------------------
    # Locate query spectra.
    # ---------------------------------------------------------

    metadata_keys = (
        metadata[
            "inchikey14"
        ]
        .astype(str)
        .to_numpy()
    )

    metadata_libraries = (
        metadata[
            "ingest_lib"
        ]
        .astype(str)
        .to_numpy()
    )

    reproduced = 0
    mismatches = []

    fallback_count = 0

    print()
    print(
        "Validating Baseline 2..."
    )

    for structure_index, row in enumerate(
        manifest.itertuples(
            index=False
        )
    ):
        truth_key = str(
            row.inchikey14
        )

        query_library = str(
            row.query_library
        )

        query_indices = np.flatnonzero(
            (
                metadata_keys
                == truth_key
            )
            & (
                metadata_libraries
                == query_library
            )
        )

        if len(query_indices) == 0:
            raise RuntimeError(
                "No query spectra for "
                f"{truth_key} / "
                f"{query_library}"
            )

        query_metadata = (
            metadata.iloc[
                query_indices
            ]
        )

        query_modes = (
            query_metadata[
                "ionization_mode"
            ]
            .astype(str)
            .to_numpy()
        )

        query_masses = (
            infer_neutral_masses(
                query_metadata[
                    "precursor_mz"
                ].to_numpy(),
                query_metadata[
                    "adduct"
                ].to_numpy(),
            )
        )

        truth_candidate_id = (
            candidate_id_by_key.get(
                truth_key
            )
        )

        if truth_candidate_id is None:
            calculated_rank = None
            mass_row_count = 0
            used_fallback = False
        else:
            (
                calculated_rank,
                mass_row_count,
                used_fallback,
            ) = rank_truth(
                query_vectors=(
                    vectors[
                        query_indices
                    ]
                ),
                query_modes=(
                    query_modes
                ),
                query_masses=(
                    query_masses
                ),
                reference_vectors=(
                    reference_vectors
                ),
                reference_modes=(
                    reference_modes
                ),
                reference_masses=(
                    reference_masses
                ),
                reference_candidate_ids=(
                    reference_candidate_ids
                ),
                truth_candidate_id=(
                    truth_candidate_id
                ),
                candidate_count=(
                    candidate_count
                ),
            )

        if used_fallback:
            fallback_count += 1

        frozen_rank = baseline2.iloc[
            structure_index
        ][
            "rank"
        ]

        frozen_rank_value = (
            None
            if pd.isna(
                frozen_rank
            )
            else int(
                frozen_rank
            )
        )

        if (
            calculated_rank
            == frozen_rank_value
        ):
            reproduced += 1
        else:
            mismatches.append(
                {
                    "inchikey14": (
                        truth_key
                    ),
                    "query_library": (
                        query_library
                    ),
                    "frozen_rank": (
                        frozen_rank_value
                    ),
                    "calculated_rank": (
                        calculated_rank
                    ),
                    "mass_reference_rows": (
                        mass_row_count
                    ),
                    "used_fallback": (
                        used_fallback
                    ),
                }
            )

        if (
            (structure_index + 1)
            % 100
            == 0
        ):
            print(
                f"  "
                f"{structure_index + 1:>4}"
                f"/{len(manifest)}"
            )

    print()
    print(
        "Baseline 2 reproduction"
    )
    print(
        "-----------------------"
    )

    print(
        f"Exact ranks: "
        f"{reproduced:,} / "
        f"{len(manifest):,}"
    )

    print(
        f"Fallbacks:   "
        f"{fallback_count:,}"
    )

    print(
        f"Mismatches:  "
        f"{len(mismatches):,}"
    )

    if mismatches:
        print()
        print(
            "First mismatches"
        )
        print(
            "----------------"
        )

        print(
            pd.DataFrame(
                mismatches
            )
            .head(
                50
            )
            .to_string(
                index=False
            )
        )

        raise RuntimeError(
            "Mass-filtered scoring does not "
            "exactly reproduce frozen Baseline 2."
        )

    print()
    print(
        "PASS: frozen Baseline 2 ranks "
        "are exactly reproduced."
    )


if __name__ == "__main__":
    main()