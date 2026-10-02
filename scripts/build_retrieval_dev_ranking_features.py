from pathlib import Path

import numpy as np
import pandas as pd

from casmi26.retrieval.index import (
    load_retrieval_index,
)
from casmi26.retrieval.indexed import (
    search_structures_indexed_blockwise,
)
from casmi26.retrieval.ranking import (
    build_candidate_features,
)


INDEX_DIR = Path(
    "data/processed/retrieval_index"
)

MANIFEST_PATH = Path(
    "data/processed/splits/retrieval_dev.parquet"
)

CANDIDATES_PATH = Path(
    "data/processed/results/"
    "retrieval_dev_baseline3_candidates.parquet"
)

CANDIDATE_MAP_PATH = Path(
    "data/processed/results/"
    "retrieval_dev_baseline3_candidate_map.parquet"
)

OUTPUT_PATH = Path(
    "data/processed/results/"
    "retrieval_dev_baseline3_ranking_features.parquet"
)

BLOCK_SIZE = 25_000

MAX_QUERY_SPECTRA_PER_BATCH = 64

SPECTRAL_TOP_K = 25


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
                metadata["inchikey14"],
                metadata["ingest_lib"],
                strict=True,
            )
        ),
        dtype=bool,
        count=len(metadata),
    )


def build_query_batches(
    query_indices_by_structure: list[np.ndarray],
    max_query_spectra: int,
) -> list[list[int]]:
    """
    Batch structures without splitting a structure's query spectra.
    """
    batches: list[list[int]] = []

    current_batch: list[int] = []
    current_spectrum_count = 0

    for structure_index, query_indices in enumerate(
        query_indices_by_structure
    ):
        spectrum_count = len(
            query_indices
        )

        if (
            current_batch
            and current_spectrum_count
            + spectrum_count
            > max_query_spectra
        ):
            batches.append(
                current_batch
            )

            current_batch = []
            current_spectrum_count = 0

        current_batch.append(
            structure_index
        )

        current_spectrum_count += (
            spectrum_count
        )

    if current_batch:
        batches.append(
            current_batch
        )

    return batches


def main() -> None:
    print(
        "Loading frozen artifacts..."
    )

    manifest = pd.read_parquet(
        MANIFEST_PATH
    )

    candidates = pd.read_parquet(
        CANDIDATES_PATH
    )

    candidate_map = pd.read_parquet(
        CANDIDATE_MAP_PATH
    )

    metadata, vectors = (
        load_retrieval_index(
            INDEX_DIR
        )
    )

    # ---------------------------------------------------------
    # Validate frozen artifacts.
    # ---------------------------------------------------------

    if len(manifest) != len(
        candidates
    ):
        raise RuntimeError(
            "Manifest and candidate artifact "
            "have different structure counts."
        )

    if not np.array_equal(
        manifest[
            "inchikey14"
        ].astype(str).to_numpy(),
        candidates[
            "inchikey14"
        ].astype(str).to_numpy(),
    ):
        raise RuntimeError(
            "Candidate artifact order does not "
            "match retrieval-dev manifest."
        )

    expected_candidate_ids = np.arange(
        len(candidate_map),
        dtype=np.int64,
    )

    actual_candidate_ids = (
        candidate_map[
            "candidate_id"
        ].to_numpy(
            dtype=np.int64
        )
    )

    if not np.array_equal(
        expected_candidate_ids,
        actual_candidate_ids,
    ):
        raise RuntimeError(
            "Candidate map IDs are not "
            "contiguous and ordered."
        )

    candidate_keys = (
        candidate_map[
            "inchikey14"
        ]
        .astype(str)
        .to_numpy()
    )

    candidate_count = len(
        candidate_keys
    )

    print(
        f"Structures: "
        f"{len(manifest):,}"
    )

    print(
        f"Candidate structures: "
        f"{candidate_count:,}"
    )

    # ---------------------------------------------------------
    # Reproduce the exact frozen reference pool.
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
        .reset_index(drop=True)
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

    # Recreate the candidate IDs independently and make sure they
    # are exactly the IDs persisted by candidate generation.
    (
        rebuilt_candidate_keys,
        reference_candidate_ids,
    ) = np.unique(
        reference_keys,
        return_inverse=True,
    )

    if not np.array_equal(
        rebuilt_candidate_keys,
        candidate_keys,
    ):
        raise RuntimeError(
            "Candidate map does not match "
            "the reference pool."
        )

    reference_candidate_ids = (
        reference_candidate_ids.astype(
            np.int64,
            copy=False,
        )
    )

    reference_modes = (
        reference_metadata[
            "ionization_mode"
        ]
        .astype(str)
        .to_numpy()
    )

    print(
        f"Reference spectra: "
        f"{len(reference_metadata):,}"
    )

    print(
        f"Query spectra removed: "
        f"{removed:,}"
    )

    # ---------------------------------------------------------
    # Locate the frozen query spectra.
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

    query_indices_by_structure: list[
        np.ndarray
    ] = []

    for row in manifest.itertuples(
        index=False
    ):
        truth_key = str(
            row.inchikey14
        )

        query_library = str(
            row.query_library
        )

        query_indices = np.flatnonzero(
            (metadata_keys == truth_key)
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

        query_indices_by_structure.append(
            query_indices
        )

    batches = build_query_batches(
        query_indices_by_structure,
        MAX_QUERY_SPECTRA_PER_BATCH,
    )

    print(
        f"Ranking batches: "
        f"{len(batches):,}"
    )

    # ---------------------------------------------------------
    # Run the frozen same-polarity cosine search.
    #
    # We calculate the full score vector temporarily for each
    # batch, but persist only the hybrid candidate scores.
    # ---------------------------------------------------------

    feature_frames: list[
        pd.DataFrame
    ] = []

    spectral_order_matches = 0

    spectral_order_checked = 0

    for batch_number, structure_indices in enumerate(
        batches,
        start=1,
    ):
        batch_query_indices: list[int] = []

        batch_query_structure_ids: list[int] = []

        batch_query_modes: list[str] = []

        for (
            local_structure_id,
            structure_index,
        ) in enumerate(
            structure_indices
        ):
            query_indices = (
                query_indices_by_structure[
                    structure_index
                ]
            )

            batch_query_indices.extend(
                int(index)
                for index in query_indices
            )

            batch_query_structure_ids.extend(
                [local_structure_id]
                * len(query_indices)
            )

            batch_query_modes.extend(
                metadata.iloc[
                    query_indices
                ][
                    "ionization_mode"
                ]
                .astype(str)
                .tolist()
            )

        batch_scores = (
            search_structures_indexed_blockwise(
                query_vectors=vectors[
                    np.asarray(
                        batch_query_indices,
                        dtype=np.int64,
                    )
                ],
                query_structure_ids=(
                    np.asarray(
                        batch_query_structure_ids,
                        dtype=np.int64,
                    )
                ),
                query_modes=np.asarray(
                    batch_query_modes,
                    dtype=object,
                ),
                reference_vectors=(
                    reference_vectors
                ),
                reference_candidate_ids=(
                    reference_candidate_ids
                ),
                reference_modes=(
                    reference_modes
                ),
                query_structure_count=(
                    len(
                        structure_indices
                    )
                ),
                candidate_count=(
                    candidate_count
                ),
                block_size=BLOCK_SIZE,
            )
        )

        for (
            local_structure_id,
            structure_index,
        ) in enumerate(
            structure_indices
        ):
            candidate_row = (
                candidates.iloc[
                    structure_index
                ]
            )

            hybrid_ids = np.asarray(
                candidate_row[
                    "hybrid_candidate_ids"
                ],
                dtype=np.int64,
            )

            same_ids = np.asarray(
                candidate_row[
                    "same_candidate_ids"
                ],
                dtype=np.int64,
            )

            opposite_ids = np.asarray(
                candidate_row[
                    "opposite_candidate_ids"
                ],
                dtype=np.int64,
            )

            spectral_ids = np.asarray(
                candidate_row[
                    "spectral_candidate_ids"
                ],
                dtype=np.int64,
            )

            scores = batch_scores[
                local_structure_id
            ]

            features = (
                build_candidate_features(
                    hybrid_candidate_ids=(
                        hybrid_ids
                    ),
                    same_candidate_ids=(
                        same_ids
                    ),
                    opposite_candidate_ids=(
                        opposite_ids
                    ),
                    spectral_candidate_ids=(
                        spectral_ids
                    ),
                    scores=scores,
                )
            )

            # -------------------------------------------------
            # Consistency check:
            #
            # Reconstruct the Top-25 from the full score vector
            # using exactly the deterministic ordering used by
            # top_k_candidate_ids:
            #
            #   score descending
            #   candidate ID ascending for ties
            # -------------------------------------------------

            finite_ids = np.flatnonzero(
                np.isfinite(
                    scores
                )
            ).astype(
                np.int64,
                copy=False,
            )

            finite_scores = scores[
                finite_ids
            ]

            order = np.lexsort(
                (
                    finite_ids,
                    -finite_scores,
                )
            )

            reconstructed_top25 = (
                finite_ids[
                    order[
                        :SPECTRAL_TOP_K
                    ]
                ]
            )

            spectral_order_checked += 1

            if np.array_equal(
                reconstructed_top25,
                spectral_ids,
            ):
                spectral_order_matches += 1

            structure_key = str(
                candidate_row[
                    "inchikey14"
                ]
            )

            query_library = str(
                candidate_row[
                    "query_library"
                ]
            )

            candidate_ids = features[
                "candidate_id"
            ]

            frame = pd.DataFrame(
                {
                    "inchikey14": (
                        structure_key
                    ),
                    "query_library": (
                        query_library
                    ),
                    "candidate_id": (
                        candidate_ids
                    ),
                    "candidate_inchikey14": (
                        candidate_keys[
                            candidate_ids
                        ]
                    ),
                    "is_truth": (
                        candidate_keys[
                            candidate_ids
                        ]
                        == structure_key
                    ),
                    "in_same_mass": (
                        features[
                            "in_same_mass"
                        ]
                    ),
                    "in_opposite_mass": (
                        features[
                            "in_opposite_mass"
                        ]
                    ),
                    "in_spectral_top25": (
                        features[
                            "in_spectral_top25"
                        ]
                    ),
                    "spectral_rank": (
                        features[
                            "spectral_rank"
                        ]
                    ),
                    "same_polarity_cosine": (
                        features[
                            "same_polarity_cosine"
                        ]
                    ),
                }
            )

            feature_frames.append(
                frame
            )

        if (
            batch_number % 10 == 0
            or batch_number
            == len(batches)
        ):
            print(
                f"Ranking batches: "
                f"{batch_number:>4}/"
                f"{len(batches)}"
            )

    # ---------------------------------------------------------
    # Assemble the candidate-level feature table.
    # ---------------------------------------------------------

    features = pd.concat(
        feature_frames,
        ignore_index=True,
    )

    print()
    print(
        "Ranking feature artifact"
    )
    print(
        "------------------------"
    )

    print(
        f"Rows:               "
        f"{len(features):,}"
    )

    print(
        f"Structures:         "
        f"{features['inchikey14'].nunique():,}"
    )

    print(
        f"Truth rows:         "
        f"{features['is_truth'].sum():,}"
    )

    print(
        f"Spectral checks:    "
        f"{spectral_order_matches:,} / "
        f"{spectral_order_checked:,}"
    )

    # ---------------------------------------------------------
    # Integrity checks.
    # ---------------------------------------------------------

    expected_rows = int(
        candidates[
            "all_candidate_count"
        ].sum()
    )

    if len(features) != expected_rows:
        raise RuntimeError(
            "Feature row count does not match "
            "the frozen hybrid candidate counts."
        )

    if (
        features[
            "inchikey14"
        ].nunique()
        != len(manifest)
    ):
        raise RuntimeError(
            "Not all retrieval-dev structures "
            "are represented."
        )

    expected_truth_rows = int(
        candidates[
            "truth_in_all"
        ].sum()
    )

    actual_truth_rows = int(
        features[
            "is_truth"
        ].sum()
    )

    if (
        actual_truth_rows
        != expected_truth_rows
    ):
        raise RuntimeError(
            "Truth-row count does not match "
            "frozen hybrid candidate coverage."
        )

    if (
        spectral_order_matches
        != spectral_order_checked
    ):
        raise RuntimeError(
            "Spectral Top-25 ordering does not "
            "reproduce the frozen candidate artifact."
        )

    print()
    print(
        "PASS: candidate counts match "
        "the frozen artifact."
    )

    print(
        "PASS: truth coverage matches "
        "the frozen artifact."
    )

    print(
        "PASS: spectral Top-25 ordering "
        "is exactly reproduced."
    )

    # ---------------------------------------------------------
    # Save only after all checks have passed.
    # ---------------------------------------------------------

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    features.to_parquet(
        OUTPUT_PATH,
        index=False,
    )

    print()
    print(
        "Saved:"
    )

    print(
        f"  {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()