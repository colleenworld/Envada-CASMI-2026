from pathlib import Path
import time

import numpy as np
import pandas as pd

from casmi26.retrieval.index import (
    load_retrieval_index,
)
from casmi26.retrieval.indexed import (
    search_structures_blockwise,
    search_structures_indexed_blockwise,
)


INDEX_DIR = Path(
    "data/processed/retrieval_index"
)

MANIFEST_PATH = Path(
    "data/processed/splits/retrieval_dev.parquet"
)

BLOCK_SIZE = 25_000


def build_reference_pool(
    metadata: pd.DataFrame,
    vectors,
    manifest: pd.DataFrame,
):
    """
    Remove the designated query spectra for every retrieval-dev
    structure.
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

    reference_mask = np.fromiter(
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

    reference_indices = np.flatnonzero(
        reference_mask
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

    return (
        reference_metadata,
        reference_vectors,
    )


def find_dictionary_rank(
    truth_key: str,
    scores: dict[str, float],
) -> int | None:
    """
    Rank a truth structure using the existing dictionary result.
    """
    ranked = sorted(
        scores.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    for rank, (key, _) in enumerate(
        ranked,
        start=1,
    ):
        if key == truth_key:
            return rank

    return None


def find_indexed_rank(
    truth_candidate_id: int,
    scores: np.ndarray,
) -> int | None:
    """
    Return the truth rank from one numeric candidate-score row.

    Candidates that were never compared have -inf scores and are
    excluded from ranking.

    Rank is 1 + the number of candidates with a strictly greater
    score than the truth.
    """
    truth_score = scores[
        truth_candidate_id
    ]

    if np.isneginf(
        truth_score
    ):
        return None

    return (
        1
        + int(
            np.count_nonzero(
                scores > truth_score
            )
        )
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

    print(
        f"Index spectra:       "
        f"{len(metadata):,}"
    )

    print(
        f"Dev structures:      "
        f"{len(manifest):,}"
    )

    # ---------------------------------------------------------
    # Same 16 structures used by our previous smoke comparison:
    # two structures from each query library.
    # ---------------------------------------------------------

    comparison_manifest = (
        manifest
        .groupby(
            "query_library",
            sort=True,
            group_keys=False,
        )
        .head(2)
        .reset_index(drop=True)
    )

    print()
    print(
        "Comparison query libraries"
    )
    print(
        "--------------------------"
    )

    print(
        comparison_manifest[
            "query_library"
        ]
        .value_counts()
        .sort_index()
    )

    # ---------------------------------------------------------
    # Build the complete dev reference pool.
    # ---------------------------------------------------------

    print()
    print(
        "Building dev reference pool..."
    )

    (
        reference_metadata,
        reference_vectors,
    ) = build_reference_pool(
        metadata=metadata,
        vectors=vectors,
        manifest=manifest,
    )

    removed = (
        len(metadata)
        - len(reference_metadata)
    )

    print(
        f"Reference spectra:   "
        f"{len(reference_metadata):,}"
    )

    print(
        f"Query spectra removed: "
        f"{removed:,}"
    )

    if removed != 19_812:
        raise RuntimeError(
            "Expected exactly 19,812 "
            "query spectra to be removed"
        )

    # ---------------------------------------------------------
    # Build the 16-structure query batch.
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

    query_indices_parts = []
    query_key_parts = []

    for row in (
        comparison_manifest.itertuples(
            index=False
        )
    ):
        truth_key = str(
            row.inchikey14
        )

        query_library = str(
            row.query_library
        )

        query_mask = (
            (metadata_keys == truth_key)
            & (
                metadata_libraries
                == query_library
            )
        )

        query_indices = np.flatnonzero(
            query_mask
        )

        if len(query_indices) == 0:
            raise RuntimeError(
                "No query spectra found for "
                f"{truth_key} / "
                f"{query_library}"
            )

        query_indices_parts.append(
            query_indices
        )

        query_key_parts.append(
            np.repeat(
                truth_key,
                len(query_indices),
            )
        )

    query_indices = np.concatenate(
        query_indices_parts
    )

    query_keys = np.concatenate(
        query_key_parts
    )

    query_vectors = vectors[
        query_indices
    ]

    query_modes = (
        metadata.iloc[
            query_indices
        ][
            "ionization_mode"
        ]
        .astype(str)
        .to_numpy()
    )

    print()
    print(
        f"Comparison structures: "
        f"{len(comparison_manifest)}"
    )

    print(
        f"Comparison spectra:    "
        f"{len(query_indices)}"
    )

    # ---------------------------------------------------------
    # Existing dictionary implementation.
    # ---------------------------------------------------------

    print()
    print(
        "Running dictionary batched search..."
    )

    dictionary_started = (
        time.monotonic()
    )

    dictionary_results = (
        search_structures_blockwise(
            query_vectors=query_vectors,
            query_structure_keys=(
                query_keys
            ),
            query_modes=query_modes,
            reference_vectors=(
                reference_vectors
            ),
            reference_metadata=(
                reference_metadata
            ),
            block_size=BLOCK_SIZE,
        )
    )

    dictionary_elapsed = (
        time.monotonic()
        - dictionary_started
    )

    # ---------------------------------------------------------
    # Encode reference candidate structures as integer IDs.
    # ---------------------------------------------------------

    print()
    print(
        "Encoding candidate structures..."
    )

    reference_keys = (
        reference_metadata[
            "inchikey14"
        ]
        .astype(str)
        .to_numpy()
    )

    candidate_keys, reference_candidate_ids = (
        np.unique(
            reference_keys,
            return_inverse=True,
        )
    )

    reference_candidate_ids = (
        reference_candidate_ids.astype(
            np.int64,
            copy=False,
        )
    )

    candidate_id_by_key = {
        key: candidate_id
        for candidate_id, key
        in enumerate(
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

    print(
        f"Candidate structures: "
        f"{len(candidate_keys):,}"
    )

    # ---------------------------------------------------------
    # Encode the 16 query structures as local integer IDs.
    # ---------------------------------------------------------

    query_structure_keys = (
        comparison_manifest[
            "inchikey14"
        ]
        .astype(str)
        .to_numpy()
    )

    query_id_by_key = {
        key: query_id
        for query_id, key
        in enumerate(
            query_structure_keys
        )
    }

    query_structure_ids = np.array(
        [
            query_id_by_key[
                key
            ]
            for key in query_keys
        ],
        dtype=np.int64,
    )

    # ---------------------------------------------------------
    # Numeric-ID implementation.
    # ---------------------------------------------------------

    print()
    print(
        "Running numeric-ID batched search..."
    )

    indexed_started = (
        time.monotonic()
    )

    indexed_scores = (
        search_structures_indexed_blockwise(
            query_vectors=query_vectors,
            query_structure_ids=(
                query_structure_ids
            ),
            query_modes=query_modes,
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
                len(query_structure_keys)
            ),
            candidate_count=(
                len(candidate_keys)
            ),
            block_size=BLOCK_SIZE,
        )
    )

    indexed_elapsed = (
        time.monotonic()
        - indexed_started
    )

    # ---------------------------------------------------------
    # Compare every candidate score and truth rank.
    # ---------------------------------------------------------

    print()
    print(
        "Comparing results..."
    )
    print()

    for position, truth_key in enumerate(
        query_structure_keys,
        start=1,
    ):
        query_id = (
            query_id_by_key[
                truth_key
            ]
        )

        dictionary_scores = (
            dictionary_results[
                truth_key
            ]
        )

        numeric_scores = (
            indexed_scores[
                query_id
            ]
        )

        # Compare every candidate produced by the dictionary
        # implementation.
        for (
            candidate_key,
            expected_score,
        ) in dictionary_scores.items():
            candidate_id = (
                candidate_id_by_key[
                    candidate_key
                ]
            )

            actual_score = (
                numeric_scores[
                    candidate_id
                ]
            )

            if not np.isclose(
                expected_score,
                actual_score,
                rtol=1e-6,
                atol=1e-7,
            ):
                raise AssertionError(
                    "Score mismatch: "
                    f"{truth_key} -> "
                    f"{candidate_key}: "
                    f"dictionary="
                    f"{expected_score}, "
                    f"numeric="
                    f"{actual_score}"
                )

        dictionary_rank = (
            find_dictionary_rank(
                truth_key,
                dictionary_scores,
            )
        )

        truth_candidate_id = (
            candidate_id_by_key.get(
                truth_key
            )
        )

        if truth_candidate_id is None:
            numeric_rank = None
        else:
            numeric_rank = (
                find_indexed_rank(
                    truth_candidate_id,
                    numeric_scores,
                )
            )

        if (
            dictionary_rank
            != numeric_rank
        ):
            raise AssertionError(
                f"Rank mismatch for "
                f"{truth_key}: "
                f"dictionary="
                f"{dictionary_rank}, "
                f"numeric="
                f"{numeric_rank}"
            )

        print(
            f"{position:>2}/"
            f"{len(query_structure_keys)}  "
            f"{truth_key}  "
            f"rank="
            f"{str(numeric_rank):>6}  "
            f"PASS"
        )

    # ---------------------------------------------------------
    # Timing summary.
    # ---------------------------------------------------------

    print()
    print(
        "Comparison complete"
    )
    print(
        "-------------------"
    )

    print(
        f"Structures:       "
        f"{len(query_structure_keys)}"
    )

    print(
        f"Query spectra:    "
        f"{len(query_indices)}"
    )

    print(
        f"Candidates:       "
        f"{len(candidate_keys):,}"
    )

    print(
        f"Dictionary time:  "
        f"{dictionary_elapsed:.2f}s"
    )

    print(
        f"Numeric-ID time:  "
        f"{indexed_elapsed:.2f}s"
    )

    if indexed_elapsed > 0:
        print(
            f"Speedup:          "
            f"{dictionary_elapsed / indexed_elapsed:.2f}x"
        )

    print()
    print(
        "Candidate scores: PASS"
    )

    print(
        "Truth ranks:      PASS"
    )


if __name__ == "__main__":
    main()