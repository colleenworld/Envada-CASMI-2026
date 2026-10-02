from pathlib import Path

import numpy as np
import pandas as pd

from casmi26.chemistry.adducts import neutral_mass
from casmi26.retrieval.candidates import (
    merge_candidate_ids,
    opposite_polarity_mass_candidates,
    same_polarity_mass_candidates,
    top_k_candidate_ids,
)
from casmi26.retrieval.index import (
    load_retrieval_index,
)
from casmi26.retrieval.indexed import (
    search_structures_indexed_blockwise,
)
from casmi26.retrieval.mass_index import (
    build_neutral_mass_index,
)


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

RESULTS_DIR = Path(
    "data/processed/results"
)

CANDIDATE_RESULTS_PATH = (
    RESULTS_DIR
    / "retrieval_dev_baseline3_candidates.parquet"
)

CANDIDATE_MAP_PATH = (
    RESULTS_DIR
    / "retrieval_dev_baseline3_candidate_map.parquet"
)

TOLERANCE_DA = 0.01

SPECTRAL_TOP_K = 25

BLOCK_SIZE = 25_000

# Keep batches small enough that the dense
# query-structure x candidate score matrix remains manageable.
MAX_QUERY_SPECTRA_PER_BATCH = 64


def infer_neutral_masses(
    precursor_mz: np.ndarray,
    adducts: np.ndarray,
) -> np.ndarray:
    """
    Infer neutral molecular masses from precursor m/z and adduct.

    Unknown or unusable adducts produce NaN.
    """
    masses = np.full(
        len(precursor_mz),
        np.nan,
        dtype=np.float64,
    )

    for index, (mz, adduct) in enumerate(
        zip(
            precursor_mz,
            adducts,
            strict=True,
        )
    ):
        try:
            mz_value = float(mz)
        except (TypeError, ValueError):
            continue

        if not np.isfinite(mz_value):
            continue

        value = neutral_mass(
            mz_value,
            str(adduct),
        )

        if (
            value is not None
            and np.isfinite(value)
        ):
            masses[index] = value

    return masses


def build_reference_mask(
    metadata: pd.DataFrame,
    manifest: pd.DataFrame,
) -> np.ndarray:
    """
    Reproduce the retrieval-dev reference pool.

    For every benchmark structure, remove spectra belonging to its
    assigned query library. Spectra for that structure from other
    libraries remain valid references.
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
    Group query structures into batches while keeping all spectra
    belonging to a structure together.

    A single structure is never split between batches.
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
    # Reproduce the exact Baseline 1 / Baseline 2 reference pool.
    # ---------------------------------------------------------

    reference_mask = build_reference_mask(
        metadata,
        manifest,
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

    removed = (
        len(metadata)
        - len(reference_metadata)
    )

    print(
        f"Reference spectra: "
        f"{len(reference_metadata):,}"
    )

    print(
        f"Query spectra removed: "
        f"{removed:,}"
    )

    if removed != 19_812:
        raise RuntimeError(
            "Reference pool does not match "
            "Baseline 2."
        )

    # ---------------------------------------------------------
    # Encode reference structures as integer candidate IDs.
    # ---------------------------------------------------------

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

    candidate_id_by_key = {
        key: candidate_id
        for candidate_id, key
        in enumerate(candidate_keys)
    }

    candidate_count = len(
        candidate_keys
    )

    print(
        f"Candidate structures: "
        f"{candidate_count:,}"
    )

    # ---------------------------------------------------------
    # Reference modes are used both by the mass candidate
    # generator and by same-polarity spectral retrieval.
    # ---------------------------------------------------------

    reference_modes = (
        reference_metadata[
            "ionization_mode"
        ]
        .astype(str)
        .to_numpy()
    )

    # ---------------------------------------------------------
    # Build neutral-mass index over the reference spectra.
    # ---------------------------------------------------------

    print(
        "Building neutral-mass index..."
    )

    reference_masses = (
        infer_neutral_masses(
            reference_metadata[
                "precursor_mz"
            ].to_numpy(),
            reference_metadata[
                "adduct"
            ]
            .astype(str)
            .to_numpy(),
        )
    )

    mass_index = build_neutral_mass_index(
        neutral_masses=reference_masses,
        modes=reference_modes,
    )

    # ---------------------------------------------------------
    # Locate every structure's query spectra once.
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

    truth_candidate_ids: list[
        int | None
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

        truth_candidate_ids.append(
            candidate_id_by_key.get(
                truth_key
            )
        )

    # ---------------------------------------------------------
    # Sources A and B:
    #
    # A = same-polarity neutral mass
    # B = opposite-polarity neutral mass
    # ---------------------------------------------------------

    print()
    print(
        "Generating mass candidates..."
    )

    rows: list[dict] = []

    for position, (
        row,
        query_indices,
        truth_candidate_id,
    ) in enumerate(
        zip(
            manifest.itertuples(
                index=False
            ),
            query_indices_by_structure,
            truth_candidate_ids,
            strict=True,
        ),
        start=1,
    ):
        truth_key = str(
            row.inchikey14
        )

        query_library = str(
            row.query_library
        )

        query_metadata = metadata.iloc[
            query_indices
        ]

        query_masses = (
            infer_neutral_masses(
                query_metadata[
                    "precursor_mz"
                ].to_numpy(),
                query_metadata[
                    "adduct"
                ]
                .astype(str)
                .to_numpy(),
            )
        )

        query_modes = (
            query_metadata[
                "ionization_mode"
            ]
            .astype(str)
            .to_numpy()
        )

        # -----------------------------------------------------
        # Source A:
        # same-polarity neutral-mass candidates.
        # -----------------------------------------------------

        same_candidates = (
            same_polarity_mass_candidates(
                index=mass_index,
                neutral_masses=query_masses,
                modes=query_modes,
                reference_candidate_ids=(
                    reference_candidate_ids
                ),
                tolerance_da=TOLERANCE_DA,
            )
        )

        # -----------------------------------------------------
        # Source B:
        # opposite-polarity neutral-mass candidates.
        # -----------------------------------------------------

        opposite_candidates = (
            opposite_polarity_mass_candidates(
                index=mass_index,
                neutral_masses=query_masses,
                modes=query_modes,
                reference_candidate_ids=(
                    reference_candidate_ids
                ),
                tolerance_da=TOLERANCE_DA,
            )
        )

        combined_mass_candidates = (
            merge_candidate_ids(
                same_candidates,
                opposite_candidates,
            )
        )

        truth_in_same = (
            truth_candidate_id is not None
            and np.any(
                same_candidates
                == truth_candidate_id
            )
        )

        truth_in_opposite = (
            truth_candidate_id is not None
            and np.any(
                opposite_candidates
                == truth_candidate_id
            )
        )

        truth_in_combined_mass = (
            truth_candidate_id is not None
            and np.any(
                combined_mass_candidates
                == truth_candidate_id
            )
        )

        rows.append(
            {
                "inchikey14": truth_key,
                "query_library": (
                    query_library
                ),
                "same_candidate_count": (
                    len(
                        same_candidates
                    )
                ),
                "opposite_candidate_count": (
                    len(
                        opposite_candidates
                    )
                ),
                "combined_mass_candidate_count": (
                    len(
                        combined_mass_candidates
                    )
                ),
                "truth_in_same": bool(
                    truth_in_same
                ),
                "truth_in_opposite": bool(
                    truth_in_opposite
                ),
                "truth_in_combined_mass": bool(
                    truth_in_combined_mass
                ),

                # Keep the actual IDs temporarily. These are
                # persisted after source C has been generated.
                "_same_candidates": (
                    same_candidates
                ),
                "_opposite_candidates": (
                    opposite_candidates
                ),
                "_combined_mass_candidates": (
                    combined_mass_candidates
                ),
            }
        )

        if (
            position % 100 == 0
            or position == len(manifest)
        ):
            print(
                f"Mass candidates: "
                f"{position:>4}/"
                f"{len(manifest)}"
            )

    # ---------------------------------------------------------
    # Source C:
    #
    # Same-polarity unfiltered spectral Top-25.
    #
    # Use batched indexed retrieval so we search the complete
    # reference pool efficiently.
    # ---------------------------------------------------------

    print()
    print(
        "Generating spectral Top-25 candidates..."
    )

    batches = build_query_batches(
        query_indices_by_structure=(
            query_indices_by_structure
        ),
        max_query_spectra=(
            MAX_QUERY_SPECTRA_PER_BATCH
        ),
    )

    print(
        f"Spectral batches: "
        f"{len(batches):,}"
    )

    spectral_candidates_by_structure: list[
        np.ndarray | None
    ] = [
        None
        for _ in range(
            len(manifest)
        )
    ]

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

        batch_query_indices_array = (
            np.asarray(
                batch_query_indices,
                dtype=np.int64,
            )
        )

        batch_query_vectors = vectors[
            batch_query_indices_array
        ]

        batch_scores = (
            search_structures_indexed_blockwise(
                query_vectors=(
                    batch_query_vectors
                ),
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
            spectral_candidates = (
                top_k_candidate_ids(
                    batch_scores[
                        local_structure_id
                    ],
                    k=SPECTRAL_TOP_K,
                )
            )

            spectral_candidates_by_structure[
                structure_index
            ] = spectral_candidates

        if (
            batch_number % 10 == 0
            or batch_number == len(batches)
        ):
            print(
                f"Spectral batches: "
                f"{batch_number:>4}/"
                f"{len(batches)}"
            )

    # ---------------------------------------------------------
    # Combine A + B + C and preserve the candidate sets.
    # ---------------------------------------------------------

    for structure_index, result in enumerate(
        rows
    ):
        spectral_candidates = (
            spectral_candidates_by_structure[
                structure_index
            ]
        )

        if spectral_candidates is None:
            raise RuntimeError(
                "Missing spectral candidates "
                f"for structure "
                f"{structure_index}"
            )

        same_candidates = (
            result[
                "_same_candidates"
            ]
        )

        opposite_candidates = (
            result[
                "_opposite_candidates"
            ]
        )

        combined_mass_candidates = (
            result[
                "_combined_mass_candidates"
            ]
        )

        all_candidates = (
            merge_candidate_ids(
                combined_mass_candidates,
                spectral_candidates,
            )
        )

        truth_candidate_id = (
            truth_candidate_ids[
                structure_index
            ]
        )

        truth_in_spectral = (
            truth_candidate_id is not None
            and np.any(
                spectral_candidates
                == truth_candidate_id
            )
        )

        truth_in_all = (
            truth_candidate_id is not None
            and np.any(
                all_candidates
                == truth_candidate_id
            )
        )

        result[
            "spectral_candidate_count"
        ] = len(
            spectral_candidates
        )

        result[
            "all_candidate_count"
        ] = len(
            all_candidates
        )

        result[
            "truth_in_spectral"
        ] = bool(
            truth_in_spectral
        )

        result[
            "truth_in_all"
        ] = bool(
            truth_in_all
        )

        # -----------------------------------------------------
        # Persist the candidate IDs for later ranking
        # experiments.
        #
        # Candidate IDs are positions in candidate_keys. The
        # candidate map saved below preserves that mapping.
        # -----------------------------------------------------

        result[
            "same_candidate_ids"
        ] = (
            same_candidates.tolist()
        )

        result[
            "opposite_candidate_ids"
        ] = (
            opposite_candidates.tolist()
        )

        result[
            "spectral_candidate_ids"
        ] = (
            spectral_candidates.tolist()
        )

        result[
            "hybrid_candidate_ids"
        ] = (
            all_candidates.tolist()
        )

        # Remove temporary NumPy-array fields before constructing
        # the final DataFrame.
        del result[
            "_same_candidates"
        ]

        del result[
            "_opposite_candidates"
        ]

        del result[
            "_combined_mass_candidates"
        ]

    results = pd.DataFrame(
        rows
    )

    # ---------------------------------------------------------
    # Source A must still exactly reproduce frozen Baseline 2.
    # ---------------------------------------------------------

    comparison = results.merge(
        baseline2[
            [
                "inchikey14",
                "candidate_count",
                "truth_in_mass_filter",
            ]
        ],
        on="inchikey14",
        validate="one_to_one",
    )

    candidate_count_match = (
        comparison[
            "same_candidate_count"
        ]
        == comparison[
            "candidate_count"
        ]
    )

    truth_match = (
        comparison[
            "truth_in_same"
        ]
        == comparison[
            "truth_in_mass_filter"
        ]
    )

    # ---------------------------------------------------------
    # Incremental recovery.
    # ---------------------------------------------------------

    missed_by_a = (
        ~results[
            "truth_in_same"
        ]
    )

    recovered_by_b = (
        missed_by_a
        & results[
            "truth_in_opposite"
        ]
    )

    missed_by_ab = (
        ~results[
            "truth_in_combined_mass"
        ]
    )

    recovered_by_c = (
        missed_by_ab
        & results[
            "truth_in_spectral"
        ]
    )

    still_missing = (
        ~results[
            "truth_in_all"
        ]
    )

    # ---------------------------------------------------------
    # Summary.
    # ---------------------------------------------------------

    print()
    print(
        "Baseline 3 candidate sources"
    )
    print(
        "----------------------------"
    )

    print(
        f"Structures: "
        f"{len(results):,}"
    )

    print()
    print(
        "A: same-polarity mass"
    )

    print(
        f"  Truth coverage:    "
        f"{results['truth_in_same'].mean():.2%}"
    )

    print(
        f"  Median candidates: "
        f"{results['same_candidate_count'].median():,.0f}"
    )

    print(
        f"  Mean candidates:   "
        f"{results['same_candidate_count'].mean():,.2f}"
    )

    print()
    print(
        "B: opposite-polarity mass"
    )

    print(
        f"  Truth coverage:    "
        f"{results['truth_in_opposite'].mean():.2%}"
    )

    print(
        f"  Median candidates: "
        f"{results['opposite_candidate_count'].median():,.0f}"
    )

    print(
        f"  Mean candidates:   "
        f"{results['opposite_candidate_count'].mean():,.2f}"
    )

    print()
    print(
        "A + B: combined mass candidates"
    )

    print(
        f"  Truth coverage:    "
        f"{results['truth_in_combined_mass'].mean():.2%}"
    )

    print(
        f"  Median candidates: "
        f"{results['combined_mass_candidate_count'].median():,.0f}"
    )

    print(
        f"  Mean candidates:   "
        f"{results['combined_mass_candidate_count'].mean():,.2f}"
    )

    print()
    print(
        "C: same-polarity spectral Top-25"
    )

    print(
        f"  Truth coverage:    "
        f"{results['truth_in_spectral'].mean():.2%}"
    )

    print(
        f"  Median candidates: "
        f"{results['spectral_candidate_count'].median():,.0f}"
    )

    print(
        f"  Mean candidates:   "
        f"{results['spectral_candidate_count'].mean():,.2f}"
    )

    print()
    print(
        "A + B + C: hybrid candidates"
    )

    print(
        f"  Truth coverage:    "
        f"{results['truth_in_all'].mean():.2%}"
    )

    print(
        f"  Median candidates: "
        f"{results['all_candidate_count'].median():,.0f}"
    )

    print(
        f"  Mean candidates:   "
        f"{results['all_candidate_count'].mean():,.2f}"
    )

    print()
    print(
        "Incremental recovery"
    )

    print(
        f"  Missed by A:       "
        f"{missed_by_a.sum():,}"
    )

    print(
        f"  Recovered by B:    "
        f"{recovered_by_b.sum():,}"
    )

    print(
        f"  Missed by A+B:     "
        f"{missed_by_ab.sum():,}"
    )

    print(
        f"  Recovered by C:    "
        f"{recovered_by_c.sum():,}"
    )

    print(
        f"  Still missing:     "
        f"{still_missing.sum():,}"
    )

    print()
    print(
        "Baseline 2 reproduction"
    )

    print(
        f"  Candidate counts:  "
        f"{candidate_count_match.sum():,} / "
        f"{len(comparison):,}"
    )

    print(
        f"  Truth status:      "
        f"{truth_match.sum():,} / "
        f"{len(comparison):,}"
    )

    # ---------------------------------------------------------
    # Protect against accidentally changing source A while
    # developing Baseline 3.
    # ---------------------------------------------------------

    if not candidate_count_match.all():
        print()
        print(
            "Candidate-count mismatches:"
        )

        print(
            comparison.loc[
                ~candidate_count_match,
                [
                    "inchikey14",
                    "same_candidate_count",
                    "candidate_count",
                ],
            ]
            .head(20)
            .to_string(
                index=False
            )
        )

    if not truth_match.all():
        print()
        print(
            "Truth-status mismatches:"
        )

        print(
            comparison.loc[
                ~truth_match,
                [
                    "inchikey14",
                    "truth_in_same",
                    "truth_in_mass_filter",
                ],
            ]
            .head(20)
            .to_string(
                index=False
            )
        )

    if not (
        candidate_count_match.all()
        and truth_match.all()
    ):
        raise RuntimeError(
            "Source A no longer exactly "
            "reproduces Baseline 2."
        )

    print()
    print(
        "PASS: source A still exactly "
        "reproduces Baseline 2."
    )

    # ---------------------------------------------------------
    # Show the remaining misses.
    #
    # This is diagnostic only. Truth status must not be used to
    # make inference-time candidate-generation decisions.
    # ---------------------------------------------------------

    if still_missing.any():
        print()
        print(
            "Still missing after A+B+C:"
        )

        print(
            results.loc[
                still_missing,
                [
                    "inchikey14",
                    "query_library",
                    "same_candidate_count",
                    "opposite_candidate_count",
                    "spectral_candidate_count",
                    "all_candidate_count",
                ],
            ]
            .to_string(
                index=False
            )
        )

    # ---------------------------------------------------------
    # Final artifact integrity checks before saving.
    # ---------------------------------------------------------

    if len(results) != len(
        manifest
    ):
        raise RuntimeError(
            "Candidate result count does not "
            "match manifest."
        )

    hybrid_lengths_match = (
        results[
            "hybrid_candidate_ids"
        ].map(len)
        == results[
            "all_candidate_count"
        ]
    )

    if not hybrid_lengths_match.all():
        raise RuntimeError(
            "Persisted hybrid candidate IDs "
            "do not match candidate counts."
        )

    # ---------------------------------------------------------
    # Save frozen candidate-generation artifacts.
    # ---------------------------------------------------------

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    results.to_parquet(
        CANDIDATE_RESULTS_PATH,
        index=False,
    )

    candidate_map = pd.DataFrame(
        {
            "candidate_id": (
                np.arange(
                    len(candidate_keys),
                    dtype=np.int64,
                )
            ),
            "inchikey14": (
                candidate_keys
            ),
        }
    )

    candidate_map.to_parquet(
        CANDIDATE_MAP_PATH,
        index=False,
    )

    print()
    print(
        "Saved candidate artifacts:"
    )

    print(
        f"  {CANDIDATE_RESULTS_PATH}"
    )

    print(
        f"  {CANDIDATE_MAP_PATH}"
    )

    print()
    print(
        "Artifact summary"
    )

    print(
        f"  Structures:        "
        f"{len(results):,}"
    )

    print(
        f"  Candidate map:     "
        f"{len(candidate_map):,}"
    )

    print(
        f"  Hybrid coverage:   "
        f"{results['truth_in_all'].mean():.2%}"
    )

    print(
        f"  Hybrid median:     "
        f"{results['all_candidate_count'].median():,.0f}"
    )

    print(
        "  Candidate lengths: "
        "PASS"
    )


if __name__ == "__main__":
    main()