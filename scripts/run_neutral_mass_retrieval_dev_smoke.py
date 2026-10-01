from pathlib import Path
import time

import numpy as np
import pandas as pd

from casmi26.chemistry.adducts import neutral_mass
from casmi26.retrieval.index import load_retrieval_index
from casmi26.retrieval.indexed import (
    search_structures_indexed_blockwise,
)
from casmi26.retrieval.mass_index import (
    build_neutral_mass_index,
    search_neutral_mass,
    search_neutral_masses,
)


INDEX_DIR = Path(
    "data/processed/retrieval_index"
)

MANIFEST_PATH = Path(
    "data/processed/splits/retrieval_dev.parquet"
)

BLOCK_SIZE = 25_000
NEUTRAL_MASS_TOLERANCE_DA = 0.01


def build_reference_pool(
    metadata: pd.DataFrame,
    vectors,
    manifest: pd.DataFrame,
):
    """
    Build exactly the same dev reference pool used by Baseline 1.

    For every benchmark structure, remove spectra belonging to its
    assigned query library. Spectra for that structure from other
    libraries remain valid references.
    """
    query_pairs = set(
        zip(
            manifest["inchikey14"].astype(str),
            manifest["query_library"].astype(str),
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


def infer_neutral_masses(
    precursor_mz: np.ndarray,
    adducts: np.ndarray,
) -> np.ndarray:
    """
    Infer neutral masses for spectra whose adducts we understand.

    Unsupported or invalid adducts receive NaN.
    """
    masses = np.full(
        len(precursor_mz),
        np.nan,
        dtype=np.float64,
    )

    for index, (
        mz,
        adduct,
    ) in enumerate(
        zip(
            precursor_mz,
            adducts,
            strict=True,
        )
    ):
        try:
            mz_value = float(mz)
        except (
            TypeError,
            ValueError,
        ):
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


def find_rank(
    truth_candidate_id: int,
    scores: np.ndarray,
) -> int | None:
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


def candidate_mask_for_query(
    *,
    query_neutral_masses: np.ndarray,
    query_modes: np.ndarray,
    candidate_neutral_masses: np.ndarray,
    candidate_modes: np.ndarray,
    tolerance_da: float,
) -> np.ndarray:
    """
    Determine which candidate spectra are chemically compatible
    with at least one query spectrum.

    This experiment remains same-polarity.

    A candidate spectrum is retained when:
      1. it has the same ionization mode as a query spectrum,
      2. both neutral masses are known, and
      3. their neutral masses differ by <= tolerance_da.
    """
    keep = np.zeros(
        len(candidate_neutral_masses),
        dtype=bool,
    )

    for mode in np.unique(
        query_modes
    ):
        query_mode_mask = (
            query_modes == mode
        )

        candidate_mode_mask = (
            candidate_modes == mode
        )

        mode_query_masses = (
            query_neutral_masses[
                query_mode_mask
            ]
        )

        mode_query_masses = (
            mode_query_masses[
                np.isfinite(
                    mode_query_masses
                )
            ]
        )

        if len(mode_query_masses) == 0:
            continue

        candidate_indices = (
            np.flatnonzero(
                candidate_mode_mask
                & np.isfinite(
                    candidate_neutral_masses
                )
            )
        )

        if len(candidate_indices) == 0:
            continue

        masses = (
            candidate_neutral_masses[
                candidate_indices
            ]
        )

        # There may be multiple query spectra/adduct observations
        # for one structure. A reference spectrum only needs to
        # agree with one of them.
        compatible = np.zeros(
            len(candidate_indices),
            dtype=bool,
        )

        for query_mass in (
            mode_query_masses
        ):
            compatible |= (
                np.abs(
                    masses
                    - query_mass
                )
                <= tolerance_da
            )

        keep[
            candidate_indices[
                compatible
            ]
        ] = True

    return keep


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
    # Use exactly the same 16-structure stratified smoke sample
    # as the earlier retrieval comparison.
    # ---------------------------------------------------------

    comparison_manifest = (
        manifest
        .sort_values(
            [
                "query_library",
                "inchikey14",
            ]
        )
        .groupby(
            "query_library",
            sort=True,
        )
        .head(2)
        .reset_index(drop=True)
    )

    print()
    print(
        "Smoke query libraries"
    )
    print(
        "---------------------"
    )

    print(
        comparison_manifest[
            "query_library"
        ].value_counts(
            sort=False
        ).sort_index()
    )

    # ---------------------------------------------------------
    # Build the complete dev reference pool, not merely a
    # reference pool for these 16 smoke structures.
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
            "query spectra to be removed, "
            f"but removed {removed:,}"
        )

    # ---------------------------------------------------------
    # Encode reference candidate structures.
    # ---------------------------------------------------------

    print()
    print(
        "Encoding reference candidates..."
    )

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

    reference_precursor_mz = (
        reference_metadata[
            "precursor_mz"
        ].to_numpy()
    )

    reference_adducts = (
        reference_metadata[
            "adduct"
        ]
        .astype(str)
        .to_numpy()
    )

    print(
        f"Candidate structures: "
        f"{len(candidate_keys):,}"
    )

    # ---------------------------------------------------------
    # Infer neutral masses once for the entire reference pool.
    # ---------------------------------------------------------

    print()
    print(
        "Inferring reference neutral masses..."
    )

    reference_neutral_masses = (
        infer_neutral_masses(
            reference_precursor_mz,
            reference_adducts,
        )
    )

    mass_index = (
        build_neutral_mass_index(
            neutral_masses=(
                reference_neutral_masses
            ),
            modes=reference_modes,
        )
    )

    known_reference_masses = int(
        np.count_nonzero(
            np.isfinite(
                reference_neutral_masses
            )
        )
    )

    print(
        f"Reference spectra with "
        f"neutral mass: "
        f"{known_reference_masses:,} / "
        f"{len(reference_metadata):,} "
        f"("
        f"{known_reference_masses / len(reference_metadata):.2%}"
        f")"
    )

    # ---------------------------------------------------------
    # Locate query spectra for the 16 smoke structures.
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

    print()
    print(
        "Running neutral-mass smoke test..."
    )
    print()

    results: list[dict] = []

    total_started = time.monotonic()

    for number, row in enumerate(
        comparison_manifest.itertuples(
            index=False
        ),
        start=1,
    ):
        truth_key = str(
            row.inchikey14
        )

        query_library = str(
            row.query_library
        )

        query_mask = (
            (
                metadata_keys
                == truth_key
            )
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

        query_metadata = (
            metadata.iloc[
                query_indices
            ]
        )

        query_vectors = vectors[
            query_indices
        ]

        query_modes = (
            query_metadata[
                "ionization_mode"
            ]
            .astype(str)
            .to_numpy()
        )

        query_precursor_mz = (
            query_metadata[
                "precursor_mz"
            ].to_numpy()
        )

        query_adducts = (
            query_metadata[
                "adduct"
            ]
            .astype(str)
            .to_numpy()
        )

        query_neutral_masses = (
            infer_neutral_masses(
                query_precursor_mz,
                query_adducts,
            )
        )

        known_query_masses = int(
            np.count_nonzero(
                np.isfinite(
                    query_neutral_masses
                )
            )
        )

        # -----------------------------------------------------
        # Baseline 1 score: unrestricted same-polarity cosine.
        # -----------------------------------------------------

        query_structure_ids = np.zeros(
            len(query_indices),
            dtype=np.int64,
        )

        baseline_started = (
            time.monotonic()
        )

        baseline_scores = (
            search_structures_indexed_blockwise(
                query_vectors=(
                    query_vectors
                ),
                query_structure_ids=(
                    query_structure_ids
                ),
                query_modes=(
                    query_modes
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
                query_structure_count=1,
                candidate_count=(
                    len(candidate_keys)
                ),
                block_size=BLOCK_SIZE,
            )[0]
        )

        baseline_elapsed = (
            time.monotonic()
            - baseline_started
        )

        truth_candidate_id = (
            candidate_id_by_key.get(
                truth_key
            )
        )

        if truth_candidate_id is None:
            baseline_rank = None
        else:
            baseline_rank = find_rank(
                truth_candidate_id,
                baseline_scores,
            )

        baseline_candidate_count = int(
            np.count_nonzero(
                np.isfinite(
                    baseline_scores
                )
            )
        )

        # -----------------------------------------------------
        # Neutral-mass filter.
        #
        # First filter reference spectra. Then run the exact same
        # same-polarity cosine retrieval over that reduced pool.
        # -----------------------------------------------------

        filtered_reference_indices = (
            search_neutral_masses(
                index=mass_index,
                neutral_masses=(
                    query_neutral_masses
                ),
                modes=query_modes,
                tolerance_da=(
                    NEUTRAL_MASS_TOLERANCE_DA
                ),
            )
        )

        if len(
            filtered_reference_indices
        ) == 0:
            filtered_scores = np.full(
                len(candidate_keys),
                -np.inf,
                dtype=np.float32,
            )

            filtered_elapsed = 0.0

        else:
            filtered_started = (
                time.monotonic()
            )

            filtered_scores = (
                search_structures_indexed_blockwise(
                    query_vectors=(
                        query_vectors
                    ),
                    query_structure_ids=(
                        query_structure_ids
                    ),
                    query_modes=(
                        query_modes
                    ),
                    reference_vectors=(
                        reference_vectors[
                            filtered_reference_indices
                        ]
                    ),
                    reference_candidate_ids=(
                        reference_candidate_ids[
                            filtered_reference_indices
                        ]
                    ),
                    reference_modes=(
                        reference_modes[
                            filtered_reference_indices
                        ]
                    ),
                    query_structure_count=1,
                    candidate_count=(
                        len(candidate_keys)
                    ),
                    block_size=BLOCK_SIZE,
                )[0]
            )

            filtered_elapsed = (
                time.monotonic()
                - filtered_started
            )

        if truth_candidate_id is None:
            filtered_rank = None
        else:
            filtered_rank = find_rank(
                truth_candidate_id,
                filtered_scores,
            )

        filtered_candidate_count = int(
            np.count_nonzero(
                np.isfinite(
                    filtered_scores
                )
            )
        )

        truth_survived = (
            truth_candidate_id
            is not None
            and np.isfinite(
                filtered_scores[
                    truth_candidate_id
                ]
            )
        )

        reduction = (
            1.0
            - (
                filtered_candidate_count
                / baseline_candidate_count
            )
            if baseline_candidate_count
            else 0.0
        )

        results.append(
            {
                "inchikey14": truth_key,
                "query_library": (
                    query_library
                ),
                "query_spectra": (
                    len(query_indices)
                ),
                "known_query_masses": (
                    known_query_masses
                ),
                "baseline_rank": (
                    baseline_rank
                ),
                "filtered_rank": (
                    filtered_rank
                ),
                "baseline_candidates": (
                    baseline_candidate_count
                ),
                "filtered_candidates": (
                    filtered_candidate_count
                ),
                "candidate_reduction": (
                    reduction
                ),
                "truth_survived": (
                    truth_survived
                ),
            }
        )

        baseline_rank_text = (
            str(baseline_rank)
            if baseline_rank is not None
            else "None"
        )

        filtered_rank_text = (
            str(filtered_rank)
            if filtered_rank is not None
            else "None"
        )

        status = (
            "PASS"
            if truth_survived
            else "TRUTH FILTERED"
        )

        print(
            f"{number:>2}/16  "
            f"{truth_key}  "
            f"{query_library:<12}  "
            f"rank "
            f"{baseline_rank_text:>6}"
            f" -> "
            f"{filtered_rank_text:>6}  "
            f"candidates "
            f"{baseline_candidate_count:>6,}"
            f" -> "
            f"{filtered_candidate_count:>5,}  "
            f"({reduction:>6.2%} removed)  "
            f"{status}"
        )

        print(
            f"       "
            f"query spectra="
            f"{len(query_indices):,}, "
            f"neutral masses="
            f"{known_query_masses:,}, "
            f"baseline="
            f"{baseline_elapsed:.2f}s, "
            f"filtered="
            f"{filtered_elapsed:.2f}s"
        )

    total_elapsed = (
        time.monotonic()
        - total_started
    )

    # ---------------------------------------------------------
    # Summary.
    # ---------------------------------------------------------

    results_frame = pd.DataFrame(
        results
    )

    truth_survival_count = int(
        results_frame[
            "truth_survived"
        ].sum()
    )

    baseline_top1 = int(
        (
            results_frame[
                "baseline_rank"
            ]
            == 1
        ).sum()
    )

    filtered_top1 = int(
        (
            results_frame[
                "filtered_rank"
            ]
            == 1
        ).sum()
    )

    baseline_top25 = int(
        (
            results_frame[
                "baseline_rank"
            ].notna()
            & (
                results_frame[
                    "baseline_rank"
                ]
                <= 25
            )
        ).sum()
    )

    filtered_top25 = int(
        (
            results_frame[
                "filtered_rank"
            ].notna()
            & (
                results_frame[
                    "filtered_rank"
                ]
                <= 25
            )
        ).sum()
    )

    def rr25(
        rank,
    ) -> float:
        if (
            pd.isna(rank)
            or rank > 25
        ):
            return 0.0

        return 1.0 / float(
            rank
        )

    baseline_mrr = (
        results_frame[
            "baseline_rank"
        ]
        .map(rr25)
        .mean()
    )

    filtered_mrr = (
        results_frame[
            "filtered_rank"
        ]
        .map(rr25)
        .mean()
    )

    print()
    print(
        "Neutral-mass smoke summary"
    )
    print(
        "--------------------------"
    )

    print(
        f"Structures:              "
        f"{len(results_frame)}"
    )

    print(
        f"Tolerance:               "
        f"±{NEUTRAL_MASS_TOLERANCE_DA:.3f} Da"
    )

    print(
        f"Truth survived filter:   "
        f"{truth_survival_count}/"
        f"{len(results_frame)}"
    )

    print()
    print(
        f"Baseline MRR@25:         "
        f"{baseline_mrr:.4f}"
    )

    print(
        f"Filtered MRR@25:         "
        f"{filtered_mrr:.4f}"
    )

    print()
    print(
        f"Baseline Top-1:          "
        f"{baseline_top1}/"
        f"{len(results_frame)}"
    )

    print(
        f"Filtered Top-1:          "
        f"{filtered_top1}/"
        f"{len(results_frame)}"
    )

    print()
    print(
        f"Baseline Top-25:         "
        f"{baseline_top25}/"
        f"{len(results_frame)}"
    )

    print(
        f"Filtered Top-25:         "
        f"{filtered_top25}/"
        f"{len(results_frame)}"
    )

    print()
    print(
        f"Median baseline candidates: "
        f"{results_frame['baseline_candidates'].median():,.0f}"
    )

    print(
        f"Median filtered candidates: "
        f"{results_frame['filtered_candidates'].median():,.0f}"
    )

    print(
        f"Median candidate reduction: "
        f"{results_frame['candidate_reduction'].median():.2%}"
    )

    print()
    print(
        f"Total elapsed:           "
        f"{total_elapsed:.2f}s"
    )


if __name__ == "__main__":
    main()