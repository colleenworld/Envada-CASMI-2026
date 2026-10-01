from pathlib import Path

import numpy as np
import pandas as pd

from casmi26.chemistry.adducts import neutral_mass


INDEX_METADATA_PATH = Path(
    "data/processed/retrieval_index/metadata.parquet"
)

MANIFEST_PATH = Path(
    "data/processed/splits/retrieval_dev.parquet"
)

MASS_ANALYSIS_PATH = Path(
    "data/processed/results/retrieval_dev_mass_filter_analysis.parquet"
)

BASELINE1_PATH = Path(
    "data/processed/results/retrieval_dev_baseline1.parquet"
)

RESULTS_PATH = Path(
    "data/processed/results/"
    "retrieval_dev_cross_polarity_mass_coverage.parquet"
)

TOLERANCE_DA = 0.01


def infer_neutral_masses(
    precursor_mz: np.ndarray,
    adducts: np.ndarray,
) -> np.ndarray:
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


def build_reference_pool(
    metadata: pd.DataFrame,
    manifest: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build exactly the same reference pool as the retrieval-dev
    baselines by removing each benchmark structure's designated
    query-library spectra.
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

    return (
        metadata.iloc[
            reference_indices
        ]
        .reset_index(drop=True)
    )


def truth_survives_mass_filter(
    truth_key: str,
    query_masses: np.ndarray,
    reference_keys: np.ndarray,
    reference_masses: np.ndarray,
    tolerance_da: float,
) -> bool:
    """
    Return True if any reference spectrum belonging to the truth
    structure has a neutral mass within tolerance of any query
    neutral mass.

    Polarity is deliberately ignored here.
    """
    query_masses = query_masses[
        np.isfinite(
            query_masses
        )
    ]

    if len(query_masses) == 0:
        return False

    truth_mask = (
        reference_keys
        == truth_key
    )

    truth_masses = (
        reference_masses[
            truth_mask
        ]
    )

    truth_masses = truth_masses[
        np.isfinite(
            truth_masses
        )
    ]

    if len(truth_masses) == 0:
        return False

    for query_mass in query_masses:
        if np.any(
            np.abs(
                truth_masses
                - query_mass
            )
            <= tolerance_da
        ):
            return True

    return False


def main() -> None:
    print(
        "Loading metadata and existing results..."
    )

    metadata = pd.read_parquet(
        INDEX_METADATA_PATH
    )

    manifest = pd.read_parquet(
        MANIFEST_PATH
    )

    mass_analysis = pd.read_parquet(
        MASS_ANALYSIS_PATH
    )

    baseline1 = pd.read_parquet(
        BASELINE1_PATH
    )

    print(
        f"Metadata spectra: "
        f"{len(metadata):,}"
    )

    print(
        f"Dev structures:   "
        f"{len(manifest):,}"
    )

    # ---------------------------------------------------------
    # Build the same frozen reference pool.
    # ---------------------------------------------------------

    print()
    print(
        "Building reference pool..."
    )

    reference = build_reference_pool(
        metadata=metadata,
        manifest=manifest,
    )

    removed = (
        len(metadata)
        - len(reference)
    )

    print(
        f"Reference spectra: "
        f"{len(reference):,}"
    )

    print(
        f"Query spectra removed: "
        f"{removed:,}"
    )

    if removed != 19_812:
        raise RuntimeError(
            "Expected 19,812 removed query "
            f"spectra; got {removed:,}"
        )

    # ---------------------------------------------------------
    # Calculate reference neutral masses once.
    # ---------------------------------------------------------

    print()
    print(
        "Inferring reference neutral masses..."
    )

    reference_keys = (
        reference[
            "inchikey14"
        ]
        .astype(str)
        .to_numpy()
    )

    reference_masses = (
        infer_neutral_masses(
            reference[
                "precursor_mz"
            ].to_numpy(),
            reference[
                "adduct"
            ]
            .astype(str)
            .to_numpy(),
        )
    )

    known_reference_masses = int(
        np.count_nonzero(
            np.isfinite(
                reference_masses
            )
        )
    )

    print(
        f"Known reference masses: "
        f"{known_reference_masses:,} / "
        f"{len(reference):,}"
    )

    # ---------------------------------------------------------
    # Join previous results.
    # ---------------------------------------------------------

    previous = (
        mass_analysis.merge(
            baseline1[
                [
                    "inchikey14",
                    "rank",
                ]
            ],
            on="inchikey14",
            how="left",
            validate="one_to_one",
        )
        .rename(
            columns={
                "rank": "baseline1_rank",
            }
        )
    )

    problematic = (
        previous[
            previous[
                "status"
            ].isin(
                [
                    "no_query_mass",
                    "zero_candidates",
                    "truth_excluded",
                ]
            )
        ]
        .copy()
    )

    print()
    print(
        f"Problematic structures: "
        f"{len(problematic):,}"
    )

    if len(problematic) != 98:
        raise RuntimeError(
            "Expected 98 problematic "
            f"structures; got {len(problematic)}"
        )

    # ---------------------------------------------------------
    # Prepare full metadata arrays for query lookup.
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

    results = []

    print()
    print(
        "Testing cross-polarity truth coverage..."
    )

    # ---------------------------------------------------------
    # Only the 98 problematic structures need analysis.
    # ---------------------------------------------------------

    for position, row in enumerate(
        problematic.itertuples(
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

        query_rows = (
            metadata.iloc[
                np.flatnonzero(
                    query_mask
                )
            ]
        )

        query_masses = (
            infer_neutral_masses(
                query_rows[
                    "precursor_mz"
                ].to_numpy(),
                query_rows[
                    "adduct"
                ]
                .astype(str)
                .to_numpy(),
            )
        )

        cross_polarity_survived = (
            truth_survives_mass_filter(
                truth_key=truth_key,
                query_masses=query_masses,
                reference_keys=(
                    reference_keys
                ),
                reference_masses=(
                    reference_masses
                ),
                tolerance_da=(
                    TOLERANCE_DA
                ),
            )
        )

        results.append(
            {
                "inchikey14": truth_key,
                "query_library": (
                    query_library
                ),
                "previous_status": (
                    row.status
                ),
                "baseline1_rank": (
                    row.baseline1_rank
                ),
                "known_query_mass_count": int(
                    np.count_nonzero(
                        np.isfinite(
                            query_masses
                        )
                    )
                ),
                "cross_polarity_truth_survived": (
                    cross_polarity_survived
                ),
            }
        )

        if (
            position % 10 == 0
            or position == len(problematic)
        ):
            print(
                f"{position:>3}/"
                f"{len(problematic)}"
            )

    results_frame = pd.DataFrame(
        results
    )

    RESULTS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_frame.to_parquet(
        RESULTS_PATH,
        index=False,
    )

    # ---------------------------------------------------------
    # Summary.
    # ---------------------------------------------------------

    rescued = (
        results_frame[
            "cross_polarity_truth_survived"
        ]
    )

    print()
    print(
        "Cross-polarity mass coverage"
    )
    print(
        "----------------------------"
    )

    print(
        f"Problematic structures: "
        f"{len(results_frame):,}"
    )

    print(
        f"Recovered by ±"
        f"{TOLERANCE_DA:.3f} Da "
        f"cross-polarity mass: "
        f"{rescued.sum():,} / "
        f"{len(results_frame):,} "
        f"({rescued.mean():.2%})"
    )

    # ---------------------------------------------------------
    # Breakdown by previous mass-filter status.
    # ---------------------------------------------------------

    print()
    print(
        "Recovery by previous status"
    )
    print(
        "---------------------------"
    )

    for status in [
        "no_query_mass",
        "zero_candidates",
        "truth_excluded",
    ]:
        subset = (
            results_frame[
                results_frame[
                    "previous_status"
                ]
                == status
            ]
        )

        count = int(
            subset[
                "cross_polarity_truth_survived"
            ].sum()
        )

        print(
            f"{status:<18} "
            f"{count:>3}/"
            f"{len(subset):<3} "
            f"({count / len(subset):.2%})"
        )

    # ---------------------------------------------------------
    # The critical 17 Baseline-1-unavailable structures.
    # ---------------------------------------------------------

    unavailable = (
        results_frame[
            results_frame[
                "baseline1_rank"
            ].isna()
        ]
    )

    unavailable_rescued = int(
        unavailable[
            "cross_polarity_truth_survived"
        ].sum()
    )

    print()
    print(
        "No same-polarity Baseline 1 truth"
    )
    print(
        "---------------------------------"
    )

    print(
        f"Structures: "
        f"{len(unavailable):,}"
    )

    print(
        f"Recovered by cross-polarity mass: "
        f"{unavailable_rescued:,} / "
        f"{len(unavailable):,} "
        f"("
        f"{unavailable_rescued / len(unavailable):.2%}"
        f")"
    )

    print()

    print(
        unavailable[
            [
                "inchikey14",
                "query_library",
                "previous_status",
                "known_query_mass_count",
                "cross_polarity_truth_survived",
            ]
        ].to_string(
            index=False
        )
    )

    # ---------------------------------------------------------
    # Estimate hybrid candidate-generation truth coverage:
    #
    #   same-polarity mass
    #   + cross-polarity mass
    #   + Baseline 1 Top-25
    #
    # We already know the 1,902 truth-retained structures are
    # covered by same-polarity mass.
    # ---------------------------------------------------------

    spectral_top25 = (
        results_frame[
            "baseline1_rank"
        ].notna()
        & (
            results_frame[
                "baseline1_rank"
            ]
            <= 25
        )
    )

    hybrid_problem_recovered = (
        rescued
        | spectral_top25
    )

    hybrid_problem_count = int(
        hybrid_problem_recovered.sum()
    )

    overall_covered = (
        1_902
        + hybrid_problem_count
    )

    overall_coverage = (
        overall_covered
        / 2_000
    )

    print()
    print(
        "Proposed hybrid coverage"
    )
    print(
        "------------------------"
    )

    print(
        "same-polarity mass ±0.01"
    )

    print(
        "+ cross-polarity mass ±0.01"
    )

    print(
        "+ same-polarity spectral Top-25"
    )

    print()

    print(
        f"Problematic recovered: "
        f"{hybrid_problem_count:,} / 98"
    )

    print(
        f"Overall truth coverage: "
        f"{overall_covered:,} / 2,000 "
        f"({overall_coverage:.2%})"
    )

    # ---------------------------------------------------------
    # Show anything still not covered.
    # ---------------------------------------------------------

    remaining = (
        results_frame[
            ~hybrid_problem_recovered
        ]
        .copy()
    )

    print()
    print(
        "Still uncovered by proposed hybrid"
    )
    print(
        "----------------------------------"
    )

    print(
        f"Structures: "
        f"{len(remaining):,}"
    )

    if len(remaining):
        print()

        print(
            remaining[
                [
                    "inchikey14",
                    "query_library",
                    "previous_status",
                    "baseline1_rank",
                    "known_query_mass_count",
                ]
            ]
            .sort_values(
                "baseline1_rank",
                na_position="first",
            )
            .to_string(
                index=False
            )
        )

    print()
    print(
        f"Results written to: "
        f"{RESULTS_PATH}"
    )


if __name__ == "__main__":
    main()