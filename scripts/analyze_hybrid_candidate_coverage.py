from pathlib import Path

import pandas as pd


RESULTS_DIR = Path(
    "data/processed/results"
)

BASELINE1_PATH = (
    RESULTS_DIR
    / "retrieval_dev_baseline1.parquet"
)

MASS_ANALYSIS_PATH = (
    RESULTS_DIR
    / "retrieval_dev_mass_filter_analysis.parquet"
)


SPECTRAL_CUTOFFS = [
    25,
    100,
    500,
    1_000,
    5_000,
    10_000,
    25_000,
    50_000,
    100_000,
    250_000,
]


def main() -> None:
    print(
        "Loading existing results..."
    )

    baseline = pd.read_parquet(
        BASELINE1_PATH
    )

    mass = pd.read_parquet(
        MASS_ANALYSIS_PATH
    )

    print(
        f"Baseline 1 rows: "
        f"{len(baseline):,}"
    )

    print(
        f"Mass-analysis rows: "
        f"{len(mass):,}"
    )

    print()
    print(
        "Baseline columns:"
    )
    print(
        list(baseline.columns)
    )

    # ---------------------------------------------------------
    # Normalize the Baseline 1 rank column.
    #
    # Our previous runner should have written `rank`, but this
    # check makes the script fail clearly if the schema differs.
    # ---------------------------------------------------------

    if "rank" not in baseline.columns:
        raise RuntimeError(
            "Baseline 1 results do not contain "
            "a 'rank' column. "
            f"Columns are: {list(baseline.columns)}"
        )

    required_mass_columns = {
        "inchikey14",
        "query_library",
        "status",
        "candidate_count",
        "truth_survived",
    }

    missing = (
        required_mass_columns
        - set(
            mass.columns
        )
    )

    if missing:
        raise RuntimeError(
            "Mass analysis is missing columns: "
            f"{sorted(missing)}"
        )

    # ---------------------------------------------------------
    # One row per dev structure should exist in both files.
    # ---------------------------------------------------------

    if baseline[
        "inchikey14"
    ].duplicated().any():
        raise RuntimeError(
            "Baseline 1 contains duplicate "
            "inchikey14 rows."
        )

    if mass[
        "inchikey14"
    ].duplicated().any():
        raise RuntimeError(
            "Mass analysis contains duplicate "
            "inchikey14 rows."
        )

    joined = mass.merge(
        baseline[
            [
                "inchikey14",
                "rank",
            ]
        ],
        on="inchikey14",
        how="left",
        validate="one_to_one",
    )

    if len(joined) != 2_000:
        raise RuntimeError(
            "Expected 2,000 joined structures; "
            f"got {len(joined):,}"
        )

    joined = joined.rename(
        columns={
            "rank": (
                "baseline1_rank"
            ),
        }
    )

    # ---------------------------------------------------------
    # Define the mass-filter failure population.
    # ---------------------------------------------------------

    problematic_statuses = {
        "no_query_mass",
        "zero_candidates",
        "truth_excluded",
    }

    problematic = (
        joined[
            joined[
                "status"
            ].isin(
                problematic_statuses
            )
        ]
        .copy()
    )

    retained = (
        joined[
            joined[
                "status"
            ]
            == "truth_retained"
        ]
        .copy()
    )

    print()
    print(
        "Mass-filter populations"
    )
    print(
        "-----------------------"
    )

    print(
        f"Truth retained: "
        f"{len(retained):,}"
    )

    print(
        f"Problematic:     "
        f"{len(problematic):,}"
    )

    print()

    print(
        problematic[
            "status"
        ]
        .value_counts()
        .to_string()
    )

    if (
        len(retained)
        + len(problematic)
        != len(joined)
    ):
        raise RuntimeError(
            "Unexpected mass-filter status "
            "encountered."
        )

    # ---------------------------------------------------------
    # Baseline 1 availability.
    # ---------------------------------------------------------

    finite_rank = (
        problematic[
            "baseline1_rank"
        ].notna()
    )

    print()
    print(
        "Baseline 1 ranks for problematic cases"
    )
    print(
        "--------------------------------------"
    )

    print(
        f"Finite truth rank: "
        f"{finite_rank.sum():,} / "
        f"{len(problematic):,}"
    )

    print(
        f"No truth rank:     "
        f"{(~finite_rank).sum():,} / "
        f"{len(problematic):,}"
    )

    # ---------------------------------------------------------
    # Hybrid coverage.
    #
    # A structure is covered if:
    #
    #   1. mass filtering retained truth
    #
    # OR
    #
    #   2. Baseline 1 ranked truth within spectral Top-N.
    #
    # This estimates candidate-generation recall without
    # rerunning cosine retrieval.
    # ---------------------------------------------------------

    print()
    print(
        "Hybrid candidate coverage"
    )
    print(
        "-------------------------"
    )

    print(
        "Mass candidates + "
        "Baseline 1 spectral Top-N"
    )

    print()

    print(
        f"{'Top-N':>10} "
        f"{'Problem recovered':>20} "
        f"{'Problem coverage':>18} "
        f"{'Overall coverage':>18}"
    )

    print(
        "-" * 70
    )

    for cutoff in SPECTRAL_CUTOFFS:
        recovered_problem = (
            problematic[
                "baseline1_rank"
            ].notna()
            & (
                problematic[
                    "baseline1_rank"
                ]
                <= cutoff
            )
        )

        recovered_count = int(
            recovered_problem.sum()
        )

        problem_coverage = (
            recovered_count
            / len(problematic)
        )

        overall_covered = (
            len(retained)
            + recovered_count
        )

        overall_coverage = (
            overall_covered
            / len(joined)
        )

        print(
            f"{cutoff:>10,} "
            f"{recovered_count:>8}/"
            f"{len(problematic):<8} "
            f"{problem_coverage:>17.2%} "
            f"{overall_coverage:>17.2%}"
        )

    # ---------------------------------------------------------
    # Break down spectral recovery by mass-filter status.
    # ---------------------------------------------------------

    print()
    print(
        "Coverage by failure type"
    )
    print(
        "------------------------"
    )

    for status in [
        "no_query_mass",
        "zero_candidates",
        "truth_excluded",
    ]:
        subset = (
            problematic[
                problematic[
                    "status"
                ]
                == status
            ]
        )

        print()
        print(
            f"{status} "
            f"(N={len(subset)})"
        )

        for cutoff in [
            25,
            100,
            500,
            1_000,
            10_000,
            250_000,
        ]:
            recovered = int(
                (
                    subset[
                        "baseline1_rank"
                    ].notna()
                    & (
                        subset[
                            "baseline1_rank"
                        ]
                        <= cutoff
                    )
                ).sum()
            )

            print(
                f"  Top-{cutoff:<7,} "
                f"{recovered:>3}/"
                f"{len(subset):<3} "
                f"("
                f"{recovered / len(subset):.2%}"
                f")"
            )

    # ---------------------------------------------------------
    # Rank distribution for problematic cases.
    # ---------------------------------------------------------

    ranked = (
        problematic[
            problematic[
                "baseline1_rank"
            ].notna()
        ]
    )

    print()
    print(
        "Problematic Baseline 1 rank distribution"
    )
    print(
        "----------------------------------------"
    )

    if len(ranked):
        ranks = (
            ranked[
                "baseline1_rank"
            ]
        )

        print(
            f"Min:       "
            f"{ranks.min():,.0f}"
        )

        print(
            f"Median:    "
            f"{ranks.median():,.0f}"
        )

        print(
            f"75th pct:  "
            f"{ranks.quantile(0.75):,.0f}"
        )

        print(
            f"90th pct:  "
            f"{ranks.quantile(0.90):,.0f}"
        )

        print(
            f"95th pct:  "
            f"{ranks.quantile(0.95):,.0f}"
        )

        print(
            f"99th pct:  "
            f"{ranks.quantile(0.99):,.0f}"
        )

        print(
            f"Max:       "
            f"{ranks.max():,.0f}"
        )

    # ---------------------------------------------------------
    # Show the hardest cases.
    # ---------------------------------------------------------

    print()
    print(
        "Worst Baseline 1 ranks among "
        "mass-filter failures"
    )
    print(
        "---------------------------------------"
    )

    hardest = (
        problematic
        .sort_values(
            "baseline1_rank",
            ascending=False,
            na_position="first",
        )
    )

    print(
        hardest[
            [
                "inchikey14",
                "query_library",
                "status",
                "candidate_count",
                "baseline1_rank",
            ]
        ]
        .head(30)
        .to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()