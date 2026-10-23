from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(
    "data/processed/diagnostics/large_validation_audit"
)

BENCHMARK_DIR = (
    ROOT
    / "true_formula_neighbor_benchmark"
)

QUERY_SPLIT_PATH = (
    BENCHMARK_DIR
    / "learned_ranker/query_split.parquet"
)

CANDIDATES_PATH = (
    BENCHMARK_DIR
    / "top3_neighbor_collapse/top3_neighbor_top100.parquet"
)

HYPOTHESES_PATH = (
    ROOT
    / "formula_hypotheses.parquet"
)

PUBCHEM_COVERAGE_PATH = (
    ROOT
    / "pubchem_coverage.parquet"
)

OUTPUT_PATH = (
    BENCHMARK_DIR
    / "learned_ranker/holdout_candidate_miss_diagnostic.parquet"
)


def main() -> None:
    for path in [
        QUERY_SPLIT_PATH,
        CANDIDATES_PATH,
        HYPOTHESES_PATH,
        PUBCHEM_COVERAGE_PATH,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"Missing required artifact: {path}"
            )

    split = pd.read_parquet(
        QUERY_SPLIT_PATH
    )

    candidates = pd.read_parquet(
        CANDIDATES_PATH
    )

    hypotheses = pd.read_parquet(
        HYPOTHESES_PATH
    )

    coverage = pd.read_parquet(
        PUBCHEM_COVERAGE_PATH
    )

    holdout = (
        split.loc[
            split[
                "split"
            ]
            == "holdout",
            [
                "inchikey14",
                "truth_present",
            ],
        ]
        .copy()
    )

    holdout[
        "inchikey14"
    ] = holdout[
        "inchikey14"
    ].astype(str)

    holdout_keys = set(
        holdout[
            "inchikey14"
        ]
    )

    hypotheses[
        "inchikey14"
    ] = hypotheses[
        "inchikey14"
    ].astype(str)

    hypotheses = hypotheses.loc[
        hypotheses[
            "inchikey14"
        ]
        .isin(
            holdout_keys
        )
    ].copy()

    hypotheses = (
        hypotheses
        .sort_values(
            [
                "inchikey14",
                "mass_error_best_abs_ppm",
                "formula_support_count",
                "candidate_formula",
            ],
            ascending=[
                True,
                True,
                False,
                True,
            ],
            kind="stable",
        )
    )

    hypotheses[
        "formula_rank"
    ] = (
        hypotheses.groupby(
            "inchikey14"
        )
        .cumcount()
        + 1
    )

    truth_formula = (
        hypotheses.loc[
            hypotheses[
                "is_truth_formula"
            ].astype(bool),
            [
                "inchikey14",
                "candidate_formula",
                "formula_rank",
            ],
        ]
        .drop_duplicates(
            subset=[
                "inchikey14"
            ]
        )
        .rename(
            columns={
                "candidate_formula":
                    "truth_formula",
                "formula_rank":
                    "truth_formula_rank",
            }
        )
    )

    candidates[
        "inchikey14"
    ] = candidates[
        "inchikey14"
    ].astype(str)

    candidates[
        "candidate_inchikey14"
    ] = candidates[
        "candidate_inchikey14"
    ].astype(str)

    candidates = candidates.loc[
        candidates[
            "inchikey14"
        ]
        .isin(
            holdout_keys
        )
    ].copy()

    retained_truth = (
        candidates.loc[
            candidates[
                "candidate_inchikey14"
            ]
            == candidates[
                "inchikey14"
            ],
            [
                "inchikey14",
                "prior_rank",
                "formula_rank",
            ],
        ]
        .drop_duplicates(
            subset=[
                "inchikey14"
            ]
        )
        .rename(
            columns={
                "prior_rank":
                    "truth_prior_rank_top100",
                "formula_rank":
                    "retained_formula_rank",
            }
        )
    )

    coverage[
        "inchikey14"
    ] = coverage[
        "inchikey14"
    ].astype(str)

    coverage = coverage.loc[
        coverage[
            "inchikey14"
        ]
        .isin(
            holdout_keys
        ),
        [
            "inchikey14",
            "truth_formula_recovered",
            "truth_formula_in_pubchem_asset",
            "truth_structure_in_pubchem_hypotheses",
            "truth_formula_pubchem_candidate_count",
        ],
    ].copy()

    result = (
        holdout
        .merge(
            truth_formula,
            on="inchikey14",
            how="left",
            validate="one_to_one",
        )
        .merge(
            coverage,
            on="inchikey14",
            how="left",
            validate="one_to_one",
        )
        .merge(
            retained_truth,
            on="inchikey14",
            how="left",
            validate="one_to_one",
        )
    )

    result[
        "truth_formula_recovered"
    ] = (
        result[
            "truth_formula_recovered"
        ]
        .fillna(False)
        .astype(bool)
    )

    result[
        "truth_formula_in_pubchem_asset"
    ] = (
        result[
            "truth_formula_in_pubchem_asset"
        ]
        .fillna(False)
        .astype(bool)
    )

    result[
        "truth_structure_in_pubchem_hypotheses"
    ] = (
        result[
            "truth_structure_in_pubchem_hypotheses"
        ]
        .fillna(False)
        .astype(bool)
    )

    result[
        "truth_retained_top100"
    ] = (
        result[
            "truth_prior_rank_top100"
        ]
        .notna()
    )

    def classify(
        row: pd.Series,
    ) -> str:
        if bool(
            row[
                "truth_retained_top100"
            ]
        ):
            return "retained_top100"

        if not bool(
            row[
                "truth_formula_recovered"
            ]
        ):
            return "truth_formula_not_recovered"

        if (
            pd.isna(
                row[
                    "truth_formula_rank"
                ]
            )
            or int(
                row[
                    "truth_formula_rank"
                ]
            )
            > 3
        ):
            return "truth_formula_outside_top3"

        if not bool(
            row[
                "truth_formula_in_pubchem_asset"
            ]
        ):
            return "truth_formula_absent_pubchem_asset"

        if not bool(
            row[
                "truth_structure_in_pubchem_hypotheses"
            ]
        ):
            return "truth_structure_absent_pubchem"

        return "truth_below_top100"

    result[
        "miss_reason"
    ] = result.apply(
        classify,
        axis=1,
    )

    result.to_parquet(
        OUTPUT_PATH,
        index=False,
        compression="zstd",
    )

    counts = (
        result[
            "miss_reason"
        ]
        .value_counts()
        .rename_axis(
            "reason"
        )
        .reset_index(
            name="queries"
        )
    )

    counts[
        "percent"
    ] = (
        counts[
            "queries"
        ]
        / len(
            result
        )
    )

    misses = result.loc[
        ~result[
            "truth_retained_top100"
        ]
    ]

    miss_counts = (
        misses[
            "miss_reason"
        ]
        .value_counts()
        .rename_axis(
            "reason"
        )
        .reset_index(
            name="queries"
        )
    )

    miss_counts[
        "percent_of_misses"
    ] = (
        miss_counts[
            "queries"
        ]
        / len(
            misses
        )
        if len(
            misses
        )
        else 0.0
    )

    print()
    print(
        "Holdout candidate-miss diagnostic"
    )
    print(
        "=" * 78
    )
    print(
        "Holdout queries:",
        f"{len(result):,}",
    )
    print(
        "Truth retained in top-100 candidate rows:",
        f"{int(result['truth_retained_top100'].sum()):,} "
        f"({result['truth_retained_top100'].mean():.2%})",
    )
    print(
        "Missing truths:",
        f"{len(misses):,}",
    )

    print()
    print(
        "All holdout queries"
    )
    print(
        "-" * 78
    )
    print(
        counts.to_string(
            index=False,
            formatters={
                "percent":
                    lambda value:
                    f"{value:.2%}",
            },
        )
    )

    print()
    print(
        "Missing truths only"
    )
    print(
        "-" * 78
    )
    print(
        miss_counts.to_string(
            index=False,
            formatters={
                "percent_of_misses":
                    lambda value:
                    f"{value:.2%}",
            },
        )
    )

    top3_pubchem = result.loc[
        result[
            "truth_formula_rank"
        ].between(
            1,
            3,
            inclusive="both",
        )
        & result[
            "truth_structure_in_pubchem_hypotheses"
        ]
    ]

    if not top3_pubchem.empty:
        print()
        print(
            "Top-3 formula + PubChem truth population"
        )
        print(
            "-" * 78
        )
        print(
            "Queries:",
            f"{len(top3_pubchem):,}",
        )
        print(
            "Truth retained top-100:",
            f"{int(top3_pubchem['truth_retained_top100'].sum()):,} "
            f"({top3_pubchem['truth_retained_top100'].mean():.2%})",
        )
        print(
            "Truth below top-100:",
            f"{int((~top3_pubchem['truth_retained_top100']).sum()):,} "
            f"({(~top3_pubchem['truth_retained_top100']).mean():.2%})",
        )

    print()
    print(
        "Truth-formula PubChem burden for below-top100 misses"
    )
    print(
        "-" * 78
    )

    below = result.loc[
        result[
            "miss_reason"
        ]
        == "truth_below_top100",
        "truth_formula_pubchem_candidate_count",
    ]

    if below.empty:
        print(
            "No below-top100 misses."
        )
    else:
        print(
            below.describe(
                percentiles=[
                    0.5,
                    0.75,
                    0.9,
                    0.95,
                    0.99,
                ]
            ).to_string()
        )

    print()
    print(
        "Wrote:",
        OUTPUT_PATH,
    )


if __name__ == "__main__":
    main()
