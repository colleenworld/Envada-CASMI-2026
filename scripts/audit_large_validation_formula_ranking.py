from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd


TOP_KS = [1, 2, 3, 5]


def rr_at_k(rank: float | int | None, k: int) -> float:
    if rank is None or pd.isna(rank):
        return 0.0
    rank = int(rank)
    if rank < 1 or rank > k:
        return 0.0
    return 1.0 / rank


def main() -> None:
    project_root = Path(
        os.environ.get(
            "CASMI26_PROJECT_ROOT",
            Path(__file__).resolve().parent.parent,
        )
    )

    audit_dir = (
        project_root
        / "data/processed/diagnostics/large_validation_audit"
    )

    hypotheses_path = (
        audit_dir / "formula_hypotheses.parquet"
    )
    coverage_path = (
        audit_dir / "pubchem_coverage.parquet"
    )

    if not hypotheses_path.exists():
        raise FileNotFoundError(
            f"Missing {hypotheses_path}"
        )

    if not coverage_path.exists():
        raise FileNotFoundError(
            f"Missing {coverage_path}"
        )

    hypotheses = pd.read_parquet(
        hypotheses_path
    )

    coverage = pd.read_parquet(
        coverage_path
    )

    required = {
        "inchikey14",
        "candidate_formula",
        "is_truth_formula",
        "formula_support_count",
        "formula_support_fraction",
        "mass_error_best_abs_ppm",
    }

    missing = required - set(
        hypotheses.columns
    )

    if missing:
        raise ValueError(
            "Hypothesis artifact is missing columns: "
            f"{sorted(missing)}"
        )

    #
    # Candidate formula ranking strategies.
    #
    # These are deliberately simple diagnostics using only signals
    # already available before MIST:
    #
    # 1. support count desc, best mass error asc
    # 2. support fraction desc, best mass error asc
    # 3. best mass error asc, support count desc
    #
    strategies = {
        "support_count": (
            [
                "formula_support_count",
                "mass_error_best_abs_ppm",
                "candidate_formula",
            ],
            [
                False,
                True,
                True,
            ],
        ),
        "support_fraction": (
            [
                "formula_support_fraction",
                "mass_error_best_abs_ppm",
                "candidate_formula",
            ],
            [
                False,
                True,
                True,
            ],
        ),
        "mass_error": (
            [
                "mass_error_best_abs_ppm",
                "formula_support_count",
                "candidate_formula",
            ],
            [
                True,
                False,
                True,
            ],
        ),
    }

    truth_formula_present = (
        hypotheses.loc[
            hypotheses[
                "is_truth_formula"
            ].astype(bool),
            "inchikey14",
        ]
        .astype(str)
        .nunique()
    )

    total_queries = (
        coverage[
            "inchikey14"
        ]
        .astype(str)
        .nunique()
    )

    print(
        "Formula-ranking audit"
    )
    print(
        "====================="
    )
    print(
        "Queries:",
        f"{total_queries:,}",
    )
    print(
        "Queries with recovered truth formula:",
        f"{truth_formula_present:,} / {total_queries:,} "
        f"({truth_formula_present / total_queries:.2%})",
    )

    summary_rows = []
    ranked_frames = []

    for (
        strategy_name,
        (
            sort_columns,
            ascending,
        ),
    ) in strategies.items():
        ranked = (
            hypotheses
            .sort_values(
                [
                    "inchikey14",
                    *sort_columns,
                ],
                ascending=[
                    True,
                    *ascending,
                ],
                kind="stable",
            )
            .copy()
        )

        ranked[
            "formula_rank"
        ] = (
            ranked.groupby(
                "inchikey14"
            )
            .cumcount()
            + 1
        )

        truth = (
            ranked.loc[
                ranked[
                    "is_truth_formula"
                ].astype(bool),
                [
                    "inchikey14",
                    "formula_rank",
                ],
            ]
            .drop_duplicates(
                subset=["inchikey14"],
                keep="first",
            )
        )

        metrics = coverage[
            [
                "inchikey14",
                "dominant_library",
                "dominant_adduct",
                "dominant_instrument",
                "truth_formula_recovered",
                "truth_structure_in_pubchem_hypotheses",
            ]
        ].merge(
            truth,
            on="inchikey14",
            how="left",
            validate="one_to_one",
        )

        result = {
            "strategy":
                strategy_name,
            "mrr_at_5":
                metrics[
                    "formula_rank"
                ].map(
                    lambda rank:
                        rr_at_k(
                            rank,
                            5,
                        )
                ).mean(),
        }

        for k in TOP_KS:
            result[
                f"top{k}"
            ] = (
                metrics[
                    "formula_rank"
                ]
                .between(
                    1,
                    k,
                    inclusive="both",
                )
                .mean()
            )

        summary_rows.append(
            result
        )

        truth[
            "strategy"
        ] = strategy_name

        ranked_frames.append(
            truth
        )

    summary = pd.DataFrame(
        summary_rows
    )

    print()
    print(
        "Overall formula ranking"
    )
    print(
        "-----------------------"
    )

    print(
        summary.to_string(
            index=False,
            formatters={
                "mrr_at_5":
                    lambda value:
                    f"{value:.4f}",
                "top1":
                    lambda value:
                    f"{value:.2%}",
                "top2":
                    lambda value:
                    f"{value:.2%}",
                "top3":
                    lambda value:
                    f"{value:.2%}",
                "top5":
                    lambda value:
                    f"{value:.2%}",
            },
        )
    )

    #
    # Also estimate the PubChem candidate burden retained if we keep
    # only the top N formulas according to each strategy.
    #
    pubchem_counts = pd.read_parquet(
        hypotheses_path
    )

    #
    # pubchem_candidate_count was attached to the in-memory hypothesis
    # table by the PubChem coverage script but the original hypothesis
    # artifact remains intentionally source-only. Recover counts by
    # joining from formula-level values encoded in coverage is not
    # possible, so read the PubChem coverage script's formula index
    # indirectly from the bucket metadata/candidates.
    #
    pubchem_root = (
        project_root
        / "data/processed/pubchem_formula_index"
    )
    formula_buckets = pd.read_parquet(
        pubchem_root
        / "formula_buckets.parquet"
    )[
        [
            "molecular_formula",
            "bucket",
        ]
    ]

    requested = set(
        hypotheses[
            "candidate_formula"
        ]
        .astype(str)
    )

    formula_candidate_counts = {}

    selected = formula_buckets.loc[
        formula_buckets[
            "molecular_formula"
        ]
        .astype(str)
        .isin(requested)
    ]

    for bucket in sorted(
        selected[
            "bucket"
        ]
        .astype(int)
        .unique()
    ):
        formulas_in_bucket = set(
            selected.loc[
                selected[
                    "bucket"
                ]
                .astype(int)
                == bucket,
                "molecular_formula",
            ]
            .astype(str)
        )

        frame = pd.read_parquet(
            pubchem_root
            / "candidates"
            / f"bucket-{bucket:03d}.parquet",
            columns=[
                "molecular_formula",
                "inchikey14",
            ],
        )

        frame[
            "molecular_formula"
        ] = (
            frame[
                "molecular_formula"
            ]
            .astype(str)
        )

        frame = frame.loc[
            frame[
                "molecular_formula"
            ]
            .isin(
                formulas_in_bucket
            )
        ]

        if frame.empty:
            continue

        counts = (
            frame
            .drop_duplicates(
                subset=[
                    "molecular_formula",
                    "inchikey14",
                ]
            )
            .groupby(
                "molecular_formula"
            )[
                "inchikey14"
            ]
            .nunique()
        )

        for formula, count in counts.items():
            formula_candidate_counts[
                str(formula)
            ] = int(
                count
            )

    print()
    print(
        "Candidate burden after formula gating"
    )
    print(
        "-------------------------------------"
    )

    burden_rows = []

    for (
        strategy_name,
        (
            sort_columns,
            ascending,
        ),
    ) in strategies.items():
        ranked = (
            hypotheses
            .sort_values(
                [
                    "inchikey14",
                    *sort_columns,
                ],
                ascending=[
                    True,
                    *ascending,
                ],
                kind="stable",
            )
            .copy()
        )

        ranked[
            "formula_rank"
        ] = (
            ranked.groupby(
                "inchikey14"
            )
            .cumcount()
            + 1
        )

        ranked[
            "pubchem_candidate_count"
        ] = (
            ranked[
                "candidate_formula"
            ]
            .astype(str)
            .map(
                formula_candidate_counts
            )
            .fillna(0)
            .astype(np.int64)
        )

        for top_n in [
            1,
            2,
            3,
            5,
        ]:
            retained = (
                ranked.loc[
                    ranked[
                        "formula_rank"
                    ]
                    <= top_n
                ]
                .groupby(
                    "inchikey14"
                )[
                    "pubchem_candidate_count"
                ]
                .sum()
                .reindex(
                    coverage[
                        "inchikey14"
                    ],
                    fill_value=0,
                )
            )

            truth_kept = (
                ranked.loc[
                    ranked[
                        "is_truth_formula"
                    ].astype(bool)
                    & (
                        ranked[
                            "formula_rank"
                        ]
                        <= top_n
                    ),
                    "inchikey14",
                ]
                .astype(str)
                .nunique()
                / total_queries
            )

            burden_rows.append(
                {
                    "strategy":
                        strategy_name,
                    "top_formulas":
                        top_n,
                    "truth_formula_recall":
                        truth_kept,
                    "mean_candidates":
                        retained.mean(),
                    "median_candidates":
                        retained.median(),
                    "p90_candidates":
                        retained.quantile(
                            0.90
                        ),
                    "p95_candidates":
                        retained.quantile(
                            0.95
                        ),
                }
            )

    burden = pd.DataFrame(
        burden_rows
    )

    print(
        burden.to_string(
            index=False,
            formatters={
                "truth_formula_recall":
                    lambda value:
                    f"{value:.2%}",
                "mean_candidates":
                    lambda value:
                    f"{value:,.0f}",
                "median_candidates":
                    lambda value:
                    f"{value:,.0f}",
                "p90_candidates":
                    lambda value:
                    f"{value:,.0f}",
                "p95_candidates":
                    lambda value:
                    f"{value:,.0f}",
            },
        )
    )

    #
    # Detailed breakdown for the most production-like strategy:
    # support count, then mass error.
    #
    best_name = "support_count"
    sort_columns, ascending = strategies[
        best_name
    ]

    ranked = (
        hypotheses
        .sort_values(
            [
                "inchikey14",
                *sort_columns,
            ],
            ascending=[
                True,
                *ascending,
            ],
            kind="stable",
        )
        .copy()
    )

    ranked[
        "formula_rank"
    ] = (
        ranked.groupby(
            "inchikey14"
        )
        .cumcount()
        + 1
    )

    truth = (
        ranked.loc[
            ranked[
                "is_truth_formula"
            ].astype(bool),
            [
                "inchikey14",
                "formula_rank",
            ],
        ]
        .drop_duplicates(
            subset=["inchikey14"]
        )
    )

    detailed = coverage.merge(
        truth,
        on="inchikey14",
        how="left",
        validate="one_to_one",
    )

    def grouped(
        column: str,
    ) -> pd.DataFrame:
        frame = (
            detailed.groupby(
                column,
                dropna=False,
            )
            .agg(
                structures=(
                    "inchikey14",
                    "size",
                ),
                formula_recall=(
                    "truth_formula_recovered",
                    "mean",
                ),
                formula_top1=(
                    "formula_rank",
                    lambda values:
                        values.eq(
                            1
                        ).mean(),
                ),
                formula_top3=(
                    "formula_rank",
                    lambda values:
                        values.between(
                            1,
                            3,
                            inclusive="both",
                        ).mean(),
                ),
                pubchem_structure_coverage=(
                    "truth_structure_in_pubchem_hypotheses",
                    "mean",
                ),
            )
            .reset_index()
        )

        return (
            frame.loc[
                frame[
                    "structures"
                ]
                >= 25
            ]
            .sort_values(
                [
                    "structures",
                    column,
                ],
                ascending=[
                    False,
                    True,
                ],
            )
        )

    for label, column in [
        (
            "By dominant library",
            "dominant_library",
        ),
        (
            "By dominant adduct",
            "dominant_adduct",
        ),
        (
            "By dominant instrument",
            "dominant_instrument",
        ),
    ]:
        print()
        print(label)
        print("-" * len(label))
        print(
            grouped(
                column
            ).to_string(
                index=False,
                formatters={
                    "formula_recall":
                        lambda value:
                        f"{value:.2%}",
                    "formula_top1":
                        lambda value:
                        f"{value:.2%}",
                    "formula_top3":
                        lambda value:
                        f"{value:.2%}",
                    "pubchem_structure_coverage":
                        lambda value:
                        f"{value:.2%}",
                },
            )
        )

    summary.to_csv(
        audit_dir
        / "formula_ranking_summary.csv",
        index=False,
    )
    burden.to_csv(
        audit_dir
        / "formula_gating_candidate_burden.csv",
        index=False,
    )

    print()
    print(
        "Wrote:"
    )
    print(
        audit_dir
        / "formula_ranking_summary.csv"
    )
    print(
        audit_dir
        / "formula_gating_candidate_burden.csv"
    )


if __name__ == "__main__":
    main()
