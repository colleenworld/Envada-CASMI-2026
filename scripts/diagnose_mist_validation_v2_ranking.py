from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(
    "data/processed/mist/validation_500_v2"
)

RANKED_PATH = (
    ROOT / "mist_ranked_candidates.parquet"
)

MANIFEST_PATH = (
    ROOT / "manifest.parquet"
)

OUTPUT_PATH = (
    ROOT / "mist_ranking_diagnostics.parquet"
)


def rank_bucket(rank):
    if pd.isna(rank):
        return "not_rankable"

    rank = int(rank)

    if rank == 1:
        return "rank_1"

    if rank <= 5:
        return "rank_2_5"

    if rank <= 25:
        return "rank_6_25"

    return "rank_gt_25"


def main():
    ranked = pd.read_parquet(
        RANKED_PATH
    )

    manifest = pd.read_parquet(
        MANIFEST_PATH
    )

    print(
        "Ranked candidate rows:",
        len(ranked),
    )

    print(
        "Queries in manifest:",
        len(manifest),
    )

    required = {
        "query_inchikey14",
        "candidate_inchikey14",
        "candidate_formula",
        "is_truth_structure",
        "rank",
        "score",
        "mist_cosine",
        "selected_mass_error_ppm",
        "formula_support_count",
        "formula_support_fraction",
    }

    missing = (
        required
        - set(ranked.columns)
    )

    if missing:
        raise ValueError(
            "Ranked candidate file missing columns: "
            f"{sorted(missing)}"
        )

    #
    # One diagnostic row per frozen validation query.
    #
    rows = []

    ranked_by_query = {
        query: group.copy()
        for query, group in ranked.groupby(
            "query_inchikey14",
            sort=False,
        )
    }

    for manifest_row in manifest.itertuples(
        index=False
    ):
        query = (
            manifest_row.query_inchikey14
        )

        group = ranked_by_query.get(
            query
        )

        #
        # Eight v2 queries have no formula/structure candidates.
        #
        if group is None or group.empty:
            rows.append(
                {
                    "query_inchikey14":
                        query,

                    "truth_rank":
                        np.nan,

                    "rank_bucket":
                        "not_rankable",

                    "num_formula_hypotheses":
                        0,

                    "num_structure_candidates":
                        0,

                    "truth_formula":
                        None,

                    "truth_formula_support_count":
                        np.nan,

                    "truth_formula_support_fraction":
                        np.nan,

                    "truth_formula_mass_error_abs_ppm":
                        np.nan,

                    "winner_formula":
                        None,

                    "winner_score":
                        np.nan,

                    "winner_mist_cosine":
                        np.nan,

                    "winner_same_formula_as_truth":
                        False,

                    "error_type":
                        "no_candidates",
                }
            )

            continue

        #
        # Number of formula hypotheses and unique structures.
        #
        num_formula_hypotheses = (
            group[
                "candidate_formula"
            ].nunique()
        )

        num_structure_candidates = (
            group[
                "candidate_inchikey14"
            ].nunique()
        )

        #
        # Winner = rank 1 after the frozen score.
        #
        winner = (
            group.sort_values(
                "rank"
            ).iloc[0]
        )

        truth_rows = group[
            group[
                "is_truth_structure"
            ]
        ]

        #
        # Structure not present in our candidate universe.
        #
        if truth_rows.empty:
            rows.append(
                {
                    "query_inchikey14":
                        query,

                    "truth_rank":
                        np.nan,

                    "rank_bucket":
                        "not_rankable",

                    "num_formula_hypotheses":
                        num_formula_hypotheses,

                    "num_structure_candidates":
                        num_structure_candidates,

                    "truth_formula":
                        None,

                    "truth_formula_support_count":
                        np.nan,

                    "truth_formula_support_fraction":
                        np.nan,

                    "truth_formula_mass_error_abs_ppm":
                        np.nan,

                    "winner_formula":
                        winner[
                            "candidate_formula"
                        ],

                    "winner_score":
                        float(
                            winner["score"]
                        ),

                    "winner_mist_cosine":
                        float(
                            winner[
                                "mist_cosine"
                            ]
                        ),

                    "winner_same_formula_as_truth":
                        False,

                    "error_type":
                        "truth_not_rankable",
                }
            )

            continue

        if len(truth_rows) != 1:
            raise ValueError(
                f"Expected exactly one truth row for "
                f"{query}, found {len(truth_rows)}"
            )

        truth = truth_rows.iloc[0]

        truth_rank = int(
            truth["rank"]
        )

        truth_formula = (
            truth["candidate_formula"]
        )

        same_formula = (
            winner[
                "candidate_formula"
            ]
            == truth_formula
        )

        if truth_rank == 1:
            error_type = "correct"

        elif same_formula:
            error_type = (
                "same_formula_structure_error"
            )

        else:
            error_type = (
                "cross_formula_intrusion"
            )

        rows.append(
            {
                "query_inchikey14":
                    query,

                "truth_rank":
                    truth_rank,

                "rank_bucket":
                    rank_bucket(
                        truth_rank
                    ),

                "num_formula_hypotheses":
                    num_formula_hypotheses,

                "num_structure_candidates":
                    num_structure_candidates,

                "truth_formula":
                    truth_formula,

                "truth_formula_support_count":
                    int(
                        truth[
                            "formula_support_count"
                        ]
                    ),

                "truth_formula_support_fraction":
                    float(
                        truth[
                            "formula_support_fraction"
                        ]
                    ),

                "truth_formula_mass_error_abs_ppm":
                    abs(
                        float(
                            truth[
                                "selected_mass_error_ppm"
                            ]
                        )
                    ),

                "truth_mist_cosine":
                    float(
                        truth[
                            "mist_cosine"
                        ]
                    ),

                "truth_score":
                    float(
                        truth[
                            "score"
                        ]
                    ),

                "winner_formula":
                    winner[
                        "candidate_formula"
                    ],

                "winner_score":
                    float(
                        winner["score"]
                    ),

                "winner_mist_cosine":
                    float(
                        winner[
                            "mist_cosine"
                        ]
                    ),

                "winner_formula_support_count":
                    int(
                        winner[
                            "formula_support_count"
                        ]
                    ),

                "winner_formula_support_fraction":
                    float(
                        winner[
                            "formula_support_fraction"
                        ]
                    ),

                "winner_mass_error_abs_ppm":
                    abs(
                        float(
                            winner[
                                "selected_mass_error_ppm"
                            ]
                        )
                    ),

                "winner_same_formula_as_truth":
                    bool(
                        same_formula
                    ),

                "error_type":
                    error_type,
            }
        )

    diagnostics = pd.DataFrame(
        rows
    )

    diagnostics.to_parquet(
        OUTPUT_PATH,
        index=False,
    )

    #
    # Overall error decomposition.
    #
    print("\nError decomposition")
    print("-------------------")

    print(
        diagnostics[
            "error_type"
        ]
        .value_counts(
            dropna=False
        )
        .to_string()
    )

    #
    # Rank-bucket counts.
    #
    bucket_order = [
        "rank_1",
        "rank_2_5",
        "rank_6_25",
        "rank_gt_25",
        "not_rankable",
    ]

    print("\nRank buckets")
    print("------------")

    print(
        diagnostics[
            "rank_bucket"
        ]
        .value_counts()
        .reindex(
            bucket_order,
            fill_value=0,
        )
        .to_string()
    )

    #
    # Summary by rank bucket.
    #
    print(
        "\nDiagnostics by truth-rank bucket"
    )
    print(
        "--------------------------------"
    )

    summary = (
        diagnostics.groupby(
            "rank_bucket"
        )
        .agg(
            queries=(
                "query_inchikey14",
                "size",
            ),

            median_formulas=(
                "num_formula_hypotheses",
                "median",
            ),

            mean_formulas=(
                "num_formula_hypotheses",
                "mean",
            ),

            median_structures=(
                "num_structure_candidates",
                "median",
            ),

            mean_structures=(
                "num_structure_candidates",
                "mean",
            ),

            median_truth_support_fraction=(
                "truth_formula_support_fraction",
                "median",
            ),

            mean_truth_support_fraction=(
                "truth_formula_support_fraction",
                "mean",
            ),

            median_truth_support_count=(
                "truth_formula_support_count",
                "median",
            ),

            median_truth_mass_error_ppm=(
                "truth_formula_mass_error_abs_ppm",
                "median",
            ),
        )
        .reindex(
            bucket_order
        )
    )

    print(
        summary.to_string()
    )

    #
    # For incorrect but rankable queries, determine how often the
    # winning candidate came from a different formula.
    #
    incorrect = diagnostics[
        diagnostics[
            "error_type"
        ].isin(
            [
                "same_formula_structure_error",
                "cross_formula_intrusion",
            ]
        )
    ]

    print(
        "\nRankable incorrect queries"
    )
    print(
        "--------------------------"
    )

    print(
        "Total:",
        len(incorrect),
    )

    if len(incorrect):
        error_counts = (
            incorrect[
                "error_type"
            ]
            .value_counts()
        )

        print(
            error_counts.to_string()
        )

        cross_formula = (
            incorrect[
                "error_type"
            ]
            == "cross_formula_intrusion"
        ).sum()

        print(
            "\nCross-formula fraction:",
            f"{cross_formula / len(incorrect):.3%}",
        )

    #
    # Look specifically at cross-formula errors and compare
    # formula evidence between truth and winner.
    #
    cross = diagnostics[
        diagnostics[
            "error_type"
        ]
        == "cross_formula_intrusion"
    ].copy()

    print(
        "\nCross-formula intrusion evidence"
    )
    print(
        "--------------------------------"
    )

    print(
        "Queries:",
        len(cross),
    )

    if len(cross):
        cross[
            "truth_support_advantage"
        ] = (
            cross[
                "truth_formula_support_fraction"
            ]
            - cross[
                "winner_formula_support_fraction"
            ]
        )

        cross[
            "truth_mass_error_advantage"
        ] = (
            cross[
                "winner_mass_error_abs_ppm"
            ]
            - cross[
                "truth_formula_mass_error_abs_ppm"
            ]
        )

        print(
            "\nTruth support fraction > winner:",
            int(
                (
                    cross[
                        "truth_support_advantage"
                    ]
                    > 0
                ).sum()
            ),
            "/",
            len(cross),
        )

        print(
            "Truth support fraction == winner:",
            int(
                np.isclose(
                    cross[
                        "truth_support_advantage"
                    ],
                    0,
                ).sum()
            ),
            "/",
            len(cross),
        )

        print(
            "Truth has better mass error:",
            int(
                (
                    cross[
                        "truth_mass_error_advantage"
                    ]
                    > 0
                ).sum()
            ),
            "/",
            len(cross),
        )

        print(
            "\nTruth support advantage:"
        )

        print(
            cross[
                "truth_support_advantage"
            ]
            .describe(
                percentiles=[
                    0.25,
                    0.5,
                    0.75,
                    0.9,
                ]
            )
            .to_string()
        )

        print(
            "\nTruth-vs-winner mass-error advantage:"
        )

        print(
            cross[
                "truth_mass_error_advantage"
            ]
            .describe(
                percentiles=[
                    0.25,
                    0.5,
                    0.75,
                    0.9,
                ]
            )
            .to_string()
        )

    #
    # Breakdown of error type by rank bucket.
    #
    print(
        "\nError type by rank bucket"
    )
    print(
        "-------------------------"
    )

    table = pd.crosstab(
        diagnostics[
            "rank_bucket"
        ],
        diagnostics[
            "error_type"
        ],
    )

    table = table.reindex(
        bucket_order,
        fill_value=0,
    )

    print(
        table.to_string()
    )

    print(
        f"\nWrote {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()