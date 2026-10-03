from pathlib import Path

import numpy as np
import pandas as pd


FEATURES_PATH = Path(
    "data/processed/results/"
    "retrieval_dev_baseline3_ranking_features.parquet"
)

BASELINE3C_PATH = Path(
    "data/processed/results/"
    "retrieval_dev_baseline3c_frozen.parquet"
)

OUTPUT_PATH = Path(
    "data/processed/results/"
    "retrieval_dev_baseline3c_ranking_misses.parquet"
)


def source_tier(
    row: pd.Series,
) -> int:
    if bool(
        row[
            "in_same_mass"
        ]
    ):
        return 1

    if bool(
        row[
            "in_opposite_mass"
        ]
    ):
        return 2

    return 3


def finite_score(
    value: object,
) -> float | None:
    if pd.isna(
        value
    ):
        return None

    value = float(
        value
    )

    if not np.isfinite(
        value
    ):
        return None

    return value


def summarize_query(
    group: pd.DataFrame,
) -> dict[str, object]:
    truth = group[
        group[
            "is_truth"
        ].astype(bool)
    ]

    if len(truth) == 0:
        return {
            "truth_present": False,
            "truth_tier": None,
            "truth_same_mass_cosine": None,
            "truth_full_cosine": None,
            "candidate_count": len(group),
            "tier1_count": int(
                group[
                    "in_same_mass"
                ]
                .astype(bool)
                .sum()
            ),
            "tier2_count": int(
                (
                    ~group[
                        "in_same_mass"
                    ].astype(bool)
                    & group[
                        "in_opposite_mass"
                    ].astype(bool)
                ).sum()
            ),
            "tier3_count": int(
                (
                    ~group[
                        "in_same_mass"
                    ].astype(bool)
                    & ~group[
                        "in_opposite_mass"
                    ].astype(bool)
                ).sum()
            ),
        }

    if len(truth) != 1:
        raise RuntimeError(
            "Expected at most one truth row."
        )

    truth_row = truth.iloc[0]

    return {
        "truth_present": True,
        "truth_tier": source_tier(
            truth_row
        ),
        "truth_same_mass_cosine": finite_score(
            truth_row[
                "same_mass_cosine"
            ]
        ),
        "truth_full_cosine": finite_score(
            truth_row[
                "same_polarity_cosine"
            ]
        ),
        "candidate_count": len(group),
        "tier1_count": int(
            group[
                "in_same_mass"
            ]
            .astype(bool)
            .sum()
        ),
        "tier2_count": int(
            (
                ~group[
                    "in_same_mass"
                ].astype(bool)
                & group[
                    "in_opposite_mass"
                ].astype(bool)
            ).sum()
        ),
        "tier3_count": int(
            (
                ~group[
                    "in_same_mass"
                ].astype(bool)
                & ~group[
                    "in_opposite_mass"
                ].astype(bool)
            ).sum()
        ),
    }


def print_distribution(
    label: str,
    values: pd.Series,
) -> None:
    values = pd.to_numeric(
        values,
        errors="coerce",
    )

    values = values[
        np.isfinite(
            values
        )
    ]

    if len(values) == 0:
        print(
            f"{label:<26} no finite values"
        )
        return

    print(
        f"{label:<26} "
        f"n={len(values):4} "
        f"median={values.median():8.4f} "
        f"p25={values.quantile(0.25):8.4f} "
        f"p75={values.quantile(0.75):8.4f}"
    )


def main() -> None:
    print(
        "Loading frozen Baseline 3c artifacts..."
    )

    features = pd.read_parquet(
        FEATURES_PATH
    )

    baseline3c = pd.read_parquet(
        BASELINE3C_PATH
    )

    required_feature_columns = {
        "inchikey14",
        "query_library",
        "candidate_id",
        "candidate_inchikey14",
        "is_truth",
        "in_same_mass",
        "in_opposite_mass",
        "in_spectral_top25",
        "spectral_rank",
        "same_polarity_cosine",
        "same_mass_cosine",
        "baseline2_fallback",
    }

    missing = (
        required_feature_columns
        - set(
            features.columns
        )
    )

    if missing:
        raise RuntimeError(
            "Feature artifact is missing "
            f"columns: {sorted(missing)}"
        )

    # ---------------------------------------------------------
    # Build one diagnostic row per query.
    # ---------------------------------------------------------

    summary_rows: list[
        dict[str, object]
    ] = []

    for (
        structure_key,
        query_library,
    ), group in features.groupby(
        [
            "inchikey14",
            "query_library",
        ],
        sort=False,
    ):
        summary = summarize_query(
            group
        )

        fallback_values = (
            group[
                "baseline2_fallback"
            ]
            .astype(bool)
            .unique()
        )

        if len(fallback_values) != 1:
            raise RuntimeError(
                "baseline2_fallback is not "
                "constant within query."
            )

        summary_rows.append(
            {
                "inchikey14": str(
                    structure_key
                ),
                "query_library": str(
                    query_library
                ),
                "baseline2_fallback": bool(
                    fallback_values[
                        0
                    ]
                ),
                **summary,
            }
        )

    summary = pd.DataFrame(
        summary_rows
    )

    if len(summary) != 2_000:
        raise RuntimeError(
            "Expected 2,000 query summaries."
        )

    results = baseline3c[
        [
            "inchikey14",
            "query_library",
            "rank",
            "truth_source_tier",
        ]
    ].copy()

    analysis = results.merge(
        summary,
        on=[
            "inchikey14",
            "query_library",
        ],
        how="left",
        validate="one_to_one",
    )

    if len(analysis) != 2_000:
        raise RuntimeError(
            "Expected 2,000 merged queries."
        )

    # ---------------------------------------------------------
    # Define outcome classes.
    # ---------------------------------------------------------

    analysis[
        "top25"
    ] = (
        analysis[
            "rank"
        ]
        .le(25)
        .fillna(False)
    )

    analysis[
        "ranking_miss"
    ] = (
        analysis[
            "truth_present"
        ]
        & ~analysis[
            "top25"
        ]
    )

    analysis[
        "candidate_miss"
    ] = (
        ~analysis[
            "truth_present"
        ]
    )

    ranking_misses = analysis[
        analysis[
            "ranking_miss"
        ]
    ].copy()

    candidate_misses = analysis[
        analysis[
            "candidate_miss"
        ]
    ].copy()

    successes = analysis[
        analysis[
            "top25"
        ]
    ].copy()

    # ---------------------------------------------------------
    # Basic invariants.
    # ---------------------------------------------------------

    print()
    print(
        "Baseline 3c ranking-error analysis"
    )
    print(
        "----------------------------------"
    )

    print(
        f"Queries:                 "
        f"{len(analysis):,}"
    )

    print(
        f"Top-25 successes:        "
        f"{len(successes):,}"
    )

    print(
        f"Candidate misses:        "
        f"{len(candidate_misses):,}"
    )

    print(
        f"Ranking misses:          "
        f"{len(ranking_misses):,}"
    )

    expected_ranking_misses = 32

    if len(
        ranking_misses
    ) != expected_ranking_misses:
        raise RuntimeError(
            "Expected "
            f"{expected_ranking_misses} "
            "truth-present Top-25 misses, "
            f"found {len(ranking_misses)}."
        )

    if len(
        candidate_misses
    ) != 8:
        raise RuntimeError(
            "Expected 8 candidate-generation misses, "
            f"found {len(candidate_misses)}."
        )

    # ---------------------------------------------------------
    # Ranking misses by truth tier.
    # ---------------------------------------------------------

    print()
    print(
        "Ranking misses by truth tier"
    )
    print(
        "----------------------------"
    )

    tier_counts = (
        ranking_misses[
            "truth_tier"
        ]
        .value_counts(
            dropna=False
        )
        .sort_index()
    )

    print(
        tier_counts.to_string()
    )

    # ---------------------------------------------------------
    # Ranking misses by query library.
    # ---------------------------------------------------------

    print()
    print(
        "Ranking misses by query library"
    )
    print(
        "-------------------------------"
    )

    library_counts = (
        ranking_misses[
            "query_library"
        ]
        .value_counts()
        .sort_values(
            ascending=False
        )
    )

    print(
        library_counts.to_string()
    )

    # ---------------------------------------------------------
    # Fallback contribution.
    # ---------------------------------------------------------

    print()
    print(
        "Fallback status"
    )
    print(
        "---------------"
    )

    fallback_counts = (
        ranking_misses[
            "baseline2_fallback"
        ]
        .value_counts()
        .sort_index()
    )

    print(
        fallback_counts.to_string()
    )

    # ---------------------------------------------------------
    # Candidate-set sizes.
    # ---------------------------------------------------------

    print()
    print(
        "Candidate-set size"
    )
    print(
        "------------------"
    )

    print_distribution(
        "Top-25 successes",
        successes[
            "candidate_count"
        ],
    )

    print_distribution(
        "Ranking misses",
        ranking_misses[
            "candidate_count"
        ],
    )

    print()
    print(
        "Tier-1 candidate count"
    )
    print(
        "----------------------"
    )

    print_distribution(
        "Top-25 successes",
        successes[
            "tier1_count"
        ],
    )

    print_distribution(
        "Ranking misses",
        ranking_misses[
            "tier1_count"
        ],
    )

    # ---------------------------------------------------------
    # Truth cosine distributions.
    # ---------------------------------------------------------

    print()
    print(
        "Truth same-mass cosine"
    )
    print(
        "----------------------"
    )

    print_distribution(
        "Top-25 successes",
        successes[
            "truth_same_mass_cosine"
        ],
    )

    print_distribution(
        "Ranking misses",
        ranking_misses[
            "truth_same_mass_cosine"
        ],
    )

    print()
    print(
        "Truth full-library cosine"
    )
    print(
        "-------------------------"
    )

    print_distribution(
        "Top-25 successes",
        successes[
            "truth_full_cosine"
        ],
    )

    print_distribution(
        "Ranking misses",
        ranking_misses[
            "truth_full_cosine"
        ],
    )

    # ---------------------------------------------------------
    # Detailed miss table.
    # ---------------------------------------------------------

    detail_columns = [
        "inchikey14",
        "query_library",
        "rank",
        "truth_tier",
        "baseline2_fallback",
        "candidate_count",
        "tier1_count",
        "tier2_count",
        "tier3_count",
        "truth_same_mass_cosine",
        "truth_full_cosine",
    ]

    print()
    print(
        "Truth-present Top-25 misses"
    )
    print(
        "---------------------------"
    )

    print(
        ranking_misses[
            detail_columns
        ]
        .sort_values(
            [
                "truth_tier",
                "rank",
                "query_library",
            ],
            na_position="last",
        )
        .to_string(
            index=False
        )
    )

    # ---------------------------------------------------------
    # Save the 32-query diagnostic artifact.
    # ---------------------------------------------------------

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    ranking_misses[
        detail_columns
    ].to_parquet(
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