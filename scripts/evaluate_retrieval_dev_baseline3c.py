from pathlib import Path

import numpy as np
import pandas as pd


FEATURES_PATH = Path(
    "data/processed/results/"
    "retrieval_dev_baseline3_ranking_features.parquet"
)

BASELINE2_PATH = Path(
    "data/processed/results/"
    "retrieval_dev_baseline2_frozen.parquet"
)

OUTPUT_PATH = Path(
    "data/processed/results/"
    "retrieval_dev_baseline3c.parquet"
)


def source_tier(
    group: pd.DataFrame,
) -> np.ndarray:
    """
    Assign the Baseline 3 candidate-source hierarchy.

    Tier 1:
        Same-polarity neutral-mass candidates.

    Tier 2:
        Opposite-polarity neutral-mass candidates that are not
        already Tier 1.

    Tier 3:
        Spectral Top-25 candidates that are not already Tier 1
        or Tier 2.
    """
    in_same = group[
        "in_same_mass"
    ].to_numpy(
        dtype=bool
    )

    in_opposite = group[
        "in_opposite_mass"
    ].to_numpy(
        dtype=bool
    )

    tiers = np.full(
        len(group),
        3,
        dtype=np.int8,
    )

    tiers[
        in_opposite
    ] = 2

    tiers[
        in_same
    ] = 1

    return tiers


def truth_source_tier(
    group: pd.DataFrame,
) -> int | None:
    """
    Return the source tier containing the truth candidate.

    None means the truth is absent from the frozen A+B+C
    candidate set.
    """
    truth = group[
        group[
            "is_truth"
        ].astype(bool)
    ]

    if len(truth) == 0:
        return None

    if len(truth) != 1:
        raise RuntimeError(
            "Expected exactly one truth candidate row."
        )

    truth_row = truth.iloc[0]

    if bool(
        truth_row[
            "in_same_mass"
        ]
    ):
        return 1

    if bool(
        truth_row[
            "in_opposite_mass"
        ]
    ):
        return 2

    return 3


def rank_query(
    group: pd.DataFrame,
) -> int | None:
    """
    Rank one retrieval-dev query using Baseline 3c.

    Normal query:

        Tier 1:
            Same-polarity mass candidates ranked by
            same_mass_cosine.

        Tier 2:
            Opposite-polarity mass candidates not already
            in Tier 1, ranked by same_polarity_cosine.

        Tier 3:
            Spectral-only candidates ranked by
            same_polarity_cosine.

    Fallback query:

        Rank the entire frozen hybrid candidate set by
        same_polarity_cosine.

    Important:
        We intentionally retain candidates whose cosine score is
        -inf.

        In particular, an opposite-polarity mass candidate may
        have useful mass evidence despite having no compatible
        same-polarity reference spectrum. Removing such candidates
        was experimentally shown to reduce hybrid Top-25 recovery.

        candidate_id provides deterministic ordering for tied
        scores, including -inf.
    """
    truth = group[
        group[
            "is_truth"
        ].astype(bool)
    ]

    if len(truth) == 0:
        return None

    if len(truth) != 1:
        raise RuntimeError(
            "Expected exactly one truth candidate row."
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
            "baseline2_fallback is not constant "
            "within a query."
        )

    used_fallback = bool(
        fallback_values[0]
    )

    ranked = group.copy()

    if used_fallback:
        # Preserve the original Baseline 3c fallback behavior:
        # rank the entire hybrid candidate set, including candidates
        # without a finite same-polarity cosine score.
        ranked = ranked.sort_values(
            by=[
                "same_polarity_cosine",
                "candidate_id",
            ],
            ascending=[
                False,
                True,
            ],
            kind="stable",
        )

    else:
        ranked[
            "_source_tier"
        ] = source_tier(
            ranked
        )

        # Tier 1 must use the exact Baseline 2 mass-compatible
        # scoring signal.
        #
        # Rescue tiers use the unrestricted same-polarity cosine.
        ranked[
            "_ranking_score"
        ] = np.where(
            ranked[
                "_source_tier"
            ].to_numpy()
            == 1,
            ranked[
                "same_mass_cosine"
            ].to_numpy(
                dtype=np.float64
            ),
            ranked[
                "same_polarity_cosine"
            ].to_numpy(
                dtype=np.float64
            ),
        )

        ranked = ranked.sort_values(
            by=[
                "_source_tier",
                "_ranking_score",
                "candidate_id",
            ],
            ascending=[
                True,
                False,
                True,
            ],
            kind="stable",
        )

    truth_positions = np.flatnonzero(
        ranked[
            "is_truth"
        ].to_numpy(
            dtype=bool
        )
    )

    if len(truth_positions) != 1:
        raise RuntimeError(
            "Expected exactly one ranked truth candidate."
        )

    return int(
        truth_positions[0]
    ) + 1


def reciprocal_rank_at_25(
    rank: int | float | None,
) -> float:
    if rank is None:
        return 0.0

    if pd.isna(
        rank
    ):
        return 0.0

    if rank > 25:
        return 0.0

    return 1.0 / rank


def calculate_metrics(
    results: pd.DataFrame,
) -> dict[str, float]:
    ranks = results[
        "rank"
    ]

    reciprocal_ranks = ranks.map(
        reciprocal_rank_at_25
    )

    return {
        "mrr": float(
            reciprocal_ranks.mean()
        ),
        "top1": float(
            (
                ranks == 1
            ).mean()
        ),
        "top25": float(
            (
                ranks <= 25
            )
            .fillna(False)
            .mean()
        ),
        "candidate_recall": float(
            results[
                "truth_source_tier"
            ]
            .notna()
            .mean()
        ),
    }


def main() -> None:
    print(
        "Loading ranking features..."
    )

    features = pd.read_parquet(
        FEATURES_PATH
    )

    baseline2 = pd.read_parquet(
        BASELINE2_PATH
    )

    required_columns = {
        "inchikey14",
        "query_library",
        "candidate_id",
        "is_truth",
        "in_same_mass",
        "in_opposite_mass",
        "in_spectral_top25",
        "same_polarity_cosine",
        "same_mass_cosine",
        "baseline2_fallback",
    }

    missing_columns = (
        required_columns
        - set(
            features.columns
        )
    )

    if missing_columns:
        raise RuntimeError(
            "Ranking feature artifact is missing "
            f"columns: {sorted(missing_columns)}"
        )

    rows: list[
        dict[str, object]
    ] = []

    grouped = features.groupby(
        [
            "inchikey14",
            "query_library",
        ],
        sort=False,
    )

    for (
        structure_key,
        query_library,
    ), group in grouped:
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
                "constant within a query."
            )

        rows.append(
            {
                "inchikey14": str(
                    structure_key
                ),
                "query_library": str(
                    query_library
                ),
                "rank": rank_query(
                    group
                ),
                "truth_source_tier": (
                    truth_source_tier(
                        group
                    )
                ),
                "baseline2_fallback": bool(
                    fallback_values[
                        0
                    ]
                ),
            }
        )

    results = pd.DataFrame(
        rows
    )

    if len(results) != 2_000:
        raise RuntimeError(
            "Expected exactly 2,000 "
            "retrieval-dev results."
        )

    metrics = calculate_metrics(
        results
    )

    # ---------------------------------------------------------
    # Baseline 2 comparison.
    # ---------------------------------------------------------

    baseline2_compare = (
        baseline2[
            [
                "inchikey14",
                "query_library",
                "rank",
            ]
        ]
        .rename(
            columns={
                "rank": "baseline2_rank",
            }
        )
    )

    comparison = results.merge(
        baseline2_compare,
        on=[
            "inchikey14",
            "query_library",
        ],
        how="left",
        validate="one_to_one",
    )

    baseline2_ranks = comparison[
        "baseline2_rank"
    ]

    baseline2_mrr = float(
        baseline2_ranks.map(
            reciprocal_rank_at_25
        ).mean()
    )

    baseline2_top1 = float(
        (
            baseline2_ranks == 1
        ).mean()
    )

    baseline2_top25 = float(
        (
            baseline2_ranks <= 25
        )
        .fillna(False)
        .mean()
    )

    # ---------------------------------------------------------
    # Top-25 transitions.
    # ---------------------------------------------------------

    comparison[
        "baseline2_top25"
    ] = (
        comparison[
            "baseline2_rank"
        ]
        .le(25)
        .fillna(False)
    )

    comparison[
        "baseline3c_top25"
    ] = (
        comparison[
            "rank"
        ]
        .le(25)
        .fillna(False)
    )

    regressions = comparison[
        comparison[
            "baseline2_top25"
        ]
        & ~comparison[
            "baseline3c_top25"
        ]
    ].copy()

    recoveries = comparison[
        ~comparison[
            "baseline2_top25"
        ]
        & comparison[
            "baseline3c_top25"
        ]
    ].copy()

    # ---------------------------------------------------------
    # Baseline 2 preservation diagnostics.
    # ---------------------------------------------------------

    tier1_nonfallback = comparison[
        (
            comparison[
                "truth_source_tier"
            ]
            == 1
        )
        & ~comparison[
            "baseline2_fallback"
        ]
    ].copy()

    tier1_nonfallback[
        "rank_matches_baseline2"
    ] = (
        tier1_nonfallback[
            "rank"
        ].fillna(-1)
        == tier1_nonfallback[
            "baseline2_rank"
        ].fillna(-1)
    )

    exact_tier1_rank_matches = int(
        tier1_nonfallback[
            "rank_matches_baseline2"
        ].sum()
    )

    fallback_results = comparison[
        comparison[
            "baseline2_fallback"
        ]
    ].copy()

    fallback_results[
        "rank_matches_baseline2"
    ] = (
        fallback_results[
            "rank"
        ].fillna(-1)
        == fallback_results[
            "baseline2_rank"
        ].fillna(-1)
    )

    exact_fallback_rank_matches = int(
        fallback_results[
            "rank_matches_baseline2"
        ].sum()
    )

    # ---------------------------------------------------------
    # Report.
    # ---------------------------------------------------------

    print()
    print(
        "Retrieval Baseline 3c"
    )
    print(
        "---------------------"
    )

    print(
        "Mass-compatible ranking + "
        "hybrid rescue candidates"
    )

    print()
    print(
        "Normal query:"
    )
    print(
        "  1. same-polarity mass "
        "-> same-mass cosine"
    )
    print(
        "  2. opposite-polarity mass "
        "-> full same-polarity cosine"
    )
    print(
        "  3. spectral-only "
        "-> full same-polarity cosine"
    )

    print()
    print(
        "Fallback query:"
    )
    print(
        "  full same-polarity cosine"
    )

    print()
    print(
        f"Structures:       "
        f"{len(results):,}"
    )

    print(
        f"Candidate recall: "
        f"{metrics['candidate_recall']:.2%}"
    )

    print(
        f"MRR@25:           "
        f"{metrics['mrr']:.4f}"
    )

    print(
        f"Top-1:            "
        f"{metrics['top1']:.2%}"
    )

    print(
        f"Top-25:           "
        f"{metrics['top25']:.2%}"
    )

    print()
    print(
        "Baseline 2"
    )

    print(
        f"MRR@25:           "
        f"{baseline2_mrr:.4f}"
    )

    print(
        f"Top-1:            "
        f"{baseline2_top1:.2%}"
    )

    print(
        f"Top-25:           "
        f"{baseline2_top25:.2%}"
    )

    print()
    print(
        "Delta vs Baseline 2"
    )

    print(
        f"MRR@25:           "
        f"{metrics['mrr'] - baseline2_mrr:+.4f}"
    )

    print(
        f"Top-1:            "
        f"{100 * (metrics['top1'] - baseline2_top1):+.2f} pp"
    )

    print(
        f"Top-25:           "
        f"{100 * (metrics['top25'] - baseline2_top25):+.2f} pp"
    )

    # ---------------------------------------------------------
    # Truth source tiers.
    # ---------------------------------------------------------

    print()
    print(
        "Truth candidate source tier"
    )

    tier_counts = (
        results[
            "truth_source_tier"
        ]
        .value_counts()
        .sort_index()
    )

    for tier in (
        1,
        2,
        3,
    ):
        print(
            f"  Tier {tier}: "
            f"{int(tier_counts.get(tier, 0)):,}"
        )

    absent_truth = int(
        results[
            "truth_source_tier"
        ]
        .isna()
        .sum()
    )

    print(
        f"  Absent: "
        f"{absent_truth:,}"
    )

    # ---------------------------------------------------------
    # Preservation report.
    # ---------------------------------------------------------

    print()
    print(
        "Baseline 2 preservation"
    )
    print(
        "-----------------------"
    )

    print(
        "Non-fallback Tier-1 truths:"
        f" {len(tier1_nonfallback):,}"
    )

    print(
        "Exact rank matches:          "
        f"{exact_tier1_rank_matches:,}"
        f" / {len(tier1_nonfallback):,}"
    )

    print()
    print(
        "Fallback queries:            "
        f"{len(fallback_results):,}"
    )

    print(
        "Exact fallback rank matches: "
        f"{exact_fallback_rank_matches:,}"
        f" / {len(fallback_results):,}"
    )

    # ---------------------------------------------------------
    # Top-25 transition matrix.
    # ---------------------------------------------------------

    print()
    print(
        "Top-25 transitions"
    )
    print(
        "------------------"
    )

    for baseline2_hit in (
        True,
        False,
    ):
        for baseline3c_hit in (
            True,
            False,
        ):
            count = int(
                (
                    (
                        comparison[
                            "baseline2_top25"
                        ]
                        == baseline2_hit
                    )
                    & (
                        comparison[
                            "baseline3c_top25"
                        ]
                        == baseline3c_hit
                    )
                ).sum()
            )

            print(
                f"B2={str(baseline2_hit):5} "
                f"3c={str(baseline3c_hit):5}: "
                f"{count:4}"
            )

    print()
    print(
        f"Regressions: "
        f"{len(regressions):,}"
    )

    print(
        f"Recoveries:  "
        f"{len(recoveries):,}"
    )

    # ---------------------------------------------------------
    # Recovery/regression source tiers.
    # ---------------------------------------------------------

    if len(regressions):
        print()
        print(
            "Regression truth tiers"
        )

        print(
            regressions[
                "truth_source_tier"
            ]
            .value_counts(
                dropna=False
            )
            .sort_index()
            .to_string()
        )

    if len(recoveries):
        print()
        print(
            "Recovery truth tiers"
        )

        print(
            recoveries[
                "truth_source_tier"
            ]
            .value_counts(
                dropna=False
            )
            .sort_index()
            .to_string()
        )

    detail_columns = [
        "inchikey14",
        "query_library",
        "baseline2_rank",
        "rank",
        "truth_source_tier",
        "baseline2_fallback",
    ]

    if len(regressions):
        print()
        print(
            "Regressions"
        )
        print(
            "-----------"
        )

        print(
            regressions[
                detail_columns
            ]
            .sort_values(
                [
                    "query_library",
                    "baseline2_rank",
                    "rank",
                ]
            )
            .to_string(
                index=False
            )
        )

    if len(recoveries):
        print()
        print(
            "Recoveries"
        )
        print(
            "----------"
        )

        print(
            recoveries[
                detail_columns
            ]
            .sort_values(
                [
                    "truth_source_tier",
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
    # Tier-1 mismatches.
    # ---------------------------------------------------------

    tier1_mismatches = tier1_nonfallback[
        ~tier1_nonfallback[
            "rank_matches_baseline2"
        ]
    ].copy()

    print()
    print(
        "Tier-1 non-fallback rank mismatches"
    )
    print(
        "-----------------------------------"
    )

    print(
        f"Count: "
        f"{len(tier1_mismatches):,}"
    )

    if len(
        tier1_mismatches
    ):
        print()

        print(
            tier1_mismatches[
                [
                    "inchikey14",
                    "query_library",
                    "baseline2_rank",
                    "rank",
                ]
            ]
            .sort_values(
                [
                    "baseline2_rank",
                    "rank",
                ]
            )
            .to_string(
                index=False
            )
        )

    # ---------------------------------------------------------
    # Save only after the evaluation has completed.
    # ---------------------------------------------------------

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results.to_parquet(
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