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

BASELINE3B_PATH = Path(
    "data/processed/results/"
    "retrieval_dev_baseline3b.parquet"
)


def source_tier(
    group: pd.DataFrame,
) -> np.ndarray:
    """
    Assign the frozen Baseline 3 candidate-source hierarchy:

        1. same-polarity mass candidate
        2. opposite-polarity mass candidate
        3. spectral-only candidate

    Membership in an earlier source always wins.
    """
    in_same = (
        group[
            "in_same_mass"
        ].to_numpy(
            dtype=bool
        )
    )

    in_opposite = (
        group[
            "in_opposite_mass"
        ].to_numpy(
            dtype=bool
        )
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

    None means the truth is absent from the hybrid candidate set.
    """
    truth = group[
        group[
            "is_truth"
        ]
    ]

    if len(truth) == 0:
        return None

    if len(truth) != 1:
        raise RuntimeError(
            "Expected exactly one truth candidate row."
        )

    if bool(
        truth.iloc[0][
            "in_same_mass"
        ]
    ):
        return 1

    if bool(
        truth.iloc[0][
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

    Normal mass-filtered query:

        Tier 1:
            same-polarity mass candidates,
            ordered by Baseline 2's mass-filtered cosine.

        Tier 2:
            opposite-polarity mass candidates not already in Tier 1,
            ordered by unrestricted same-polarity cosine.

        Tier 3:
            spectral-only candidates,
            ordered by unrestricted same-polarity cosine.

    Baseline 2 fallback query:

        Preserve the fallback semantics and rank the entire hybrid
        candidate set by unrestricted same-polarity cosine.

    Candidate ID is the deterministic final tie-breaker.
    """
    truth = group[
        group[
            "is_truth"
        ]
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
        # Baseline 2 fallback used unrestricted same-polarity
        # retrieval. Preserve that ordering rather than imposing
        # the A/B/C source hierarchy.
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

        # Tier 1 uses the exact Baseline 2 mass-filtered score.
        #
        # Tiers 2 and 3 are rescue candidates and therefore use
        # unrestricted same-polarity cosine.
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
            ).fillna(
                False
            ).mean()
        ),
        "candidate_recall": float(
            ranks.notna().mean()
        ),
    }


def format_rank(
    value: float | int | None,
) -> str:
    if value is None:
        return "-"

    if pd.isna(
        value
    ):
        return "-"

    return str(
        int(value)
    )


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

    # ---------------------------------------------------------
    # Rank every query.
    # ---------------------------------------------------------

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

        if len(
            fallback_values
        ) != 1:
            raise RuntimeError(
                "baseline2_fallback is not "
                "constant within a query."
            )

        rows.append(
            {
                "inchikey14": (
                    str(
                        structure_key
                    )
                ),
                "query_library": (
                    str(
                        query_library
                    )
                ),
                "rank": (
                    rank_query(
                        group
                    )
                ),
                "truth_source_tier": (
                    truth_source_tier(
                        group
                    )
                ),
                "baseline2_fallback": (
                    bool(
                        fallback_values[
                            0
                        ]
                    )
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

    # ---------------------------------------------------------
    # Baseline 3c metrics.
    # ---------------------------------------------------------

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
                "rank": (
                    "baseline2_rank"
                )
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

    if comparison[
        "baseline2_rank"
    ].isna().all():
        raise RuntimeError(
            "Baseline 2 comparison failed."
        )

    baseline2_results = (
        comparison[
            [
                "inchikey14",
                "query_library",
                "baseline2_rank",
            ]
        ]
        .rename(
            columns={
                "baseline2_rank": "rank"
            }
        )
    )

    baseline2_metrics = (
        calculate_metrics(
            baseline2_results
        )
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
    # Rank equality diagnostics.
    #
    # For non-fallback Tier-1 truths, Baseline 3c is intended to
    # use exactly the same score signal as Baseline 2.
    #
    # Differences here should therefore primarily expose the
    # distinction between Baseline 2's optimistic tie-rank:
    #
    #     1 + count(score > truth_score)
    #
    # and the deterministic ordered-list ranking used by 3c.
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

    # ---------------------------------------------------------
    # Fallback diagnostics.
    # ---------------------------------------------------------

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
        f"{baseline2_metrics['mrr']:.4f}"
    )

    print(
        f"Top-1:            "
        f"{baseline2_metrics['top1']:.2%}"
    )

    print(
        f"Top-25:           "
        f"{baseline2_metrics['top25']:.2%}"
    )

    print()
    print(
        "Delta vs Baseline 2"
    )

    print(
        f"MRR@25:           "
        f"{metrics['mrr'] - baseline2_metrics['mrr']:+.4f}"
    )

    print(
        f"Top-1:            "
        f"{100 * (metrics['top1'] - baseline2_metrics['top1']):+.2f} pp"
    )

    print(
        f"Top-25:           "
        f"{100 * (metrics['top25'] - baseline2_metrics['top25']):+.2f} pp"
    )

    # ---------------------------------------------------------
    # Truth tiers.
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
        ].isna().sum()
    )

    print(
        f"  Absent: "
        f"{absent_truth:,}"
    )

    # ---------------------------------------------------------
    # Reproduction diagnostics.
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
    # Transition matrix.
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
    # Regression/recovery source tiers.
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

    # ---------------------------------------------------------
    # Detailed transitions.
    # ---------------------------------------------------------

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
    # Tier-1 rank mismatches.
    #
    # These are especially interesting because the score signal
    # should now be identical to Baseline 2. Any remaining
    # difference should be investigated as an ordering/tie issue.
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
            .head(
                100
            )
            .to_string(
                index=False
            )
        )

    # ---------------------------------------------------------
    # Save the result so later experiments can compare against
    # this exact Baseline 3c run.
    # ---------------------------------------------------------

    output_path = Path(
        "data/processed/results/"
        "retrieval_dev_baseline3c.parquet"
    )

    results.to_parquet(
        output_path,
        index=False,
    )

    print()
    print(
        "Saved:"
    )
    print(
        f"  {output_path}"
    )


if __name__ == "__main__":
    main()