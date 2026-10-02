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


def source_tier(
    group: pd.DataFrame,
) -> np.ndarray:
    """
    Assign each candidate to its highest-priority source.

    Tier 1:
        Same-polarity neutral-mass candidate.

    Tier 2:
        Opposite-polarity neutral-mass candidate that is not
        already in Tier 1.

    Tier 3:
        Spectral Top-25 candidate that is not already in either
        mass tier.
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
    Return the highest-priority source containing the truth.

    Returns None when the truth is absent from the hybrid
    candidate set.
    """
    truth = group[
        group["is_truth"]
    ]

    if len(truth) == 0:
        return None

    if len(truth) != 1:
        raise RuntimeError(
            "Expected exactly one truth candidate."
        )

    row = truth.iloc[0]

    if bool(
        row["in_same_mass"]
    ):
        return 1

    if bool(
        row["in_opposite_mass"]
    ):
        return 2

    return 3


def rank_query(
    group: pd.DataFrame,
) -> int | None:
    """
    Rank the hybrid candidate set using source priority.

    Priority:
        1. same-polarity mass
        2. opposite-polarity mass
        3. spectral-only

    Within each tier:
        same-polarity cosine descending
        candidate ID ascending
    """
    truth = group[
        group["is_truth"]
    ]

    if len(truth) == 0:
        return None

    if len(truth) != 1:
        raise RuntimeError(
            "Expected exactly one truth candidate."
        )

    ranked = group.copy()

    ranked[
        "_source_tier"
    ] = source_tier(
        ranked
    )

    ranked = ranked.sort_values(
        by=[
            "_source_tier",
            "same_polarity_cosine",
            "candidate_id",
        ],
        ascending=[
            True,
            False,
            True,
        ],
        kind="stable",
    )

    truth_candidate_id = int(
        truth.iloc[0][
            "candidate_id"
        ]
    )

    ranked_ids = ranked[
        "candidate_id"
    ].to_numpy(
        dtype=np.int64
    )

    matches = np.flatnonzero(
        ranked_ids
        == truth_candidate_id
    )

    if len(matches) != 1:
        raise RuntimeError(
            "Truth candidate was lost "
            "during ranking."
        )

    return int(
        matches[0] + 1
    )


def reciprocal_rank_at_25(
    rank: int | float | None,
) -> float:
    """
    Return reciprocal rank when rank <= 25,
    otherwise zero.
    """
    if rank is None:
        return 0.0

    if pd.isna(rank):
        return 0.0

    if rank > 25:
        return 0.0

    return 1.0 / rank


def calculate_metrics(
    results: pd.DataFrame,
) -> tuple[
    float,
    float,
    float,
    float,
]:
    """
    Calculate retrieval metrics from a DataFrame containing
    a 'rank' column.
    """
    ranks = results[
        "rank"
    ]

    mrr = (
        ranks
        .map(
            reciprocal_rank_at_25
        )
        .mean()
    )

    top1 = (
        ranks == 1
    ).mean()

    top25 = (
        ranks.notna()
        & (
            ranks <= 25
        )
    ).mean()

    candidate_recall = (
        ranks.notna()
    ).mean()

    return (
        mrr,
        top1,
        top25,
        candidate_recall,
    )


def main() -> None:
    features = pd.read_parquet(
        FEATURES_PATH
    )

    baseline2 = pd.read_parquet(
        BASELINE2_PATH
    )

    rows = []

    grouped = features.groupby(
        [
            "inchikey14",
            "query_library",
        ],
        sort=False,
    )

    for (
        inchikey14,
        query_library,
    ), group in grouped:
        rank = rank_query(
            group
        )

        rows.append(
            {
                "inchikey14": (
                    inchikey14
                ),
                "query_library": (
                    query_library
                ),
                "rank": rank,
                "truth_source_tier": (
                    truth_source_tier(
                        group
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
            "retrieval-dev queries."
        )

    (
        mrr,
        top1,
        top25,
        candidate_recall,
    ) = calculate_metrics(
        results
    )

    # ---------------------------------------------------------
    # Baseline 2.
    # ---------------------------------------------------------

    baseline2_results = pd.DataFrame(
        {
            "rank": (
                baseline2[
                    "rank"
                ]
            )
        }
    )

    (
        baseline2_mrr,
        baseline2_top1,
        baseline2_top25,
        _,
    ) = calculate_metrics(
        baseline2_results
    )

    # ---------------------------------------------------------
    # Main report.
    # ---------------------------------------------------------

    print(
        "Retrieval Baseline 3b"
    )
    print(
        "---------------------"
    )

    print(
        "Hybrid candidates ranked by "
        "source priority + cosine"
    )

    print()

    print(
        "Priority:"
    )

    print(
        "  1. same-polarity mass"
    )

    print(
        "  2. opposite-polarity mass"
    )

    print(
        "  3. spectral-only"
    )

    print()

    print(
        f"Structures:       "
        f"{len(results):,}"
    )

    print(
        f"Candidate recall: "
        f"{candidate_recall:.2%}"
    )

    print(
        f"MRR@25:           "
        f"{mrr:.4f}"
    )

    print(
        f"Top-1:            "
        f"{top1:.2%}"
    )

    print(
        f"Top-25:           "
        f"{top25:.2%}"
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
        f"{mrr - baseline2_mrr:+.4f}"
    )

    print(
        f"Top-1:            "
        f"{(top1 - baseline2_top1) * 100:+.2f} pp"
    )

    print(
        f"Top-25:           "
        f"{(top25 - baseline2_top25) * 100:+.2f} pp"
    )

    # ---------------------------------------------------------
    # Truth source tiers.
    # ---------------------------------------------------------

    truth_rows = features[
        features[
            "is_truth"
        ]
    ].copy()

    truth_rows[
        "source_tier"
    ] = source_tier(
        truth_rows
    )

    print()
    print(
        "Truth candidate source tier"
    )

    tier_counts = (
        truth_rows[
            "source_tier"
        ]
        .value_counts()
        .sort_index()
    )

    for tier in (
        1,
        2,
        3,
    ):
        count = int(
            tier_counts.get(
                tier,
                0,
            )
        )

        print(
            f"  Tier {tier}: "
            f"{count:,}"
        )

    # ---------------------------------------------------------
    # Query-by-query comparison with Baseline 2.
    # ---------------------------------------------------------

    comparison = results.merge(
        baseline2[
            [
                "inchikey14",
                "query_library",
                "rank",
            ]
        ].rename(
            columns={
                "rank": (
                    "baseline2_rank"
                ),
            }
        ),
        on=[
            "inchikey14",
            "query_library",
        ],
        how="left",
        validate="one_to_one",
    )

    if len(comparison) != 2_000:
        raise RuntimeError(
            "Baseline 2 comparison did not "
            "produce exactly 2,000 rows."
        )

    comparison[
        "baseline2_top25"
    ] = (
        comparison[
            "baseline2_rank"
        ].notna()
        & (
            comparison[
                "baseline2_rank"
            ] <= 25
        )
    )

    comparison[
        "baseline3b_top25"
    ] = (
        comparison[
            "rank"
        ].notna()
        & (
            comparison[
                "rank"
            ] <= 25
        )
    )

    regressions = comparison[
        comparison[
            "baseline2_top25"
        ]
        & ~comparison[
            "baseline3b_top25"
        ]
    ].copy()

    recoveries = comparison[
        ~comparison[
            "baseline2_top25"
        ]
        & comparison[
            "baseline3b_top25"
        ]
    ].copy()

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
        for baseline3b_hit in (
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
                            "baseline3b_top25"
                        ]
                        == baseline3b_hit
                    )
                ).sum()
            )

            print(
                f"B2={str(baseline2_hit):5} "
                f"3b={str(baseline3b_hit):5}: "
                f"{count:4}"
            )

    print()
    print(
        f"Regressions: "
        f"{len(regressions)}"
    )

    print(
        f"Recoveries:  "
        f"{len(recoveries)}"
    )

    # ---------------------------------------------------------
    # Source-tier breakdown of transitions.
    # ---------------------------------------------------------

    print()
    print(
        "Regression truth tiers"
    )

    if len(regressions) == 0:
        print(
            "  none"
        )
    else:
        regression_tiers = (
            regressions[
                "truth_source_tier"
            ]
            .value_counts(
                dropna=False
            )
            .sort_index()
        )

        print(
            regression_tiers.to_string()
        )

    print()
    print(
        "Recovery truth tiers"
    )

    if len(recoveries) == 0:
        print(
            "  none"
        )
    else:
        recovery_tiers = (
            recoveries[
                "truth_source_tier"
            ]
            .value_counts(
                dropna=False
            )
            .sort_index()
        )

        print(
            recovery_tiers.to_string()
        )

    # ---------------------------------------------------------
    # Detailed transition rows.
    # ---------------------------------------------------------

    print()
    print(
        "Regressions"
    )
    print(
        "-----------"
    )

    if len(regressions) == 0:
        print(
            "none"
        )
    else:
        print(
            regressions[
                [
                    "inchikey14",
                    "query_library",
                    "baseline2_rank",
                    "rank",
                    "truth_source_tier",
                ]
            ]
            .sort_values(
                by=[
                    "truth_source_tier",
                    "baseline2_rank",
                    "rank",
                    "inchikey14",
                ],
                na_position="last",
            )
            .to_string(
                index=False
            )
        )

    print()
    print(
        "Recoveries"
    )
    print(
        "----------"
    )

    if len(recoveries) == 0:
        print(
            "none"
        )
    else:
        print(
            recoveries[
                [
                    "inchikey14",
                    "query_library",
                    "baseline2_rank",
                    "rank",
                    "truth_source_tier",
                ]
            ]
            .sort_values(
                by=[
                    "truth_source_tier",
                    "rank",
                    "baseline2_rank",
                    "inchikey14",
                ],
                na_position="last",
            )
            .to_string(
                index=False
            )
        )


if __name__ == "__main__":
    main()