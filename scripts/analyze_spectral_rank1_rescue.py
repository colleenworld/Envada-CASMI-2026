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
    "retrieval_dev_spectral_rank1_rescue.parquet"
)


def source_tier(
    row: pd.Series,
) -> int:
    if bool(
        row["in_same_mass"]
    ):
        return 1

    if bool(
        row["in_opposite_mass"]
    ):
        return 2

    return 3


def optional_int(
    value: object,
) -> int | None:
    if pd.isna(value):
        return None

    return int(value)


def reciprocal_rank_at_25(
    rank: int | None,
) -> float:
    if rank is None:
        return 0.0

    if rank > 25:
        return 0.0

    return 1.0 / rank


def get_rank1_candidate(
    group: pd.DataFrame,
) -> pd.Series | None:
    """
    Return the candidate having spectral_rank == 1.

    The spectral candidate list is deterministic, so there should
    be at most one rank-1 candidate per query.
    """
    spectral_rank = pd.to_numeric(
        group["spectral_rank"],
        errors="coerce",
    )

    candidates = group[
        spectral_rank == 1
    ]

    if len(candidates) == 0:
        return None

    if len(candidates) != 1:
        raise RuntimeError(
            "Expected at most one spectral-rank-1 "
            "candidate per query."
        )

    return candidates.iloc[0]


def summarize_query(
    group: pd.DataFrame,
    current_rank: int | None,
) -> dict[str, object]:
    rank1 = get_rank1_candidate(
        group
    )

    truth_rows = group[
        group["is_truth"].astype(bool)
    ]

    if len(truth_rows) > 1:
        raise RuntimeError(
            "Expected at most one truth candidate."
        )

    truth_present = (
        len(truth_rows) == 1
    )

    if truth_present:
        truth_candidate_id = int(
            truth_rows.iloc[0][
                "candidate_id"
            ]
        )
    else:
        truth_candidate_id = None

    if rank1 is None:
        return {
            "truth_present": truth_present,
            "current_rank": current_rank,
            "rank1_present": False,
            "rank1_candidate_id": None,
            "rank1_tier": None,
            "rank1_is_truth": False,
            "rank1_cosine": None,
        }

    rank1_candidate_id = int(
        rank1["candidate_id"]
    )

    rank1_is_truth = (
        truth_candidate_id is not None
        and rank1_candidate_id
        == truth_candidate_id
    )

    rank1_cosine = float(
        rank1[
            "same_polarity_cosine"
        ]
    )

    if not np.isfinite(
        rank1_cosine
    ):
        rank1_cosine = None

    return {
        "truth_present": truth_present,
        "current_rank": current_rank,
        "rank1_present": True,
        "rank1_candidate_id": (
            rank1_candidate_id
        ),
        "rank1_tier": source_tier(
            rank1
        ),
        "rank1_is_truth": (
            rank1_is_truth
        ),
        "rank1_cosine": (
            rank1_cosine
        ),
    }


def promoted_rank(
    current_rank: int | None,
    rank1_tier: int | None,
    rank1_is_truth: bool,
) -> int | None:
    """
    Counterfactual rule:

        If the spectral-rank-1 candidate is Tier 3,
        promote it to rank 1.

        Otherwise preserve Baseline 3c exactly.

    We only need the truth rank to evaluate this rule.

    If the promoted candidate is truth:
        truth rank becomes 1.

    If the promoted candidate is not truth:
        an existing ranked truth is displaced by one position.

    If truth is absent from the candidate set:
        it remains absent.
    """
    if rank1_tier != 3:
        return current_rank

    if rank1_is_truth:
        return 1

    if current_rank is None:
        return None

    return current_rank + 1


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

    required_columns = {
        "inchikey14",
        "query_library",
        "candidate_id",
        "is_truth",
        "in_same_mass",
        "in_opposite_mass",
        "in_spectral_top25",
        "spectral_rank",
        "same_polarity_cosine",
    }

    missing = (
        required_columns
        - set(features.columns)
    )

    if missing:
        raise RuntimeError(
            "Feature artifact is missing "
            f"columns: {sorted(missing)}"
        )

    result_lookup = baseline3c.set_index(
        [
            "inchikey14",
            "query_library",
        ]
    )

    rows: list[
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
        key = (
            str(structure_key),
            str(query_library),
        )

        if key not in result_lookup.index:
            raise RuntimeError(
                "Missing Baseline 3c result for "
                f"{key[0]} / {key[1]}"
            )

        result = result_lookup.loc[
            key
        ]

        current_rank = optional_int(
            result["rank"]
        )

        summary = summarize_query(
            group=group,
            current_rank=current_rank,
        )

        counterfactual_rank = promoted_rank(
            current_rank=current_rank,
            rank1_tier=summary[
                "rank1_tier"
            ],
            rank1_is_truth=bool(
                summary[
                    "rank1_is_truth"
                ]
            ),
        )

        rows.append(
            {
                "inchikey14": key[0],
                "query_library": key[1],
                **summary,
                "counterfactual_rank": (
                    counterfactual_rank
                ),
            }
        )

    analysis = pd.DataFrame(
        rows
    )

    if len(analysis) != 2_000:
        raise RuntimeError(
            "Expected exactly 2,000 queries."
        )

    # ---------------------------------------------------------
    # Current and counterfactual metrics.
    # ---------------------------------------------------------

    analysis[
        "current_top1"
    ] = (
        analysis[
            "current_rank"
        ]
        == 1
    )

    analysis[
        "current_top25"
    ] = (
        analysis[
            "current_rank"
        ]
        .le(25)
        .fillna(False)
    )

    analysis[
        "counterfactual_top1"
    ] = (
        analysis[
            "counterfactual_rank"
        ]
        == 1
    )

    analysis[
        "counterfactual_top25"
    ] = (
        analysis[
            "counterfactual_rank"
        ]
        .le(25)
        .fillna(False)
    )

    analysis[
        "current_rr25"
    ] = [
        reciprocal_rank_at_25(
            optional_int(value)
        )
        for value in analysis[
            "current_rank"
        ]
    ]

    analysis[
        "counterfactual_rr25"
    ] = [
        reciprocal_rank_at_25(
            optional_int(value)
        )
        for value in analysis[
            "counterfactual_rank"
        ]
    ]

    current_mrr = float(
        analysis[
            "current_rr25"
        ].mean()
    )

    counterfactual_mrr = float(
        analysis[
            "counterfactual_rr25"
        ].mean()
    )

    current_top1 = float(
        analysis[
            "current_top1"
        ].mean()
    )

    counterfactual_top1 = float(
        analysis[
            "counterfactual_top1"
        ].mean()
    )

    current_top25 = float(
        analysis[
            "current_top25"
        ].mean()
    )

    counterfactual_top25 = float(
        analysis[
            "counterfactual_top25"
        ].mean()
    )

    # ---------------------------------------------------------
    # Rank-1 candidate precision.
    # ---------------------------------------------------------

    rank1 = analysis[
        analysis[
            "rank1_present"
        ]
    ].copy()

    tier3_rank1 = rank1[
        rank1[
            "rank1_tier"
        ]
        == 3
    ].copy()

    print()
    print(
        "Spectral rank-1 rescue diagnostic"
    )
    print(
        "---------------------------------"
    )

    print(
        f"Queries:                     "
        f"{len(analysis):,}"
    )

    print(
        f"Queries with spectral rank 1: "
        f"{len(rank1):,}"
    )

    if len(rank1):
        print(
            "Overall rank-1 truth precision: "
            f"{rank1['rank1_is_truth'].mean():.2%}"
        )

    # ---------------------------------------------------------
    # Precision by existing source tier.
    # ---------------------------------------------------------

    print()
    print(
        "Spectral rank-1 precision by source tier"
    )
    print(
        "----------------------------------------"
    )

    tier_summary = (
        rank1.groupby(
            "rank1_tier"
        )
        .agg(
            queries=(
                "inchikey14",
                "size",
            ),
            truths=(
                "rank1_is_truth",
                "sum",
            ),
            precision=(
                "rank1_is_truth",
                "mean",
            ),
        )
    )

    print(
        tier_summary.to_string(
            formatters={
                "precision": (
                    lambda x: f"{x:.2%}"
                )
            }
        )
    )

    # ---------------------------------------------------------
    # The proposed rescue population.
    # ---------------------------------------------------------

    print()
    print(
        "Tier-3 spectral-rank-1 population"
    )
    print(
        "--------------------------------"
    )

    print(
        f"Queries:       "
        f"{len(tier3_rank1):,}"
    )

    tier3_truths = int(
        tier3_rank1[
            "rank1_is_truth"
        ].sum()
    )

    print(
        f"Truths:        "
        f"{tier3_truths:,}"
    )

    if len(tier3_rank1):
        print(
            f"Precision:     "
            f"{tier3_rank1['rank1_is_truth'].mean():.2%}"
        )

    # ---------------------------------------------------------
    # How often would the rule fire on currently correct queries?
    # ---------------------------------------------------------

    tier3_current_top1 = tier3_rank1[
        tier3_rank1[
            "current_top1"
        ]
    ].copy()

    tier3_current_top25 = tier3_rank1[
        tier3_rank1[
            "current_top25"
        ]
    ].copy()

    wrong_promotions = tier3_rank1[
        ~tier3_rank1[
            "rank1_is_truth"
        ]
    ].copy()

    print()
    print(
        "Potential damage"
    )
    print(
        "----------------"
    )

    print(
        "Tier-3 rank-1 promotions that "
        "would be wrong: "
        f"{len(wrong_promotions):,}"
    )

    print(
        "Rule fires on current Top-1 "
        "successes: "
        f"{len(tier3_current_top1):,}"
    )

    print(
        "Wrong promotions on current "
        "Top-1 successes: "
        f"{int((~tier3_current_top1['rank1_is_truth']).sum()):,}"
    )

    print(
        "Rule fires on current Top-25 "
        "successes: "
        f"{len(tier3_current_top25):,}"
    )

    print(
        "Wrong promotions on current "
        "Top-25 successes: "
        f"{int((~tier3_current_top25['rank1_is_truth']).sum()):,}"
    )

    # ---------------------------------------------------------
    # Top-25 transitions.
    # ---------------------------------------------------------

    regressions = analysis[
        analysis[
            "current_top25"
        ]
        & ~analysis[
            "counterfactual_top25"
        ]
    ].copy()

    recoveries = analysis[
        ~analysis[
            "current_top25"
        ]
        & analysis[
            "counterfactual_top25"
        ]
    ].copy()

    print()
    print(
        "Counterfactual Top-25 transitions"
    )
    print(
        "---------------------------------"
    )

    print(
        f"Regressions: "
        f"{len(regressions):,}"
    )

    print(
        f"Recoveries:  "
        f"{len(recoveries):,}"
    )

    # ---------------------------------------------------------
    # Metric impact.
    # ---------------------------------------------------------

    print()
    print(
        "Counterfactual metric impact"
    )
    print(
        "----------------------------"
    )

    print(
        f"Baseline 3c MRR@25: "
        f"{current_mrr:.4f}"
    )

    print(
        f"Counterfactual MRR:  "
        f"{counterfactual_mrr:.4f}"
    )

    print(
        f"Delta:               "
        f"{counterfactual_mrr - current_mrr:+.4f}"
    )

    print()

    print(
        f"Baseline 3c Top-1:   "
        f"{current_top1:.2%}"
    )

    print(
        f"Counterfactual Top-1:"
        f" {counterfactual_top1:.2%}"
    )

    print(
        f"Delta:               "
        f"{100 * (counterfactual_top1 - current_top1):+.2f} pp"
    )

    print()

    print(
        f"Baseline 3c Top-25:  "
        f"{current_top25:.2%}"
    )

    print(
        f"Counterfactual Top-25:"
        f" {counterfactual_top25:.2%}"
    )

    print(
        f"Delta:               "
        f"{100 * (counterfactual_top25 - current_top25):+.2f} pp"
    )

    # ---------------------------------------------------------
    # Detailed wrong promotions.
    # ---------------------------------------------------------

    print()
    print(
        "Wrong Tier-3 rank-1 promotions"
    )
    print(
        "------------------------------"
    )

    print(
        f"Count: {len(wrong_promotions):,}"
    )

    if len(wrong_promotions):
        print()

        wrong_columns = [
            "inchikey14",
            "query_library",
            "current_rank",
            "rank1_candidate_id",
            "rank1_cosine",
            "current_top1",
            "current_top25",
        ]

        print(
            wrong_promotions[
                wrong_columns
            ]
            .sort_values(
                [
                    "current_top1",
                    "current_top25",
                    "current_rank",
                ],
                ascending=[
                    False,
                    False,
                    True,
                ],
                na_position="last",
            )
            .to_string(
                index=False
            )
        )

    # ---------------------------------------------------------
    # Detailed correct promotions.
    # ---------------------------------------------------------

    correct_promotions = tier3_rank1[
        tier3_rank1[
            "rank1_is_truth"
        ]
    ].copy()

    print()
    print(
        "Correct Tier-3 rank-1 promotions"
    )
    print(
        "-------------------------------"
    )

    print(
        f"Count: {len(correct_promotions):,}"
    )

    if len(correct_promotions):
        print()

        print(
            correct_promotions[
                [
                    "inchikey14",
                    "query_library",
                    "current_rank",
                    "rank1_cosine",
                    "current_top1",
                    "current_top25",
                ]
            ]
            .sort_values(
                "current_rank",
                na_position="last",
            )
            .to_string(
                index=False
            )
        )

    # ---------------------------------------------------------
    # Top-25 regression/recovery details.
    # ---------------------------------------------------------

    if len(regressions):
        print()
        print(
            "Top-25 regressions"
        )
        print(
            "------------------"
        )

        print(
            regressions[
                [
                    "inchikey14",
                    "query_library",
                    "current_rank",
                    "counterfactual_rank",
                    "rank1_is_truth",
                    "rank1_tier",
                    "rank1_cosine",
                ]
            ].to_string(
                index=False
            )
        )

    if len(recoveries):
        print()
        print(
            "Top-25 recoveries"
        )
        print(
            "-----------------"
        )

        print(
            recoveries[
                [
                    "inchikey14",
                    "query_library",
                    "current_rank",
                    "counterfactual_rank",
                    "rank1_is_truth",
                    "rank1_tier",
                    "rank1_cosine",
                ]
            ]
            .sort_values(
                "current_rank"
            )
            .to_string(
                index=False
            )
        )

    # ---------------------------------------------------------
    # Save all queries for subsequent diagnostics.
    # ---------------------------------------------------------

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    analysis.to_parquet(
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