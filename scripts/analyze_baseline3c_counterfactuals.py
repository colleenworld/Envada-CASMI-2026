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
    "retrieval_dev_baseline3c_counterfactuals.parquet"
)


def optional_float(
    value: object,
) -> float | None:
    if pd.isna(value):
        return None

    value = float(value)

    if not np.isfinite(value):
        return None

    return value


def optional_int(
    value: object,
) -> int | None:
    if pd.isna(value):
        return None

    return int(value)


def truth_source_tier(
    truth: pd.Series,
) -> int:
    if bool(truth["in_same_mass"]):
        return 1

    if bool(truth["in_opposite_mass"]):
        return 2

    return 3


def deterministic_rank(
    frame: pd.DataFrame,
    score_column: str,
    truth_candidate_id: int,
) -> int | None:
    """
    Rank candidates by:

        score descending
        candidate_id ascending

    Non-finite scores remain in the ranking. This deliberately
    matches the deterministic ordering semantics used by Baseline 3c.
    """
    if len(frame) == 0:
        return None

    ranked = frame.sort_values(
        by=[
            score_column,
            "candidate_id",
        ],
        ascending=[
            False,
            True,
        ],
        kind="stable",
    )

    positions = np.flatnonzero(
        ranked[
            "candidate_id"
        ].to_numpy(
            dtype=np.int64
        )
        == truth_candidate_id
    )

    if len(positions) == 0:
        return None

    if len(positions) != 1:
        raise RuntimeError(
            "Truth candidate appears more than once."
        )

    return int(positions[0]) + 1


def optimistic_rank(
    frame: pd.DataFrame,
    score_column: str,
    truth_candidate_id: int,
) -> int | None:
    """
    Baseline-2-style optimistic rank:

        1 + count(scores > truth_score)

    Returns None if the truth is absent or has no finite score.
    """
    truth_rows = frame[
        frame[
            "candidate_id"
        ]
        == truth_candidate_id
    ]

    if len(truth_rows) == 0:
        return None

    if len(truth_rows) != 1:
        raise RuntimeError(
            "Truth candidate appears more than once."
        )

    truth_score = optional_float(
        truth_rows.iloc[0][
            score_column
        ]
    )

    if truth_score is None:
        return None

    scores = frame[
        score_column
    ].to_numpy(
        dtype=np.float64
    )

    return 1 + int(
        (
            scores
            > truth_score
        ).sum()
    )


def summarize_query(
    group: pd.DataFrame,
    current_rank: int | None,
) -> dict[str, object]:
    truth_rows = group[
        group[
            "is_truth"
        ].astype(bool)
    ]

    if len(truth_rows) == 0:
        return {
            "truth_present": False,
            "truth_tier": None,
            "current_rank": current_rank,
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
            "truth_same_mass_cosine": None,
            "truth_full_cosine": None,
            "truth_spectral_rank": None,
            "truth_full_cosine_rank": None,
            "truth_full_cosine_optimistic_rank": None,
            "truth_tier1_rank": None,
            "tier1_blockers": None,
            "current_top25_lower_full_cosine": None,
            "current_top25_nonfinite_full_cosine": None,
        }

    if len(truth_rows) != 1:
        raise RuntimeError(
            "Expected exactly one truth candidate."
        )

    truth = truth_rows.iloc[0]

    truth_candidate_id = int(
        truth[
            "candidate_id"
        ]
    )

    tier = truth_source_tier(
        truth
    )

    tier1_mask = (
        group[
            "in_same_mass"
        ].astype(bool)
    )

    tier1 = group[
        tier1_mask
    ].copy()

    truth_full_cosine = optional_float(
        truth[
            "same_polarity_cosine"
        ]
    )

    full_cosine_rank = deterministic_rank(
        frame=group,
        score_column="same_polarity_cosine",
        truth_candidate_id=truth_candidate_id,
    )

    full_cosine_optimistic_rank = (
        optimistic_rank(
            frame=group,
            score_column="same_polarity_cosine",
            truth_candidate_id=truth_candidate_id,
        )
    )

    if tier == 1:
        tier1_rank = deterministic_rank(
            frame=tier1,
            score_column="same_mass_cosine",
            truth_candidate_id=truth_candidate_id,
        )
    else:
        tier1_rank = None

    # Under the hard hierarchy, every Tier-1 candidate blocks a
    # Tier-2/3 truth. A Tier-3 truth is additionally blocked by
    # Tier-2 candidates, but keeping Tier-1 blockers separate tells
    # us how much of the problem is caused by the protected mass tier.
    if tier in (2, 3):
        tier1_blockers = len(
            tier1
        )
    else:
        tier1_blockers = 0

    # Reconstruct the actual current 3c ordering so that we can
    # inspect the candidates occupying its Top 25.
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
            "within query."
        )

    used_fallback = bool(
        fallback_values[0]
    )

    ranked = group.copy()

    if used_fallback:
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
        in_same = ranked[
            "in_same_mass"
        ].astype(bool)

        in_opposite = ranked[
            "in_opposite_mass"
        ].astype(bool)

        ranked[
            "_tier"
        ] = 3

        ranked.loc[
            in_opposite,
            "_tier",
        ] = 2

        ranked.loc[
            in_same,
            "_tier",
        ] = 1

        ranked[
            "_score"
        ] = np.where(
            ranked[
                "_tier"
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
                "_tier",
                "_score",
                "candidate_id",
            ],
            ascending=[
                True,
                False,
                True,
            ],
            kind="stable",
        )

    current_top25 = ranked.head(
        25
    )

    if truth_full_cosine is None:
        lower_full_cosine = None
        nonfinite_full_cosine = int(
            (
                ~np.isfinite(
                    current_top25[
                        "same_polarity_cosine"
                    ].to_numpy(
                        dtype=np.float64
                    )
                )
            ).sum()
        )
    else:
        top25_scores = current_top25[
            "same_polarity_cosine"
        ].to_numpy(
            dtype=np.float64
        )

        lower_full_cosine = int(
            (
                top25_scores
                < truth_full_cosine
            ).sum()
        )

        nonfinite_full_cosine = int(
            (
                ~np.isfinite(
                    top25_scores
                )
            ).sum()
        )

    return {
        "truth_present": True,
        "truth_tier": tier,
        "current_rank": current_rank,
        "candidate_count": len(group),
        "tier1_count": len(tier1),
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
        "truth_same_mass_cosine": (
            optional_float(
                truth[
                    "same_mass_cosine"
                ]
            )
        ),
        "truth_full_cosine": (
            truth_full_cosine
        ),
        "truth_spectral_rank": (
            optional_int(
                truth[
                    "spectral_rank"
                ]
            )
        ),
        "truth_full_cosine_rank": (
            full_cosine_rank
        ),
        "truth_full_cosine_optimistic_rank": (
            full_cosine_optimistic_rank
        ),
        "truth_tier1_rank": (
            tier1_rank
        ),
        "tier1_blockers": (
            tier1_blockers
        ),
        "current_top25_lower_full_cosine": (
            lower_full_cosine
        ),
        "current_top25_nonfinite_full_cosine": (
            nonfinite_full_cosine
        ),
    }


def print_counts(
    title: str,
    series: pd.Series,
) -> None:
    print()
    print(
        title
    )
    print(
        "-" * len(title)
    )

    print(
        series
        .value_counts(
            dropna=False
        )
        .sort_index()
        .to_string()
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
        "same_mass_cosine",
        "baseline2_fallback",
    }

    missing = (
        required_columns
        - set(
            features.columns
        )
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
            str(
                structure_key
            ),
            str(
                query_library
            ),
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
            result[
                "rank"
            ]
        )

        rows.append(
            {
                "inchikey14": key[0],
                "query_library": key[1],
                **summarize_query(
                    group=group,
                    current_rank=current_rank,
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
        "ranking_miss"
    ] = (
        analysis[
            "truth_present"
        ]
        & ~analysis[
            "current_top25"
        ]
    )

    misses = analysis[
        analysis[
            "ranking_miss"
        ]
    ].copy()

    if len(misses) != 32:
        raise RuntimeError(
            "Expected exactly 32 "
            "truth-present Top-25 misses, "
            f"found {len(misses)}."
        )

    # ---------------------------------------------------------
    # Headline opportunity.
    # ---------------------------------------------------------

    misses[
        "full_cosine_top25"
    ] = (
        misses[
            "truth_full_cosine_rank"
        ]
        .le(25)
        .fillna(False)
    )

    misses[
        "spectral_top25"
    ] = (
        misses[
            "truth_spectral_rank"
        ]
        .le(25)
        .fillna(False)
    )

    print()
    print(
        "Baseline 3c counterfactual analysis"
    )
    print(
        "----------------------------------"
    )

    print(
        f"Truth-present Top-25 misses: "
        f"{len(misses):,}"
    )

    print(
        "Recoverable by hybrid-set full "
        "cosine Top-25: "
        f"{int(misses['full_cosine_top25'].sum()):,}"
    )

    print(
        "Already present in spectral "
        "Top-25 source C: "
        f"{int(misses['spectral_top25'].sum()):,}"
    )

    # ---------------------------------------------------------
    # Opportunity by truth tier.
    # ---------------------------------------------------------

    print()
    print(
        "Counterfactual full-cosine Top-25 by truth tier"
    )
    print(
        "----------------------------------------------"
    )

    opportunity = (
        misses.groupby(
            "truth_tier"
        )
        .agg(
            misses=(
                "inchikey14",
                "size",
            ),
            full_cosine_top25=(
                "full_cosine_top25",
                "sum",
            ),
            spectral_top25=(
                "spectral_top25",
                "sum",
            ),
            median_current_rank=(
                "current_rank",
                "median",
            ),
            median_full_cosine_rank=(
                "truth_full_cosine_rank",
                "median",
            ),
        )
    )

    print(
        opportunity.to_string()
    )

    # ---------------------------------------------------------
    # Tier blockers.
    # ---------------------------------------------------------

    rescue_misses = misses[
        misses[
            "truth_tier"
        ].isin(
            [
                2,
                3,
            ]
        )
    ].copy()

    print()
    print(
        "Tier-2/3 blocker analysis"
    )
    print(
        "-------------------------"
    )

    print(
        f"Tier-2/3 misses:                 "
        f"{len(rescue_misses):,}"
    )

    print(
        "Blocked by >=25 Tier-1 candidates: "
        f"{int((rescue_misses['tier1_blockers'] >= 25).sum()):,}"
    )

    print(
        "Blocked by <25 Tier-1 candidates:  "
        f"{int((rescue_misses['tier1_blockers'] < 25).sum()):,}"
    )

    # ---------------------------------------------------------
    # Existing Top-25 candidates weaker than truth.
    # ---------------------------------------------------------

    finite_truth = misses[
        misses[
            "truth_full_cosine"
        ].notna()
    ].copy()

    print()
    print(
        "Spectral evidence versus current Top-25"
    )
    print(
        "--------------------------------------"
    )

    print(
        f"Misses with finite truth cosine: "
        f"{len(finite_truth):,}"
    )

    print(
        "Truth beats >=1 current Top-25 "
        "candidate by full cosine: "
        f"{int((finite_truth['current_top25_lower_full_cosine'] > 0).sum()):,}"
    )

    print(
        "Truth beats >=5 current Top-25 "
        "candidates by full cosine: "
        f"{int((finite_truth['current_top25_lower_full_cosine'] >= 5).sum()):,}"
    )

    print(
        "Truth beats >=10 current Top-25 "
        "candidates by full cosine: "
        f"{int((finite_truth['current_top25_lower_full_cosine'] >= 10).sum()):,}"
    )

    # ---------------------------------------------------------
    # Detailed miss table.
    # ---------------------------------------------------------

    detail_columns = [
        "inchikey14",
        "query_library",
        "truth_tier",
        "current_rank",
        "candidate_count",
        "tier1_count",
        "tier2_count",
        "tier3_count",
        "tier1_blockers",
        "truth_same_mass_cosine",
        "truth_full_cosine",
        "truth_spectral_rank",
        "truth_full_cosine_rank",
        "truth_full_cosine_optimistic_rank",
        "truth_tier1_rank",
        "current_top25_lower_full_cosine",
        "full_cosine_top25",
        "spectral_top25",
    ]

    print()
    print(
        "Detailed ranking misses"
    )
    print(
        "-----------------------"
    )

    print(
        misses[
            detail_columns
        ]
        .sort_values(
            [
                "truth_tier",
                "current_rank",
                "query_library",
            ],
            na_position="last",
        )
        .to_string(
            index=False
        )
    )

    # ---------------------------------------------------------
    # Save all 2,000 query-level counterfactuals.
    #
    # Saving all queries is intentional: when we design a ranking
    # rule later, we need to measure both recoveries AND damage to
    # currently successful queries.
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