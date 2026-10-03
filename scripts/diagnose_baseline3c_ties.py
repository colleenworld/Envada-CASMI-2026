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

BASELINE3C_PATH = Path(
    "data/processed/results/"
    "retrieval_dev_baseline3c.parquet"
)


def normalize_rank(
    value: object,
) -> int | None:
    if pd.isna(
        value
    ):
        return None

    return int(
        value
    )


def ranks_equal(
    left: object,
    right: object,
) -> bool:
    return (
        normalize_rank(
            left
        )
        == normalize_rank(
            right
        )
    )


def reciprocal_rank_at_25(
    rank: int | None,
) -> float:
    if rank is None:
        return 0.0

    if rank > 25:
        return 0.0

    return 1.0 / rank


def get_truth_row(
    group: pd.DataFrame,
) -> pd.Series | None:
    truth = group[
        group[
            "is_truth"
        ].astype(bool)
    ]

    if len(truth) == 0:
        return None

    if len(truth) != 1:
        raise RuntimeError(
            "Expected at most one truth candidate."
        )

    return truth.iloc[0]


def classify_difference(
    group: pd.DataFrame,
    baseline2_rank: int | None,
    baseline3c_rank: int | None,
    truth_source_tier: int | None,
    baseline2_fallback: bool,
) -> str:
    """
    Classify a Baseline 2 -> Baseline 3c rank difference.

    Tie analysis is appropriate only when both systems have a
    genuinely comparable score path.

    In particular:

    - Normal Tier-1:
        exact Baseline-2-compatible same_mass_cosine.

    - Fallback with finite truth same_polarity_cosine:
        comparable fallback scoring path.

    - Fallback with non-finite truth same_polarity_cosine:
        NOT a tie-comparison case.

        The truth has no same-polarity spectral ranking evidence,
        but Baseline 3c may still contain it because of hybrid
        mass/spectral candidate generation. Its deterministic
        position is therefore an architectural hybrid behavior,
        not a discrepancy in Baseline 2 score reproduction.
    """
    truth_row = get_truth_row(
        group
    )

    if truth_row is None:
        return "truth_absent"

    if baseline2_fallback:
        truth_score = float(
            truth_row[
                "same_polarity_cosine"
            ]
        )

        if np.isfinite(
            truth_score
        ):
            return "fallback_comparable"

        return "fallback_unscored_hybrid"

    if truth_source_tier == 1:
        return "tier1_nonfallback"

    if (
        truth_source_tier in (
            2,
            3,
        )
        and baseline2_rank is None
        and baseline3c_rank is not None
    ):
        return "hybrid_recovery"

    if truth_source_tier in (
        2,
        3,
    ):
        return "hybrid_other"

    return "other"


def diagnose_tie(
    group: pd.DataFrame,
    context: str,
    baseline2_rank: int | None,
    baseline3c_rank: int | None,
) -> dict[str, object]:
    """
    Test whether a comparable-path rank difference is exactly
    explained by deterministic ordering of equal scores.

    Baseline 2 used optimistic tie semantics:

        rank = 1 + count(scores > truth_score)

    Baseline 3c produces an actual deterministic list:

        score descending
        candidate_id ascending
    """
    truth_row = get_truth_row(
        group
    )

    if truth_row is None:
        raise RuntimeError(
            "Tie diagnosis requires a truth row."
        )

    if context == "tier1_nonfallback":
        ranked_group = group[
            group[
                "in_same_mass"
            ].astype(bool)
        ].copy()

        score_column = (
            "same_mass_cosine"
        )

    elif context == "fallback_comparable":
        ranked_group = group.copy()

        score_column = (
            "same_polarity_cosine"
        )

    else:
        raise ValueError(
            "Unsupported tie context: "
            f"{context}"
        )

    truth_score = float(
        truth_row[
            score_column
        ]
    )

    if not np.isfinite(
        truth_score
    ):
        raise RuntimeError(
            "Comparable tie diagnosis received "
            "a non-finite truth score."
        )

    scores = ranked_group[
        score_column
    ].to_numpy(
        dtype=np.float64
    )

    candidate_ids = ranked_group[
        "candidate_id"
    ].to_numpy(
        dtype=np.int64
    )

    truth_candidate_id = int(
        truth_row[
            "candidate_id"
        ]
    )

    finite = np.isfinite(
        scores
    )

    finite_scores = scores[
        finite
    ]

    finite_ids = candidate_ids[
        finite
    ]

    greater_mask = (
        finite_scores
        > truth_score
    )

    # Use exact equality because these are the persisted scores
    # used by the ranker.
    equal_mask = (
        finite_scores
        == truth_score
    )

    greater_count = int(
        greater_mask.sum()
    )

    equal_ids = finite_ids[
        equal_mask
    ]

    equal_count = int(
        len(
            equal_ids
        )
    )

    equal_before_truth = int(
        (
            equal_ids
            < truth_candidate_id
        ).sum()
    )

    optimistic_rank = (
        1
        + greater_count
    )

    deterministic_rank = (
        optimistic_rank
        + equal_before_truth
    )

    tie_explains = (
        baseline2_rank
        == optimistic_rank
        and baseline3c_rank
        == deterministic_rank
    )

    return {
        "truth_score": (
            truth_score
        ),
        "greater": (
            greater_count
        ),
        "equal": (
            equal_count
        ),
        "equal_before_truth": (
            equal_before_truth
        ),
        "optimistic_rank": (
            optimistic_rank
        ),
        "deterministic_rank": (
            deterministic_rank
        ),
        "tie_explains": (
            tie_explains
        ),
    }


def main() -> None:
    print(
        "Loading frozen artifacts..."
    )

    features = pd.read_parquet(
        FEATURES_PATH
    )

    baseline2 = pd.read_parquet(
        BASELINE2_PATH
    )

    baseline3c = pd.read_parquet(
        BASELINE3C_PATH
    )

    # ---------------------------------------------------------
    # Build query-level comparison.
    # ---------------------------------------------------------

    baseline2_lookup = (
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

    baseline3c_lookup = (
        baseline3c[
            [
                "inchikey14",
                "query_library",
                "rank",
                "truth_source_tier",
                "baseline2_fallback",
            ]
        ]
        .rename(
            columns={
                "rank": "baseline3c_rank",
            }
        )
    )

    comparison = baseline3c_lookup.merge(
        baseline2_lookup,
        on=[
            "inchikey14",
            "query_library",
        ],
        how="left",
        validate="one_to_one",
    )

    if len(
        comparison
    ) != 2_000:
        raise RuntimeError(
            "Expected exactly 2,000 "
            "retrieval-dev queries."
        )

    comparison[
        "rank_matches"
    ] = [
        ranks_equal(
            baseline2_rank,
            baseline3c_rank,
        )
        for (
            baseline2_rank,
            baseline3c_rank,
        ) in zip(
            comparison[
                "baseline2_rank"
            ],
            comparison[
                "baseline3c_rank"
            ],
            strict=True,
        )
    ]

    differences = comparison[
        ~comparison[
            "rank_matches"
        ]
    ].copy()

    print()
    print(
        "Baseline 3c rank-difference diagnostic"
    )
    print(
        "--------------------------------------"
    )

    print(
        f"Total queries:       "
        f"{len(comparison):,}"
    )

    print(
        f"Exact rank matches:  "
        f"{int(comparison['rank_matches'].sum()):,}"
    )

    print(
        f"Rank differences:    "
        f"{len(differences):,}"
    )

    # ---------------------------------------------------------
    # Feature lookup.
    # ---------------------------------------------------------

    grouped_features = {
        (
            str(
                structure_key
            ),
            str(
                query_library
            ),
        ): group
        for (
            structure_key,
            query_library,
        ), group in features.groupby(
            [
                "inchikey14",
                "query_library",
            ],
            sort=False,
        )
    }

    # ---------------------------------------------------------
    # Classify every difference.
    # ---------------------------------------------------------

    classifications: list[
        str
    ] = []

    for row in differences.itertuples(
        index=False
    ):
        key = (
            str(
                row.inchikey14
            ),
            str(
                row.query_library
            ),
        )

        group = grouped_features.get(
            key
        )

        if group is None:
            raise RuntimeError(
                "No ranking features for "
                f"{key[0]} / {key[1]}"
            )

        truth_source_tier = (
            None
            if pd.isna(
                row.truth_source_tier
            )
            else int(
                row.truth_source_tier
            )
        )

        classifications.append(
            classify_difference(
                group=group,
                baseline2_rank=(
                    normalize_rank(
                        row.baseline2_rank
                    )
                ),
                baseline3c_rank=(
                    normalize_rank(
                        row.baseline3c_rank
                    )
                ),
                truth_source_tier=(
                    truth_source_tier
                ),
                baseline2_fallback=bool(
                    row.baseline2_fallback
                ),
            )
        )

    differences[
        "classification"
    ] = classifications

    print()
    print(
        "Difference classification"
    )
    print(
        "-------------------------"
    )

    classification_order = [
        "tier1_nonfallback",
        "fallback_comparable",
        "fallback_unscored_hybrid",
        "hybrid_recovery",
        "hybrid_other",
        "truth_absent",
        "other",
    ]

    for classification in (
        classification_order
    ):
        count = int(
            (
                differences[
                    "classification"
                ]
                == classification
            ).sum()
        )

        print(
            f"{classification:26} "
            f"{count:4}"
        )

    # ---------------------------------------------------------
    # Comparable-path tie analysis.
    # ---------------------------------------------------------

    comparable_contexts = {
        "tier1_nonfallback",
        "fallback_comparable",
    }

    comparable = differences[
        differences[
            "classification"
        ].isin(
            comparable_contexts
        )
    ].copy()

    tie_rows: list[
        dict[str, object]
    ] = []

    for row in comparable.itertuples(
        index=False
    ):
        structure_key = str(
            row.inchikey14
        )

        query_library = str(
            row.query_library
        )

        group = grouped_features[
            (
                structure_key,
                query_library,
            )
        ]

        baseline2_rank = (
            normalize_rank(
                row.baseline2_rank
            )
        )

        baseline3c_rank = (
            normalize_rank(
                row.baseline3c_rank
            )
        )

        diagnosis = diagnose_tie(
            group=group,
            context=str(
                row.classification
            ),
            baseline2_rank=(
                baseline2_rank
            ),
            baseline3c_rank=(
                baseline3c_rank
            ),
        )

        tie_rows.append(
            {
                "inchikey14": (
                    structure_key
                ),
                "query_library": (
                    query_library
                ),
                "context": (
                    row.classification
                ),
                "baseline2_rank": (
                    baseline2_rank
                ),
                "baseline3c_rank": (
                    baseline3c_rank
                ),
                **diagnosis,
            }
        )

    tie_diagnostics = pd.DataFrame(
        tie_rows
    )

    print()
    print(
        "Comparable-path tie analysis"
    )
    print(
        "----------------------------"
    )

    print(
        f"Cases to diagnose:   "
        f"{len(tie_diagnostics):,}"
    )

    if len(
        tie_diagnostics
    ):
        explained_count = int(
            tie_diagnostics[
                "tie_explains"
            ].sum()
        )
    else:
        explained_count = 0

    unexplained_count = (
        len(
            tie_diagnostics
        )
        - explained_count
    )

    print(
        f"Tie-explained:       "
        f"{explained_count:,} / "
        f"{len(tie_diagnostics):,}"
    )

    print(
        f"Unexplained:         "
        f"{unexplained_count:,}"
    )

    if len(
        tie_diagnostics
    ):
        print()
        print(
            "Tie details"
        )
        print(
            "-----------"
        )

        tie_columns = [
            "inchikey14",
            "query_library",
            "context",
            "baseline2_rank",
            "baseline3c_rank",
            "truth_score",
            "greater",
            "equal",
            "equal_before_truth",
            "optimistic_rank",
            "deterministic_rank",
            "tie_explains",
        ]

        print(
            tie_diagnostics[
                tie_columns
            ]
            .sort_values(
                [
                    "context",
                    "baseline2_rank",
                    "baseline3c_rank",
                    "inchikey14",
                ]
            )
            .to_string(
                index=False
            )
        )

    # ---------------------------------------------------------
    # Explicitly report unscored hybrid fallback behavior.
    # ---------------------------------------------------------

    unscored_fallback = differences[
        differences[
            "classification"
        ]
        == "fallback_unscored_hybrid"
    ].copy()

    print()
    print(
        "Unscored hybrid fallback differences"
    )
    print(
        "------------------------------------"
    )

    print(
        f"Count:               "
        f"{len(unscored_fallback):,}"
    )

    if len(
        unscored_fallback
    ):
        print()

        print(
            unscored_fallback[
                [
                    "inchikey14",
                    "query_library",
                    "baseline2_rank",
                    "baseline3c_rank",
                    "truth_source_tier",
                    "baseline2_fallback",
                ]
            ].to_string(
                index=False
            )
        )

    # ---------------------------------------------------------
    # Architectural differences.
    # ---------------------------------------------------------

    architectural_classes = {
        "fallback_unscored_hybrid",
        "hybrid_recovery",
        "hybrid_other",
        "truth_absent",
    }

    architectural = differences[
        differences[
            "classification"
        ].isin(
            architectural_classes
        )
    ].copy()

    print()
    print(
        "Architectural differences"
    )
    print(
        "-------------------------"
    )

    print(
        f"Count:               "
        f"{len(architectural):,}"
    )

    # ---------------------------------------------------------
    # Unexpected differences.
    # ---------------------------------------------------------

    unexpected = differences[
        differences[
            "classification"
        ]
        == "other"
    ].copy()

    print()
    print(
        "Unexpected differences"
    )
    print(
        "----------------------"
    )

    print(
        f"Count:               "
        f"{len(unexpected):,}"
    )

    if len(
        unexpected
    ):
        print()

        print(
            unexpected[
                [
                    "inchikey14",
                    "query_library",
                    "baseline2_rank",
                    "baseline3c_rank",
                    "truth_source_tier",
                    "baseline2_fallback",
                ]
            ].to_string(
                index=False
            )
        )

    # ---------------------------------------------------------
    # Top-25 transition sanity check.
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
            "baseline3c_rank"
        ]
        .le(25)
        .fillna(False)
    )

    regressions = int(
        (
            comparison[
                "baseline2_top25"
            ]
            & ~comparison[
                "baseline3c_top25"
            ]
        ).sum()
    )

    recoveries = int(
        (
            ~comparison[
                "baseline2_top25"
            ]
            & comparison[
                "baseline3c_top25"
            ]
        ).sum()
    )

    print()
    print(
        "Top-25 sanity check"
    )
    print(
        "-------------------"
    )

    print(
        f"Regressions:        "
        f"{regressions:,}"
    )

    print(
        f"Recoveries:         "
        f"{recoveries:,}"
    )

    # ---------------------------------------------------------
    # Metric sanity check.
    # ---------------------------------------------------------

    baseline2_mrr = float(
        np.mean(
            [
                reciprocal_rank_at_25(
                    normalize_rank(
                        value
                    )
                )
                for value in comparison[
                    "baseline2_rank"
                ]
            ]
        )
    )

    baseline3c_mrr = float(
        np.mean(
            [
                reciprocal_rank_at_25(
                    normalize_rank(
                        value
                    )
                )
                for value in comparison[
                    "baseline3c_rank"
                ]
            ]
        )
    )

    print()
    print(
        "Metric sanity check"
    )
    print(
        "-------------------"
    )

    print(
        f"Baseline 2 MRR@25:  "
        f"{baseline2_mrr:.4f}"
    )

    print(
        f"Baseline 3c MRR@25: "
        f"{baseline3c_mrr:.4f}"
    )

    # ---------------------------------------------------------
    # Final diagnosis.
    # ---------------------------------------------------------

    print()
    print(
        "Final diagnosis"
    )
    print(
        "---------------"
    )

    failures: list[
        str
    ] = []

    if unexplained_count:
        failures.append(
            f"{unexplained_count} comparable-path "
            "differences are not explained by ties"
        )

    if len(
        unexpected
    ):
        failures.append(
            f"{len(unexpected)} rank differences "
            "have an unexpected classification"
        )

    if regressions != 0:
        failures.append(
            f"expected 0 Top-25 regressions, "
            f"found {regressions}"
        )

    if recoveries != 24:
        failures.append(
            f"expected 24 Top-25 recoveries, "
            f"found {recoveries}"
        )

    if failures:
        print(
            "FAIL:"
        )

        for failure in failures:
            print(
                f"  - {failure}"
            )

        raise RuntimeError(
            "Baseline 3c diagnostic found "
            "unexplained differences."
        )

    print(
        "PASS: all comparable-path rank differences "
        "are explained by deterministic tie ordering."
    )

    print(
        "PASS: unscored fallback candidates are "
        "classified as hybrid architectural behavior, "
        "not Baseline 2 score mismatches."
    )

    print(
        "PASS: remaining rank differences are "
        "expected consequences of hybrid candidate "
        "generation."
    )

    print(
        "PASS: Baseline 3c has 0 Top-25 regressions "
        "and 24 recoveries versus Baseline 2."
    )


if __name__ == "__main__":
    main()