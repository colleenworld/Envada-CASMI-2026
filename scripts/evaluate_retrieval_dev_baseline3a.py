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


def rank_query(
    group: pd.DataFrame,
) -> int | None:
    """
    Rank candidates by same-polarity cosine.

    Higher cosine is better. Candidate ID provides deterministic
    tie-breaking only.

    Returns the 1-based truth rank, or None when the truth is not
    present in the hybrid candidate set.
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

    ranked = group.sort_values(
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
            "Truth candidate was lost during ranking."
        )

    return int(
        matches[0] + 1
    )


def reciprocal_rank_at_25(
    rank: int | float | None,
) -> float:
    if rank is None:
        return 0.0

    if pd.isna(rank):
        return 0.0

    if rank > 25:
        return 0.0

    return 1.0 / rank


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
                "reciprocal_rank_at_25": (
                    reciprocal_rank_at_25(
                        rank
                    )
                ),
            }
        )

    results = pd.DataFrame(
        rows
    )

    # The eight queries whose truth is absent from the hybrid
    # candidate set are not present as truth rows, but their query
    # groups still exist, so all 2,000 structures must be evaluated.
    if len(results) != 2_000:
        raise RuntimeError(
            "Expected exactly 2,000 retrieval-dev queries."
        )

    mrr = results[
        "reciprocal_rank_at_25"
    ].mean()

    top1 = (
        results["rank"] == 1
    ).mean()

    top25 = (
        results["rank"].notna()
        & (
            results["rank"] <= 25
        )
    ).mean()

    candidate_recall = (
        results["rank"].notna()
    ).mean()

    print(
        "Retrieval Baseline 3a"
    )
    print(
        "---------------------"
    )

    print(
        "Hybrid candidates ranked by "
        "same-polarity cosine"
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

    # ---------------------------------------------------------
    # Baseline 2 comparison.
    # ---------------------------------------------------------

    baseline2_ranks = (
        baseline2["rank"]
    )

    baseline2_mrr = (
        baseline2_ranks
        .map(
            reciprocal_rank_at_25
        )
        .mean()
    )

    baseline2_top1 = (
        baseline2_ranks == 1
    ).mean()

    baseline2_top25 = (
        baseline2_ranks.notna()
        & (
            baseline2_ranks <= 25
        )
    ).mean()

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


if __name__ == "__main__":
    main()