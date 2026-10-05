from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


INPUT_PATH = Path(
    "data/processed/mist/benchmark_100/mist_ranked_candidates.parquet"
)


def metrics(ranks: np.ndarray) -> dict[str, float]:
    ranks = np.asarray(ranks)

    return {
        "queries": len(ranks),
        "mrr@25": np.mean(
            np.where(ranks <= 25, 1.0 / ranks, 0.0)
        ),
        "top1": np.mean(ranks <= 1),
        "top5": np.mean(ranks <= 5),
        "top10": np.mean(ranks <= 10),
        "top25": np.mean(ranks <= 25),
        "median_rank": np.median(ranks),
    }


def expected_random_metrics(candidate_counts: np.ndarray) -> dict[str, float]:
    """
    Expected metrics for uniformly random ranking within each query's
    candidate set.
    """

    candidate_counts = np.asarray(candidate_counts)

    # Expected reciprocal rank truncated at 25:
    #
    # E[RR@25] = (1 / N) * sum(1/r for r=1..min(N, 25))
    random_mrr = np.mean([
        sum(1.0 / rank for rank in range(1, min(n, 25) + 1)) / n
        for n in candidate_counts
    ])

    return {
        "mrr@25": random_mrr,
        "top1": np.mean(1.0 / candidate_counts),
        "top5": np.mean(
            np.minimum(5, candidate_counts) / candidate_counts
        ),
        "top10": np.mean(
            np.minimum(10, candidate_counts) / candidate_counts
        ),
        "top25": np.mean(
            np.minimum(25, candidate_counts) / candidate_counts
        ),
    }


def main() -> None:
    df = pd.read_parquet(INPUT_PATH)

    # One row per query containing the truth's MIST rank.
    truth = df[df["is_truth"]].copy()

    # Get candidate-set size directly from the actual ranked table.
    candidate_counts = (
        df.groupby("query_inchikey14")
        .size()
        .rename("candidate_count")
    )

    truth = truth.merge(
        candidate_counts,
        left_on="query_inchikey14",
        right_index=True,
        validate="one_to_one",
    )

    bins = [1, 5, 10, 25, 50, 100]
    labels = [
        "2-5",
        "6-10",
        "11-25",
        "26-50",
        "51-100",
    ]

    truth["candidate_bin"] = pd.cut(
        truth["candidate_count"],
        bins=bins,
        labels=labels,
        include_lowest=True,
    )

    print("Overall")
    print("=======")

    observed = metrics(truth["mist_rank"].to_numpy())
    random = expected_random_metrics(
        truth["candidate_count"].to_numpy()
    )

    print(f"Queries:          {observed['queries']}")
    print(f"MIST MRR@25:      {observed['mrr@25']:.4f}")
    print(f"Random MRR@25:    {random['mrr@25']:.4f}")
    print(f"MIST Top-1:       {observed['top1']:.2%}")
    print(f"Random Top-1:     {random['top1']:.2%}")
    print(f"MIST Top-5:       {observed['top5']:.2%}")
    print(f"Random Top-5:     {random['top5']:.2%}")
    print(f"MIST Top-10:      {observed['top10']:.2%}")
    print(f"Random Top-10:    {random['top10']:.2%}")
    print(f"MIST Top-25:      {observed['top25']:.2%}")
    print(f"Random Top-25:    {random['top25']:.2%}")
    print(f"Median truth rank:{observed['median_rank']:.1f}")

    print()
    print("By candidate-set size")
    print("=====================")

    rows = []

    for label in labels:
        group = truth[truth["candidate_bin"] == label]

        if group.empty:
            continue

        observed = metrics(group["mist_rank"].to_numpy())
        random = expected_random_metrics(
            group["candidate_count"].to_numpy()
        )

        rows.append({
            "candidate_bin": label,
            "queries": observed["queries"],
            "median_candidates": group["candidate_count"].median(),
            "mist_mrr25": observed["mrr@25"],
            "random_mrr25": random["mrr@25"],
            "mist_top1": observed["top1"],
            "random_top1": random["top1"],
            "mist_top5": observed["top5"],
            "mist_top25": observed["top25"],
            "random_top25": random["top25"],
            "median_truth_rank": observed["median_rank"],
        })

    result = pd.DataFrame(rows)

    with pd.option_context(
        "display.max_columns", None,
        "display.width", 200,
    ):
        print(result.to_string(
            index=False,
            formatters={
                "mist_mrr25": "{:.4f}".format,
                "random_mrr25": "{:.4f}".format,
                "mist_top1": "{:.2%}".format,
                "random_top1": "{:.2%}".format,
                "mist_top5": "{:.2%}".format,
                "mist_top25": "{:.2%}".format,
                "random_top25": "{:.2%}".format,
                "median_candidates": "{:.1f}".format,
                "median_truth_rank": "{:.1f}".format,
            },
        ))

    print()
    print("Queries outside Top 25")
    print("======================")

    misses = truth[truth["mist_rank"] > 25].sort_values(
        "mist_rank",
        ascending=False,
    )

    if misses.empty:
        print("None")
    else:
        print(
            misses[
                [
                    "query_inchikey14",
                    "truth_formula",
                    "candidate_count",
                    "mist_rank",
                    "mist_cosine",
                ]
            ].to_string(index=False)
        )


if __name__ == "__main__":
    main()