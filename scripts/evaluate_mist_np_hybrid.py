from pathlib import Path

import numpy as np
import pandas as pd


INPUT = Path(
    "data/processed/mist/benchmark_100/mist_ranked_candidates.parquet"
)


def evaluate(df: pd.DataFrame, score: str) -> dict:
    ranked = df.sort_values(
        ["query_inchikey14", score, "candidate_inchikey14"],
        ascending=[True, False, True],
        kind="stable",
    ).copy()

    ranked["rank"] = (
        ranked.groupby("query_inchikey14").cumcount() + 1
    )

    truth = ranked[ranked["is_truth"]]
    ranks = truth["rank"].to_numpy()

    return {
        "MRR@25": np.mean(
            np.where(ranks <= 25, 1.0 / ranks, 0.0)
        ),
        "Top1": np.mean(ranks <= 1),
        "Top5": np.mean(ranks <= 5),
        "Top10": np.mean(ranks <= 10),
        "Top25": np.mean(ranks <= 25),
        "MedianRank": np.median(ranks),
    }


def percentile_score(series: pd.Series) -> pd.Series:
    """
    Convert a candidate-level score to a within-query percentile.

    Higher is always better.
    """
    return series.rank(
        method="average",
        pct=True,
        na_option="bottom",
    )


def main():
    df = pd.read_parquet(INPUT).copy()

    # Normalize both signals within each candidate set.
    df["mist_pct"] = (
        df.groupby("query_inchikey14")["mist_cosine"]
        .transform(percentile_score)
    )

    df["np_pct"] = (
        df.groupby("query_inchikey14")["np_likeness"]
        .transform(percentile_score)
    )

    # Predeclared small set of experiments.
    df["mist_only"] = df["mist_pct"]

    df["mist_np_90_10"] = (
        0.90 * df["mist_pct"]
        + 0.10 * df["np_pct"]
    )

    df["mist_np_75_25"] = (
        0.75 * df["mist_pct"]
        + 0.25 * df["np_pct"]
    )

    df["mist_np_50_50"] = (
        0.50 * df["mist_pct"]
        + 0.50 * df["np_pct"]
    )

    scores = [
        "mist_only",
        "mist_np_90_10",
        "mist_np_75_25",
        "mist_np_50_50",
    ]

    rows = []

    for score in scores:
        result = evaluate(df, score)
        result["Model"] = score
        rows.append(result)

    results = pd.DataFrame(rows)[
        [
            "Model",
            "MRR@25",
            "Top1",
            "Top5",
            "Top10",
            "Top25",
            "MedianRank",
        ]
    ]

    print(
        results.to_string(
            index=False,
            formatters={
                "MRR@25": "{:.4f}".format,
                "Top1": "{:.2%}".format,
                "Top5": "{:.2%}".format,
                "Top10": "{:.2%}".format,
                "Top25": "{:.2%}".format,
                "MedianRank": "{:.1f}".format,
            },
        )
    )


if __name__ == "__main__":
    main()