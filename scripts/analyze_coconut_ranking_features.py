from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


FEATURES_PATH = Path(
    "data/processed/results/"
    "coconut_ranking_features.parquet"
)


def reciprocal_rank(
    ranked: pd.DataFrame,
) -> float:
    truth_positions = np.flatnonzero(
        ranked["is_truth"].to_numpy(
            dtype=bool
        )
    )

    if len(truth_positions) == 0:
        return 0.0

    rank = int(truth_positions[0]) + 1

    if rank > 25:
        return 0.0

    return 1.0 / rank


def evaluate_ranking(
    frame: pd.DataFrame,
    *,
    feature: str,
    ascending: bool,
) -> dict[str, float]:
    """
    Evaluate one feature as a ranking score.

    Ties deliberately use candidate_inchikey14
    as a deterministic secondary key. We will
    separately measure tie frequency so that we
    don't mistake arbitrary tie breaking for
    ranking signal.
    """

    reciprocal_ranks: list[float] = []
    truth_ranks: list[int] = []

    top1 = 0
    top5 = 0
    top10 = 0
    top25 = 0

    queries = 0
    truth_queries = 0

    for _, group in frame.groupby(
        "query_inchikey14",
        sort=False,
    ):
        queries += 1

        ranked = group.sort_values(
            [
                feature,
                "candidate_inchikey14",
            ],
            ascending=[
                ascending,
                True,
            ],
            na_position="last",
        )

        truth_positions = np.flatnonzero(
            ranked["is_truth"].to_numpy(
                dtype=bool
            )
        )

        if len(truth_positions) == 0:
            reciprocal_ranks.append(0.0)
            continue

        truth_queries += 1

        rank = int(
            truth_positions[0]
        ) + 1

        truth_ranks.append(rank)

        if rank == 1:
            top1 += 1

        if rank <= 5:
            top5 += 1

        if rank <= 10:
            top10 += 1

        if rank <= 25:
            top25 += 1
            reciprocal_ranks.append(
                1.0 / rank
            )
        else:
            reciprocal_ranks.append(
                0.0
            )

    return {
        "queries": queries,
        "truth_queries": truth_queries,
        "mrr25": float(
            np.mean(reciprocal_ranks)
        ),
        "top1": top1 / queries,
        "top5": top5 / queries,
        "top10": top10 / queries,
        "top25": top25 / queries,
        "truth_rank_median": (
            float(np.median(truth_ranks))
            if truth_ranks
            else np.nan
        ),
        "truth_rank_p90": (
            float(
                np.percentile(
                    truth_ranks,
                    90,
                )
            )
            if truth_ranks
            else np.nan
        ),
    }


def evaluate_truth_available(
    frame: pd.DataFrame,
    *,
    feature: str,
    ascending: bool,
) -> dict[str, float]:
    """
    Same ranking evaluation, but only for queries
    whose truth survived candidate generation.

    This isolates ranking quality from candidate
    generation recall.
    """

    truth_queries = set(
        frame.loc[
            frame["is_truth"],
            "query_inchikey14",
        ].astype(str)
    )

    subset = frame[
        frame["query_inchikey14"]
        .astype(str)
        .isin(truth_queries)
    ]

    return evaluate_ranking(
        subset,
        feature=feature,
        ascending=ascending,
    )


def truth_tie_analysis(
    frame: pd.DataFrame,
    *,
    feature: str,
) -> dict[str, float]:
    """
    Determine how often the truth is tied with
    another candidate on this feature.
    """

    tied_truths = 0
    unique_truths = 0
    truth_queries = 0
    tie_sizes: list[int] = []

    for _, group in frame.groupby(
        "query_inchikey14",
        sort=False,
    ):
        truth = group[
            group["is_truth"]
        ]

        if truth.empty:
            continue

        truth_queries += 1

        truth_value = truth.iloc[0][
            feature
        ]

        if pd.isna(truth_value):
            continue

        values = group[feature]

        if np.issubdtype(
            values.dtype,
            np.number,
        ):
            tied = group[
                np.isclose(
                    values.astype(float),
                    float(truth_value),
                    rtol=1e-12,
                    atol=1e-12,
                    equal_nan=False,
                )
            ]
        else:
            tied = group[
                values == truth_value
            ]

        tie_size = len(tied)

        tie_sizes.append(tie_size)

        if tie_size > 1:
            tied_truths += 1
        else:
            unique_truths += 1

    return {
        "truth_queries": truth_queries,
        "tied_truths": tied_truths,
        "unique_truths": unique_truths,
        "tie_rate": (
            tied_truths / truth_queries
            if truth_queries
            else 0.0
        ),
        "median_tie_size": (
            float(np.median(tie_sizes))
            if tie_sizes
            else np.nan
        ),
        "p90_tie_size": (
            float(
                np.percentile(
                    tie_sizes,
                    90,
                )
            )
            if tie_sizes
            else np.nan
        ),
    }


def print_result(
    name: str,
    result: dict[str, float],
) -> None:
    print(name)
    print("-" * 70)

    print(
        f"Queries:           "
        f"{int(result['queries']):,}"
    )

    print(
        f"Truth available:   "
        f"{int(result['truth_queries']):,}"
    )

    print(
        f"MRR@25:            "
        f"{result['mrr25']:.4f}"
    )

    print(
        f"Top-1:             "
        f"{result['top1']:.3%}"
    )

    print(
        f"Top-5:             "
        f"{result['top5']:.3%}"
    )

    print(
        f"Top-10:            "
        f"{result['top10']:.3%}"
    )

    print(
        f"Top-25:            "
        f"{result['top25']:.3%}"
    )

    print(
        f"Truth rank median: "
        f"{result['truth_rank_median']:.0f}"
    )

    print(
        f"Truth rank p90:    "
        f"{result['truth_rank_p90']:.0f}"
    )

    print()


def main() -> None:
    frame = pd.read_parquet(
        FEATURES_PATH
    )

    print("COCONUT ranking feature analysis")
    print("=" * 70)

    print(
        f"Rows:              "
        f"{len(frame):,}"
    )

    print(
        f"Queries:           "
        f"{frame['query_inchikey14'].nunique():,}"
    )

    print(
        f"Truth rows:        "
        f"{int(frame['is_truth'].sum()):,}"
    )

    print()

    #
    # Direction matters:
    #
    # support       -> higher is better
    # mass error    -> lower is better
    # np_likeness   -> higher is tentatively better
    #

    features = [
        (
            "Formula support count",
            "formula_support_count",
            False,
        ),
        (
            "Formula support fraction",
            "formula_support_fraction",
            False,
        ),
        (
            "Best absolute mass error",
            "mass_error_best_abs_ppm",
            True,
        ),
        (
            "Median absolute mass error",
            "mass_error_median_abs_ppm",
            True,
        ),
        (
            "Mean absolute mass error",
            "mass_error_mean_abs_ppm",
            True,
        ),
        (
            "NP likeness",
            "np_likeness",
            False,
        ),
    ]

    print("End-to-end ranking")
    print("=" * 70)
    print(
        "Includes queries where candidate generation "
        "missed the truth."
    )
    print()

    for name, feature, ascending in features:
        result = evaluate_ranking(
            frame,
            feature=feature,
            ascending=ascending,
        )

        print_result(
            name,
            result,
        )

    print()
    print("Ranking conditional on truth availability")
    print("=" * 70)
    print(
        "Only queries where candidate generation "
        "contains the truth."
    )
    print()

    for name, feature, ascending in features:
        result = evaluate_truth_available(
            frame,
            feature=feature,
            ascending=ascending,
        )

        print_result(
            name,
            result,
        )

    #
    # Tie analysis.
    #
    # Formula-level features are identical for every
    # structure sharing a formula, so this section is
    # especially important.
    #

    print()
    print("Truth tie analysis")
    print("=" * 70)

    for name, feature, _ in features:
        result = truth_tie_analysis(
            frame,
            feature=feature,
        )

        print(name)
        print("-" * 70)

        print(
            f"Truth queries:     "
            f"{int(result['truth_queries']):,}"
        )

        print(
            f"Truth tied:        "
            f"{int(result['tied_truths']):,}"
        )

        print(
            f"Tie rate:          "
            f"{result['tie_rate']:.3%}"
        )

        print(
            f"Median tie size:   "
            f"{result['median_tie_size']:.0f}"
        )

        print(
            f"P90 tie size:      "
            f"{result['p90_tie_size']:.0f}"
        )

        print()

    #
    # Truth vs non-truth feature distributions.
    #

    print()
    print("Feature distributions")
    print("=" * 70)

    for name, feature, _ in features:
        truth = frame.loc[
            frame["is_truth"],
            feature,
        ].dropna()

        false = frame.loc[
            ~frame["is_truth"],
            feature,
        ].dropna()

        print(name)
        print("-" * 70)

        if len(truth):
            print(
                "Truth:     "
                f"median={truth.median():.4f} "
                f"p10={truth.quantile(0.10):.4f} "
                f"p90={truth.quantile(0.90):.4f}"
            )
        else:
            print("Truth:     no values")

        if len(false):
            print(
                "Non-truth: "
                f"median={false.median():.4f} "
                f"p10={false.quantile(0.10):.4f} "
                f"p90={false.quantile(0.90):.4f}"
            )
        else:
            print("Non-truth: no values")

        print()


if __name__ == "__main__":
    main()