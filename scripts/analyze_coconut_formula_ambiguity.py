from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


FEATURES_PATH = Path(
    "data/processed/results/"
    "coconut_ranking_features.parquet"
)


def main() -> None:
    frame = pd.read_parquet(
        FEATURES_PATH
    )

    truth_queries = set(
        frame.loc[
            frame["is_truth"],
            "query_inchikey14",
        ].astype(str)
    )

    frame = frame[
        frame["query_inchikey14"]
        .astype(str)
        .isin(truth_queries)
    ].copy()

    ambiguity: list[int] = []
    coconut_only_ambiguity: list[int] = []
    training_ambiguity: list[int] = []

    rows: list[dict[str, object]] = []

    for query_key, group in frame.groupby(
        "query_inchikey14",
        sort=False,
    ):
        truth = group[
            group["is_truth"]
        ]

        if len(truth) != 1:
            raise RuntimeError(
                f"{query_key}: expected one truth, "
                f"found {len(truth)}"
            )

        truth_row = truth.iloc[0]
        truth_formula = str(
            truth_row["candidate_formula"]
        )

        same_formula = group[
            group["candidate_formula"]
            == truth_formula
        ]

        count = len(same_formula)

        training_count = int(
            same_formula["in_training"].sum()
        )

        coconut_count = int(
            same_formula["in_coconut"].sum()
        )

        coconut_only_count = int(
            (
                same_formula["in_coconut"]
                & ~same_formula["in_training"]
            ).sum()
        )

        ambiguity.append(count)
        training_ambiguity.append(
            training_count
        )
        coconut_only_ambiguity.append(
            coconut_only_count
        )

        rows.append(
            {
                "query_inchikey14":
                    str(query_key),
                "truth_formula":
                    truth_formula,
                "same_formula_candidates":
                    count,
                "same_formula_training":
                    training_count,
                "same_formula_coconut":
                    coconut_count,
                "same_formula_coconut_only":
                    coconut_only_count,
            }
        )

    result = pd.DataFrame(rows)

    counts = np.asarray(
        ambiguity,
        dtype=np.int64,
    )

    print("Truth-formula structural ambiguity")
    print("=" * 70)

    print(
        f"Queries:                     "
        f"{len(counts):,}"
    )

    print()
    print("Same-formula candidates")
    print("-" * 70)

    print(
        f"median={np.median(counts):.0f} "
        f"p75={np.percentile(counts, 75):.0f} "
        f"p90={np.percentile(counts, 90):.0f} "
        f"p95={np.percentile(counts, 95):.0f} "
        f"p99={np.percentile(counts, 99):.0f} "
        f"max={counts.max():,}"
    )

    print()
    print("Competition thresholds")
    print("-" * 70)

    for threshold in [
        1,
        5,
        10,
        25,
        50,
        100,
        250,
        500,
    ]:
        n = int(
            np.sum(counts <= threshold)
        )

        print(
            f"<= {threshold:3d}: "
            f"{n:5,d} / {len(counts):,} "
            f"({n / len(counts):.3%})"
        )

    print()
    print("Candidate source composition")
    print("-" * 70)

    training = np.asarray(
        training_ambiguity,
        dtype=np.int64,
    )

    coconut_only = np.asarray(
        coconut_only_ambiguity,
        dtype=np.int64,
    )

    print(
        "Training candidates:    "
        f"median={np.median(training):.0f} "
        f"p90={np.percentile(training, 90):.0f} "
        f"p99={np.percentile(training, 99):.0f}"
    )

    print(
        "COCONUT-only candidates:"
        f" median={np.median(coconut_only):.0f} "
        f"p90={np.percentile(coconut_only, 90):.0f} "
        f"p99={np.percentile(coconut_only, 99):.0f}"
    )

    print()
    print("Hard cases")
    print("-" * 70)

    print(
        f">25 same-formula candidates: "
        f"{int(np.sum(counts > 25)):,}"
    )

    print(
        f">100 same-formula candidates:"
        f" {int(np.sum(counts > 100)):,}"
    )

    print(
        f">250 same-formula candidates:"
        f" {int(np.sum(counts > 250)):,}"
    )


if __name__ == "__main__":
    main()