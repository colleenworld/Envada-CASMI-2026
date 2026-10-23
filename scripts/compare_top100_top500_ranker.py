from __future__ import annotations

from pathlib import Path

import pandas as pd


BASE = Path(
    "data/processed/diagnostics/large_validation_audit/"
    "true_formula_neighbor_benchmark"
)

TOP100 = (
    BASE
    / "learned_ranker/holdout_predictions.parquet"
)

TOP500 = (
    BASE
    / "learned_ranker_top500/holdout_predictions.parquet"
)


def rr25(rank) -> float:
    if rank is None or pd.isna(rank):
        return 0.0

    rank = int(rank)

    if rank < 1 or rank > 25:
        return 0.0

    return 1.0 / rank


def report(
    label: str,
    frame: pd.DataFrame,
    rank_column: str,
) -> None:
    ranks = frame[
        rank_column
    ]

    print()
    print(label)
    print("-" * len(label))
    print(
        "Queries:",
        f"{len(frame):,}",
    )
    print(
        "MRR@25:",
        f"{ranks.map(rr25).mean():.4f}",
    )
    print(
        "Top-1:",
        f"{ranks.eq(1).mean():.2%}",
    )
    print(
        "Top-5:",
        f"{ranks.between(1,5).mean():.2%}",
    )
    print(
        "Top-10:",
        f"{ranks.between(1,10).mean():.2%}",
    )
    print(
        "Top-25:",
        f"{ranks.between(1,25).mean():.2%}",
    )
    print(
        "Top-100:",
        f"{ranks.between(1,100).mean():.2%}",
    )
    print(
        "Truth returned:",
        f"{ranks.notna().mean():.2%}",
    )


def main() -> None:
    for path in [
        TOP100,
        TOP500,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"Missing required artifact: {path}"
            )

    top100 = pd.read_parquet(
        TOP100
    ).rename(
        columns={
            "truth_present":
                "truth_present_top100",
            "learned_rank":
                "learned_rank_top100",
            "baseline_rank":
                "baseline_rank_top100",
        }
    )

    top500 = pd.read_parquet(
        TOP500
    ).rename(
        columns={
            "truth_present":
                "truth_present_top500",
            "learned_rank":
                "learned_rank_top500",
            "baseline_rank":
                "baseline_rank_top500",
        }
    )

    merged = top100.merge(
        top500,
        on="inchikey14",
        how="outer",
        validate="one_to_one",
    )

    if len(merged) != 420:
        raise RuntimeError(
            f"Expected 420 holdout queries, found {len(merged)}"
        )

    common = merged.loc[
        merged[
            "truth_present_top100"
        ].astype(bool)
        & merged[
            "truth_present_top500"
        ].astype(bool)
    ].copy()

    newly_recovered = merged.loc[
        ~merged[
            "truth_present_top100"
        ].astype(bool)
        & merged[
            "truth_present_top500"
        ].astype(bool)
    ].copy()

    still_missing = merged.loc[
        ~merged[
            "truth_present_top500"
        ].astype(bool)
    ].copy()

    print()
    print(
        "Top-100 vs top-500 learned-ranker comparison"
    )
    print(
        "=" * 76
    )
    print(
        "Common retained truths:",
        f"{len(common):,}",
    )
    print(
        "Newly recovered by top-500:",
        f"{len(newly_recovered):,}",
    )
    print(
        "Still missing at top-500:",
        f"{len(still_missing):,}",
    )

    report(
        "Top-100 LightGBM on common retained truths",
        common,
        "learned_rank_top100",
    )

    report(
        "Top-500 LightGBM on same common truths",
        common,
        "learned_rank_top500",
    )

    report(
        "Top-500 LightGBM on newly recovered truths",
        newly_recovered,
        "learned_rank_top500",
    )

    if not common.empty:
        delta = (
            common[
                "learned_rank_top500"
            ]
            - common[
                "learned_rank_top100"
            ]
        )

        comparable = delta.dropna()

        print()
        print(
            "Rank movement on common truths"
        )
        print(
            "-" * 76
        )
        print(
            "Improved:",
            f"{int((comparable < 0).sum()):,}",
        )
        print(
            "Unchanged:",
            f"{int((comparable == 0).sum()):,}",
        )
        print(
            "Worsened:",
            f"{int((comparable > 0).sum()):,}",
        )
        print(
            "Median rank delta:",
            f"{comparable.median():.1f}",
        )

    print()
    print(
        "Newly recovered truth ranks"
    )
    print(
        "-" * 76
    )

    if newly_recovered.empty:
        print(
            "No newly recovered truths."
        )
    else:
        print(
            newly_recovered[
                "learned_rank_top500"
            ]
            .describe(
                percentiles=[
                    0.25,
                    0.5,
                    0.75,
                    0.9,
                    0.95,
                ]
            )
            .to_string()
        )


if __name__ == "__main__":
    main()
