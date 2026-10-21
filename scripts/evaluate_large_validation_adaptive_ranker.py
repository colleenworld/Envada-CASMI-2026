from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent

COMPARISON_PATH = (
    PROJECT_ROOT
    / "data/processed/diagnostics/large_validation_audit/"
    "true_formula_neighbor_benchmark/mist_full/comparison.parquet"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data/processed/diagnostics/large_validation_audit/"
    "true_formula_neighbor_benchmark/mist_full/adaptive_policy.parquet"
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
    ranks: pd.Series,
) -> None:
    print()
    print(label)
    print("-" * len(label))
    print(
        "MRR@25:",
        f"{ranks.map(rr25).mean():.4f}",
    )
    print(
        "Top-1:",
        f"{(ranks == 1).mean():.2%}",
    )
    print(
        "Top-5:",
        f"{ranks.between(1, 5).mean():.2%}",
    )
    print(
        "Top-10:",
        f"{ranks.between(1, 10).mean():.2%}",
    )
    print(
        "Top-25:",
        f"{ranks.between(1, 25).mean():.2%}",
    )
    print(
        "Top-100:",
        f"{ranks.between(1, 100).mean():.2%}",
    )
    print(
        "Truth returned:",
        f"{ranks.notna().mean():.2%}",
    )


def better_rank(
    mist_rank,
    neighbor_rank,
):
    mist_missing = pd.isna(mist_rank)
    neighbor_missing = pd.isna(neighbor_rank)

    if mist_missing and neighbor_missing:
        return np.nan

    if mist_missing:
        return neighbor_rank

    if neighbor_missing:
        return mist_rank

    return min(
        int(mist_rank),
        int(neighbor_rank),
    )


def main() -> None:
    frame = pd.read_parquet(
        COMPARISON_PATH
    )

    required = {
        "inchikey14",
        "candidate_count",
        "mist_truth_rank",
        "neighbor_truth_rank",
    }

    missing = (
        required
        - set(frame.columns)
    )

    if missing:
        raise ValueError(
            "Comparison artifact is missing columns: "
            f"{sorted(missing)}"
        )

    #
    # Fixed policy motivated by the pre-existing burden buckets:
    # MIST for <=100 candidates, neighbor prior otherwise.
    #
    use_mist = (
        frame[
            "candidate_count"
        ]
        <= 100
    )

    frame[
        "adaptive_method"
    ] = np.where(
        use_mist,
        "mist",
        "neighbor",
    )

    frame[
        "adaptive_truth_rank"
    ] = np.where(
        use_mist,
        frame[
            "mist_truth_rank"
        ],
        frame[
            "neighbor_truth_rank"
        ],
    )

    #
    # Diagnostic only: optimistic upper bound if we somehow knew which
    # of the two methods ranked the truth better for each query.
    #
    frame[
        "oracle_truth_rank"
    ] = [
        better_rank(
            mist_rank,
            neighbor_rank,
        )
        for mist_rank, neighbor_rank
        in zip(
            frame[
                "mist_truth_rank"
            ],
            frame[
                "neighbor_truth_rank"
            ],
            strict=True,
        )
    ]

    frame.to_parquet(
        OUTPUT_PATH,
        index=False,
    )

    print(
        "Adaptive structure-ranking policy"
    )
    print(
        "================================="
    )
    print(
        "Structures:",
        f"{len(frame):,}",
    )
    print(
        "Policy: MIST when candidate_count <= 100; "
        "neighbor prior otherwise"
    )
    print(
        "MIST-selected structures:",
        f"{int(use_mist.sum()):,}",
    )
    print(
        "Neighbor-selected structures:",
        f"{int((~use_mist).sum()):,}",
    )

    report(
        "MIST only",
        frame[
            "mist_truth_rank"
        ],
    )

    report(
        "Neighbor prior only",
        frame[
            "neighbor_truth_rank"
        ],
    )

    report(
        "Fixed adaptive policy",
        frame[
            "adaptive_truth_rank"
        ],
    )

    report(
        "Best-of-two oracle ceiling",
        frame[
            "oracle_truth_rank"
        ],
    )

    #
    # Diagnostic threshold sweep. This is not a production tuning
    # recommendation; it shows whether the fixed <=100 crossover is
    # broadly sensible.
    #
    print()
    print(
        "Diagnostic threshold sweep"
    )
    print(
        "--------------------------"
    )

    rows = []

    for threshold in [
        25,
        50,
        100,
        250,
        500,
        1000,
        2500,
        5000,
    ]:
        select_mist = (
            frame[
                "candidate_count"
            ]
            <= threshold
        )

        ranks = pd.Series(
            np.where(
                select_mist,
                frame[
                    "mist_truth_rank"
                ],
                frame[
                    "neighbor_truth_rank"
                ],
            ),
            index=frame.index,
        )

        rows.append(
            {
                "threshold":
                    threshold,
                "mist_selected":
                    int(
                        select_mist.sum()
                    ),
                "mrr25":
                    ranks.map(
                        rr25
                    ).mean(),
                "top1":
                    (
                        ranks
                        == 1
                    ).mean(),
                "top25":
                    ranks.between(
                        1,
                        25,
                    ).mean(),
                "top100":
                    ranks.between(
                        1,
                        100,
                    ).mean(),
            }
        )

    sweep = pd.DataFrame(
        rows
    )

    print(
        sweep.to_string(
            index=False,
            formatters={
                "mrr25":
                    lambda value:
                    f"{value:.4f}",
                "top1":
                    lambda value:
                    f"{value:.2%}",
                "top25":
                    lambda value:
                    f"{value:.2%}",
                "top100":
                    lambda value:
                    f"{value:.2%}",
            },
        )
    )

    print()
    print(
        f"Wrote {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()
