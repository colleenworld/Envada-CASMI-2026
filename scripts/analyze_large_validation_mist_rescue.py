from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent

COMPARISON_PATH = (
    PROJECT_ROOT
    / "data/processed/diagnostics/large_validation_audit/"
    "true_formula_neighbor_benchmark/mist_full/comparison.parquet"
)


def pct(value: float) -> str:
    return f"{value:.2%}"


def main() -> None:
    frame = pd.read_parquet(
        COMPARISON_PATH
    )

    n = len(frame)

    neighbor = frame[
        "neighbor_truth_rank"
    ]
    mist = frame[
        "mist_truth_rank"
    ]

    print(
        "MIST rescue analysis"
    )
    print(
        "===================="
    )
    print(
        "Structures:",
        f"{n:,}",
    )

    #
    # Where MIST can rescue neighbor misses.
    #
    neighbor_miss25 = ~neighbor.between(
        1,
        25,
    )

    print()
    print(
        "Among neighbor Top-25 misses"
    )
    print(
        "----------------------------"
    )
    print(
        "Queries:",
        f"{int(neighbor_miss25.sum()):,}",
    )

    for cutoff in [
        1,
        5,
        10,
        25,
        100,
        500,
    ]:
        rescued = (
            neighbor_miss25
            & mist.between(
                1,
                cutoff,
            )
        )

        print(
            f"MIST Top-{cutoff:<3}: "
            f"{int(rescued.sum()):,} "
            f"({pct(rescued.sum() / neighbor_miss25.sum())})"
        )

    #
    # Where MIST is worse than already-good neighbor positions.
    #
    print()
    print(
        "When neighbor is already strong"
    )
    print(
        "-------------------------------"
    )

    for neighbor_cutoff in [
        1,
        5,
        10,
        25,
    ]:
        good = neighbor.between(
            1,
            neighbor_cutoff,
        )

        mist_worse25 = (
            good
            & ~mist.between(
                1,
                25,
            )
        )

        mist_worse100 = (
            good
            & ~mist.between(
                1,
                100,
            )
        )

        print(
            f"Neighbor Top-{neighbor_cutoff:<2}: "
            f"n={int(good.sum()):,}; "
            f"MIST outside Top-25="
            f"{int(mist_worse25.sum()):,} "
            f"({pct(mist_worse25.sum() / good.sum())}); "
            f"outside Top-100="
            f"{int(mist_worse100.sum()):,} "
            f"({pct(mist_worse100.sum() / good.sum())})"
        )

    #
    # Direct pairwise truth-rank comparison.
    #
    both = neighbor.notna() & mist.notna()

    mist_better = (
        both
        & (
            mist
            < neighbor
        )
    )

    neighbor_better = (
        both
        & (
            neighbor
            < mist
        )
    )

    tied = (
        both
        & (
            neighbor
            == mist
        )
    )

    print()
    print(
        "Direct truth-rank comparison"
    )
    print(
        "----------------------------"
    )
    print(
        "Both returned:",
        f"{int(both.sum()):,}",
    )
    print(
        "MIST better:",
        f"{int(mist_better.sum()):,} "
        f"({pct(mist_better.sum() / n)})",
    )
    print(
        "Neighbor better:",
        f"{int(neighbor_better.sum()):,} "
        f"({pct(neighbor_better.sum() / n)})",
    )
    print(
        "Tied:",
        f"{int(tied.sum()):,} "
        f"({pct(tied.sum() / n)})",
    )

    #
    # Rescue opportunities by burden bucket.
    #
    rows = []

    for bucket, group in frame.groupby(
        "burden_bucket",
        sort=False,
    ):
        nr = group[
            "neighbor_truth_rank"
        ]
        mr = group[
            "mist_truth_rank"
        ]

        miss25 = ~nr.between(
            1,
            25,
        )

        rescue25 = (
            miss25
            & mr.between(
                1,
                25,
            )
        )

        rescue100 = (
            miss25
            & mr.between(
                1,
                100,
            )
        )

        rows.append(
            {
                "burden_bucket":
                    bucket,
                "structures":
                    len(group),
                "neighbor_top25":
                    nr.between(
                        1,
                        25,
                    ).mean(),
                "neighbor_miss25":
                    int(
                        miss25.sum()
                    ),
                "mist_top25_rescues":
                    int(
                        rescue25.sum()
                    ),
                "mist_top100_rescues":
                    int(
                        rescue100.sum()
                    ),
                "top25_rescue_fraction":
                    (
                        rescue25.sum()
                        / miss25.sum()
                        if miss25.any()
                        else 0.0
                    ),
            }
        )

    report = pd.DataFrame(
        rows
    )

    print()
    print(
        "Rescue opportunity by burden"
    )
    print(
        "----------------------------"
    )
    print(
        report.to_string(
            index=False,
            formatters={
                "neighbor_top25":
                    pct,
                "top25_rescue_fraction":
                    pct,
            },
        )
    )


if __name__ == "__main__":
    main()
