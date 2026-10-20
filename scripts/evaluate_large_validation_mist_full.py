from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent

BENCHMARK_DIR = (
    PROJECT_ROOT
    / "data/processed/diagnostics/large_validation_audit/"
    "true_formula_neighbor_benchmark"
)

MIST_DIR = BENCHMARK_DIR / "mist_full"
SAMPLE_PATH = MIST_DIR / "sample.parquet"
NEIGHBOR_PATH = (
    BENCHMARK_DIR
    / "neighbor_prior_truth_ranks.parquet"
)
PREDICTION_DIR = MIST_DIR / "predictions"
OUTPUT_PATH = MIST_DIR / "comparison.parquet"

RRF_K = 60.0


def to_str(value) -> str:
    if isinstance(value, bytes):
        return value.decode()
    return str(value)


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
    ranks = frame[rank_column]

    print()
    print(label)
    print("-" * len(label))
    print(
        "Structures:",
        f"{len(frame):,}",
    )
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


def main() -> None:
    sample = pd.read_parquet(
        SAMPLE_PATH
    )

    neighbor = pd.read_parquet(
        NEIGHBOR_PATH
    )[
        [
            "inchikey14",
            "truth_rank",
        ]
    ].rename(
        columns={
            "truth_rank":
                "neighbor_truth_rank",
        }
    )

    predictions = sorted(
        PREDICTION_DIR.glob(
            "retrieval_contrastive_*.p"
        )
    )

    if len(predictions) != 1:
        raise RuntimeError(
            f"Expected one prediction artifact, found {predictions}"
        )

    with predictions[0].open("rb") as handle:
        retrieval = pickle.load(
            handle
        )

    sample_lookup = sample.set_index(
        "spec"
    )

    rows = []

    for spec, ikeys, dists in zip(
        retrieval["names"],
        retrieval["ikeys"],
        retrieval["dists"],
        strict=True,
    ):
        spec = to_str(spec)

        meta = sample_lookup.loc[
            spec
        ]

        ordered = sorted(
            [
                (
                    to_str(ikey),
                    float(distance),
                )
                for ikey, distance
                in zip(
                    ikeys,
                    dists,
                    strict=True,
                )
            ],
            key=lambda item: (
                item[1],
                item[0],
            ),
        )

        truth_key = str(
            meta.inchikey14
        )

        mist_truth_rank = np.nan

        for rank, (
            candidate_key,
            _,
        ) in enumerate(
            ordered,
            start=1,
        ):
            if candidate_key == truth_key:
                mist_truth_rank = rank
                break

        rows.append(
            {
                "inchikey14":
                    truth_key,
                "spec":
                    spec,
                "truth_formula":
                    str(
                        meta.truth_formula
                    ),
                "burden_bucket":
                    str(
                        meta.burden_bucket
                    ),
                "dominant_library":
                    str(
                        meta.dominant_library
                    ),
                "candidate_count":
                    int(
                        meta.truth_formula_pubchem_candidate_count
                    ),
                "mist_truth_rank":
                    mist_truth_rank,
            }
        )

    result = (
        pd.DataFrame(
            rows
        )
        .merge(
            neighbor,
            on="inchikey14",
            how="left",
            validate="one_to_one",
        )
    )

    result.to_parquet(
        OUTPUT_PATH,
        index=False,
    )

    print(
        "Large true-formula structure ranking"
    )
    print(
        "===================================="
    )

    report(
        "MIST full candidate set",
        result,
        "mist_truth_rank",
    )

    report(
        "Neighbor prior, same molecules",
        result,
        "neighbor_truth_rank",
    )

    print()
    print(
        "By burden bucket"
    )
    print(
        "----------------"
    )

    for bucket in [
        "le100",
        "101to500",
        "501to1000",
        "1001to5000",
        "gt5000",
    ]:
        frame = result.loc[
            result[
                "burden_bucket"
            ]
            == bucket
        ]

        if frame.empty:
            continue

        print()
        print(
            bucket,
            f"(n={len(frame):,})",
        )

        for label, column in [
            (
                "MIST",
                "mist_truth_rank",
            ),
            (
                "Neighbor",
                "neighbor_truth_rank",
            ),
        ]:
            ranks = frame[
                column
            ]

            print(
                f"  {label:8s} "
                f"MRR@25={ranks.map(rr25).mean():.4f} "
                f"Top1={(ranks == 1).mean():.2%} "
                f"Top25={ranks.between(1,25).mean():.2%} "
                f"Top100={ranks.between(1,100).mean():.2%}"
            )

    print()
    print(
        f"Wrote {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()
