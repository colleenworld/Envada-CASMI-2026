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
NEIGHBOR_PATH = BENCHMARK_DIR / "neighbor_prior_top500.parquet"
PREDICTION_DIR = MIST_DIR / "predictions"
OUTPUT_PATH = MIST_DIR / "rrf_fusion.parquet"

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


def main() -> None:
    sample = pd.read_parquet(
        SAMPLE_PATH
    )

    neighbor = pd.read_parquet(
        NEIGHBOR_PATH
    )

    predictions = sorted(
        PREDICTION_DIR.glob(
            "retrieval_contrastive_*.p"
        )
    )

    if len(predictions) != 1:
        raise RuntimeError(
            f"Expected one MIST prediction artifact, found {predictions}"
        )

    with predictions[0].open("rb") as handle:
        retrieval = pickle.load(
            handle
        )

    sample_lookup = sample.set_index(
        "spec"
    )

    mist_rows = []

    for spec, ikeys, dists in zip(
        retrieval["names"],
        retrieval["ikeys"],
        retrieval["dists"],
        strict=True,
    ):
        spec = to_str(spec)

        if spec not in sample_lookup.index:
            raise ValueError(
                f"Unexpected MIST spec: {spec}"
            )

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

        for rank, (
            candidate_key,
            distance,
        ) in enumerate(
            ordered,
            start=1,
        ):
            mist_rows.append(
                {
                    "inchikey14":
                        str(
                            meta.inchikey14
                        ),
                    "candidate_inchikey14":
                        candidate_key,
                    "mist_rank":
                        rank,
                    "mist_distance":
                        distance,
                }
            )

    mist = pd.DataFrame(
        mist_rows
    )

    sample_keys = set(
        sample[
            "inchikey14"
        ].astype(str)
    )

    neighbor = neighbor.loc[
        neighbor[
            "inchikey14"
        ]
        .astype(str)
        .isin(
            sample_keys
        )
    ].copy()

    fused = neighbor[
        [
            "inchikey14",
            "candidate_inchikey14",
            "prior_rank",
            "neighbor_prior_score",
        ]
    ].merge(
        mist,
        on=[
            "inchikey14",
            "candidate_inchikey14",
        ],
        how="outer",
        validate="one_to_one",
    )

    fused[
        "rrf_score"
    ] = 0.0

    has_neighbor = fused[
        "prior_rank"
    ].notna()

    has_mist = fused[
        "mist_rank"
    ].notna()

    fused.loc[
        has_neighbor,
        "rrf_score",
    ] += (
        1.0
        / (
            RRF_K
            + fused.loc[
                has_neighbor,
                "prior_rank",
            ].astype(float)
        )
    )

    fused.loc[
        has_mist,
        "rrf_score",
    ] += (
        1.0
        / (
            RRF_K
            + fused.loc[
                has_mist,
                "mist_rank",
            ].astype(float)
        )
    )

    fused = (
        fused
        .sort_values(
            [
                "inchikey14",
                "rrf_score",
                "candidate_inchikey14",
            ],
            ascending=[
                True,
                False,
                True,
            ],
            kind="stable",
        )
        .reset_index(drop=True)
    )

    fused[
        "fusion_rank"
    ] = (
        fused.groupby(
            "inchikey14"
        )
        .cumcount()
        + 1
    )

    truth = fused.loc[
        fused[
            "candidate_inchikey14"
        ]
        .astype(str)
        == fused[
            "inchikey14"
        ]
        .astype(str),
        [
            "inchikey14",
            "fusion_rank",
            "mist_rank",
            "prior_rank",
        ],
    ].drop_duplicates(
        subset=[
            "inchikey14"
        ]
    )

    metrics = (
        sample[
            [
                "inchikey14",
                "burden_bucket",
                "dominant_library",
                "truth_formula_pubchem_candidate_count",
            ]
        ]
        .merge(
            truth,
            on="inchikey14",
            how="left",
            validate="one_to_one",
        )
    )

    metrics.to_parquet(
        OUTPUT_PATH,
        index=False,
    )

    print(
        "Large true-formula reciprocal-rank fusion"
    )
    print(
        "========================================="
    )
    print(
        "Structures:",
        f"{len(metrics):,}",
    )
    print(
        "RRF k:",
        f"{RRF_K:g}",
    )

    report(
        "RRF fusion",
        metrics[
            "fusion_rank"
        ],
    )

    report(
        "MIST within fusion union",
        metrics[
            "mist_rank"
        ],
    )

    report(
        "Neighbor within fusion union",
        metrics[
            "prior_rank"
        ],
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
        frame = metrics.loc[
            metrics[
                "burden_bucket"
            ]
            == bucket
        ]

        if frame.empty:
            continue

        ranks = frame[
            "fusion_rank"
        ]

        print(
            f"{bucket:>10} "
            f"n={len(frame):4d} "
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
