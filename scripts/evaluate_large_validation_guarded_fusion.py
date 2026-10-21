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

OUTPUT_PATH = MIST_DIR / "guarded_fusion_summary.csv"

RRF_K = 60.0
GUARD_DEPTHS = [1, 3, 5, 10, 15, 20]


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


def build_mist_rows(
    sample: pd.DataFrame,
) -> pd.DataFrame:
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
        retrieval = pickle.load(handle)

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
        meta = sample_lookup.loc[spec]

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
            rows.append(
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

    return pd.DataFrame(rows)


def rank_guarded(
    candidate_frame: pd.DataFrame,
    guard_depth: int,
) -> pd.DataFrame:
    parts = []

    for query_key, group in candidate_frame.groupby(
        "inchikey14",
        sort=False,
    ):
        group = group.copy()

        protected = (
            group.loc[
                group[
                    "prior_rank"
                ].notna()
                & (
                    group[
                        "prior_rank"
                    ]
                    <= guard_depth
                )
            ]
            .sort_values(
                [
                    "prior_rank",
                    "candidate_inchikey14",
                ],
                ascending=[
                    True,
                    True,
                ],
            )
            .copy()
        )

        protected[
            "guarded_rank"
        ] = np.arange(
            1,
            len(protected) + 1,
            dtype=np.int64,
        )

        protected_keys = set(
            protected[
                "candidate_inchikey14"
            ].astype(str)
        )

        remainder = group.loc[
            ~group[
                "candidate_inchikey14"
            ]
            .astype(str)
            .isin(
                protected_keys
            )
        ].copy()

        remainder[
            "rrf_score"
        ] = 0.0

        has_neighbor = (
            remainder[
                "prior_rank"
            ].notna()
        )
        has_mist = (
            remainder[
                "mist_rank"
            ].notna()
        )

        remainder.loc[
            has_neighbor,
            "rrf_score",
        ] += (
            1.0
            / (
                RRF_K
                + remainder.loc[
                    has_neighbor,
                    "prior_rank",
                ].astype(float)
            )
        )

        remainder.loc[
            has_mist,
            "rrf_score",
        ] += (
            1.0
            / (
                RRF_K
                + remainder.loc[
                    has_mist,
                    "mist_rank",
                ].astype(float)
            )
        )

        remainder = (
            remainder
            .sort_values(
                [
                    "rrf_score",
                    "candidate_inchikey14",
                ],
                ascending=[
                    False,
                    True,
                ],
                kind="stable",
            )
            .reset_index(drop=True)
        )

        remainder[
            "guarded_rank"
        ] = (
            np.arange(
                len(remainder),
                dtype=np.int64,
            )
            + len(protected)
            + 1
        )

        parts.append(
            pd.concat(
                [
                    protected,
                    remainder,
                ],
                ignore_index=True,
            )
        )

    return pd.concat(
        parts,
        ignore_index=True,
    )


def summarize(
    label: str,
    ranks: pd.Series,
) -> dict:
    return {
        "policy":
            label,
        "mrr25":
            ranks.map(
                rr25
            ).mean(),
        "top1":
            (
                ranks
                == 1
            ).mean(),
        "top5":
            ranks.between(
                1,
                5,
            ).mean(),
        "top10":
            ranks.between(
                1,
                10,
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


def main() -> None:
    sample = pd.read_parquet(
        SAMPLE_PATH
    )

    neighbor = pd.read_parquet(
        NEIGHBOR_PATH
    )[
        [
            "inchikey14",
            "candidate_inchikey14",
            "prior_rank",
            "neighbor_prior_score",
        ]
    ].copy()

    sample_keys = set(
        sample[
            "inchikey14"
        ]
        .astype(str)
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

    mist = build_mist_rows(
        sample
    )

    union = neighbor.merge(
        mist,
        on=[
            "inchikey14",
            "candidate_inchikey14",
        ],
        how="outer",
        validate="one_to_one",
    )

    summary_rows = []

    #
    # Baselines from the same candidate union.
    #
    neighbor_truth = (
        union.loc[
            union[
                "candidate_inchikey14"
            ].astype(str)
            == union[
                "inchikey14"
            ].astype(str),
            [
                "inchikey14",
                "prior_rank",
            ],
        ]
        .drop_duplicates(
            subset=[
                "inchikey14"
            ]
        )
    )

    neighbor_metrics = sample[
        [
            "inchikey14",
        ]
    ].merge(
        neighbor_truth,
        on="inchikey14",
        how="left",
        validate="one_to_one",
    )

    summary_rows.append(
        summarize(
            "neighbor",
            neighbor_metrics[
                "prior_rank"
            ],
        )
    )

    #
    # Guarded fusion sweeps.
    #
    for guard_depth in GUARD_DEPTHS:
        ranked = rank_guarded(
            union,
            guard_depth,
        )

        truth = (
            ranked.loc[
                ranked[
                    "candidate_inchikey14"
                ].astype(str)
                == ranked[
                    "inchikey14"
                ].astype(str),
                [
                    "inchikey14",
                    "guarded_rank",
                ],
            ]
            .drop_duplicates(
                subset=[
                    "inchikey14"
                ]
            )
        )

        metrics = sample[
            [
                "inchikey14",
            ]
        ].merge(
            truth,
            on="inchikey14",
            how="left",
            validate="one_to_one",
        )

        summary_rows.append(
            summarize(
                f"guard_{guard_depth}",
                metrics[
                    "guarded_rank"
                ],
            )
        )

    summary = pd.DataFrame(
        summary_rows
    )

    summary.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print(
        "Guarded MIST + neighbor fusion"
    )
    print(
        "=============================="
    )
    print(
        "Structures:",
        f"{len(sample):,}",
    )
    print(
        "RRF k:",
        f"{RRF_K:g}",
    )
    print()
    print(
        summary.to_string(
            index=False,
            formatters={
                "mrr25":
                    lambda value:
                    f"{value:.4f}",
                "top1":
                    lambda value:
                    f"{value:.2%}",
                "top5":
                    lambda value:
                    f"{value:.2%}",
                "top10":
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
