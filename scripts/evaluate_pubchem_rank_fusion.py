from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent

BENCHMARK_DIR = (
    PROJECT_ROOT
    / "data/processed/diagnostics/pubchem_mist_benchmark"
)

SAMPLE_PATH = BENCHMARK_DIR / "sample.parquet"

NEIGHBOR_PATH = (
    BENCHMARK_DIR
    / "neighbor_pruning_ranked.parquet"
)

MIST_PATH = (
    BENCHMARK_DIR
    / "pruned"
    / "ranked.parquet"
)

OUTPUT_PATH = (
    BENCHMARK_DIR
    / "pruned"
    / "rank_fusion.parquet"
)

#
# Standard Reciprocal Rank Fusion constant.
# Fixed a priori; not tuned on this benchmark.
#
RRF_K = 60.0


def main() -> None:
    sample = pd.read_parquet(
        SAMPLE_PATH
    )

    neighbor = pd.read_parquet(
        NEIGHBOR_PATH
    )

    mist = pd.read_parquet(
        MIST_PATH
    )

    required_sample = {
        "spec",
        "inchikey14",
        "burden_bucket",
        "candidate_count",
    }

    missing = (
        required_sample
        - set(sample.columns)
    )

    if missing:
        raise ValueError(
            "Benchmark sample is missing columns: "
            f"{sorted(missing)}"
        )

    required_neighbor = {
        "spec",
        "candidate_inchikey14",
        "prior_rank",
        "neighbor_prior_score",
    }

    missing = (
        required_neighbor
        - set(neighbor.columns)
    )

    if missing:
        raise ValueError(
            "Neighbor ranking is missing columns: "
            f"{sorted(missing)}"
        )

    required_mist = {
        "spec",
        "candidate_inchikey14",
        "rank",
        "distance",
    }

    missing = (
        required_mist
        - set(mist.columns)
    )

    if missing:
        raise ValueError(
            "Pruned MIST ranking is missing columns: "
            f"{sorted(missing)}"
        )

    #
    # MIST's candidate universe is the actual pruned retrieval result.
    # Attach the independently-computed neighbor rank for those same
    # candidate/spec pairs.
    #
    fused = mist[
        [
            "spec",
            "candidate_inchikey14",
            "candidate_smiles",
            "distance",
            "rank",
            "burden_bucket",
            "candidate_count",
        ]
    ].rename(
        columns={
            "rank":
                "mist_rank",
        }
    ).merge(
        neighbor[
            [
                "spec",
                "candidate_inchikey14",
                "prior_rank",
                "neighbor_prior_score",
            ]
        ],
        on=[
            "spec",
            "candidate_inchikey14",
        ],
        how="left",
        validate="one_to_one",
    )

    missing_neighbor = (
        fused[
            "prior_rank"
        ].isna()
    )

    if missing_neighbor.any():
        examples = (
            fused.loc[
                missing_neighbor,
                [
                    "spec",
                    "candidate_inchikey14",
                ],
            ]
            .head(20)
        )

        raise RuntimeError(
            "Some MIST candidates have no neighbor-prior rank. "
            "This should not happen because the neighbor ranking "
            "covers the full PubChem formula candidate set. "
            f"Examples:\n{examples.to_string(index=False)}"
        )

    fused[
        "rrf_score"
    ] = (
        1.0
        / (
            RRF_K
            + fused[
                "mist_rank"
            ].astype(float)
        )
        + 1.0
        / (
            RRF_K
            + fused[
                "prior_rank"
            ].astype(float)
        )
    )

    fused = (
        fused
        .sort_values(
            [
                "spec",
                "rrf_score",
                "candidate_inchikey14",
            ],
            ascending=[
                True,
                False,
                True,
            ],
        )
        .reset_index(
            drop=True
        )
    )

    fused[
        "fusion_rank"
    ] = (
        fused.groupby(
            "spec"
        )
        .cumcount()
        + 1
    )

    truth_lookup = (
        sample[
            [
                "spec",
                "inchikey14",
                "burden_bucket",
                "candidate_count",
            ]
        ]
        .rename(
            columns={
                "inchikey14":
                    "truth_inchikey14",
            }
        )
    )

    fused = fused.merge(
        truth_lookup[
            [
                "spec",
                "truth_inchikey14",
            ]
        ],
        on="spec",
        how="left",
        validate="many_to_one",
    )

    fused.to_parquet(
        OUTPUT_PATH,
        index=False,
    )

    metrics = (
        truth_lookup.copy()
    )

    truth_rows = (
        fused.loc[
            fused[
                "candidate_inchikey14"
            ]
            == fused[
                "truth_inchikey14"
            ],
            [
                "spec",
                "fusion_rank",
                "mist_rank",
                "prior_rank",
            ],
        ]
        .drop_duplicates(
            subset=[
                "spec",
            ]
        )
    )

    metrics = metrics.merge(
        truth_rows,
        on="spec",
        how="left",
        validate="one_to_one",
    )

    metrics[
        "fusion_rr_at_25"
    ] = np.where(
        metrics[
            "fusion_rank"
        ].between(
            1,
            25,
            inclusive="both",
        ),
        1.0
        / metrics[
            "fusion_rank"
        ],
        0.0,
    )

    print(
        "Neighbor + pruned MIST reciprocal-rank fusion"
    )
    print(
        "============================================"
    )
    print(
        f"RRF k: {RRF_K:g}"
    )

    def report(
        label: str,
        frame: pd.DataFrame,
    ) -> None:
        ranks = frame[
            "fusion_rank"
        ]

        print()
        print(label)
        print(
            "-" * len(label)
        )
        print(
            "Molecules:",
            len(frame),
        )
        print(
            "MRR@25:",
            f"{frame['fusion_rr_at_25'].mean():.4f}",
        )
        print(
            "Top-1:",
            f"{(ranks == 1).mean():.1%}",
        )
        print(
            "Top-25:",
            f"{ranks.between(1, 25).mean():.1%}",
        )
        print(
            "Truth returned:",
            f"{ranks.notna().mean():.1%}",
        )

    report(
        "Overall",
        metrics,
    )

    for bucket_name in [
        "le100",
        "101to500",
        "501to1000",
        "1001to5000",
    ]:
        bucket = metrics.loc[
            metrics[
                "burden_bucket"
            ]
            == bucket_name
        ]

        if not bucket.empty:
            report(
                bucket_name,
                bucket,
            )

    print()
    print(
        f"Wrote {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()
