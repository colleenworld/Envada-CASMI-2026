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
    / "hybrid_policy_ranked.parquet"
)

RRF_K = 60.0


def main() -> None:
    sample = pd.read_parquet(SAMPLE_PATH)
    neighbor = pd.read_parquet(NEIGHBOR_PATH)
    mist = pd.read_parquet(MIST_PATH)

    required_sample = {
        "spec",
        "inchikey14",
        "burden_bucket",
        "candidate_count",
    }

    missing = required_sample - set(sample.columns)
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

    missing = required_neighbor - set(neighbor.columns)
    if missing:
        raise ValueError(
            "Neighbor ranking is missing columns: "
            f"{sorted(missing)}"
        )

    required_mist = {
        "spec",
        "candidate_inchikey14",
        "candidate_smiles",
        "truth_inchikey14",
        "rank",
        "distance",
        "burden_bucket",
        "candidate_count",
    }

    missing = required_mist - set(mist.columns)
    if missing:
        raise ValueError(
            "Pruned MIST ranking is missing columns: "
            f"{sorted(missing)}"
        )

    ranked = mist.rename(
        columns={
            "rank": "mist_rank",
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

    missing_neighbor = ranked["prior_rank"].isna()

    if missing_neighbor.any():
        examples = ranked.loc[
            missing_neighbor,
            [
                "spec",
                "candidate_inchikey14",
            ],
        ].head(20)

        raise RuntimeError(
            "Some MIST candidates have no neighbor-prior rank. "
            f"Examples:\n{examples.to_string(index=False)}"
        )

    ranked["rrf_score"] = (
        1.0
        / (
            RRF_K
            + ranked["mist_rank"].astype(float)
        )
        + 1.0
        / (
            RRF_K
            + ranked["prior_rank"].astype(float)
        )
    )

    #
    # Policy fixed before this evaluation:
    #
    # <=100 PubChem candidates: MIST alone.
    # >100 candidates: fixed reciprocal-rank fusion.
    #
    ranked["ranking_method"] = np.where(
        ranked["burden_bucket"] == "le100",
        "mist",
        "rrf",
    )

    ranked["hybrid_sort_score"] = np.where(
        ranked["burden_bucket"] == "le100",
        -ranked["mist_rank"].astype(float),
        ranked["rrf_score"].astype(float),
    )

    ranked = (
        ranked
        .sort_values(
            [
                "spec",
                "hybrid_sort_score",
                "candidate_inchikey14",
            ],
            ascending=[
                True,
                False,
                True,
            ],
        )
        .reset_index(drop=True)
    )

    ranked["hybrid_rank"] = (
        ranked.groupby("spec")
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
                "inchikey14": "truth_inchikey14",
            }
        )
    )

    expected_truth = (
        truth_lookup[
            [
                "spec",
                "truth_inchikey14",
            ]
        ]
        .set_index("spec")[
            "truth_inchikey14"
        ]
        .astype(str)
    )

    ranked_truth = (
        ranked["truth_inchikey14"]
        .astype(str)
    )

    expected_ranked_truth = (
        ranked["spec"]
        .map(expected_truth)
        .astype(str)
    )

    truth_mismatch = (
        ranked_truth
        != expected_ranked_truth
    )

    if truth_mismatch.any():
        examples = ranked.loc[
            truth_mismatch,
            [
                "spec",
                "truth_inchikey14",
            ],
        ].head(20)

        raise RuntimeError(
            "Pruned MIST truth keys disagree with the benchmark sample. "
            f"Examples:\n{examples.to_string(index=False)}"
        )

    ranked.to_parquet(
        OUTPUT_PATH,
        index=False,
    )

    truth_rows = (
        ranked.loc[
            ranked["candidate_inchikey14"]
            == ranked["truth_inchikey14"],
            [
                "spec",
                "hybrid_rank",
                "mist_rank",
                "prior_rank",
                "ranking_method",
            ],
        ]
        .drop_duplicates(
            subset=["spec"]
        )
    )

    metrics = truth_lookup.merge(
        truth_rows,
        on="spec",
        how="left",
        validate="one_to_one",
    )

    metrics["rr_at_25"] = np.where(
        metrics["hybrid_rank"].between(
            1,
            25,
            inclusive="both",
        ),
        1.0 / metrics["hybrid_rank"],
        0.0,
    )

    print(
        "Adaptive PubChem ranking policy"
    )
    print(
        "==============================="
    )
    print(
        "Policy: <=100 MIST; >100 fixed RRF"
    )
    print(
        f"RRF k: {RRF_K:g}"
    )

    def report(
        label: str,
        frame: pd.DataFrame,
    ) -> None:
        ranks = frame["hybrid_rank"]

        print()
        print(label)
        print("-" * len(label))
        print("Molecules:", len(frame))
        print(
            "MRR@25:",
            f"{frame['rr_at_25'].mean():.4f}",
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

    report("Overall", metrics)

    for bucket_name in [
        "le100",
        "101to500",
        "501to1000",
        "1001to5000",
    ]:
        bucket = metrics.loc[
            metrics["burden_bucket"] == bucket_name
        ]

        if not bucket.empty:
            report(bucket_name, bucket)

    print()
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
