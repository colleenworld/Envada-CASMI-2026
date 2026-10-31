from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(
    "data/processed/diagnostics/large_validation_audit/"
    "true_formula_neighbor_benchmark"
)

SEED = 20261030
BOOTSTRAPS = 20_000


def rr25(rank: pd.Series) -> np.ndarray:
    values = pd.to_numeric(
        rank,
        errors="coerce",
    ).to_numpy(
        dtype=float
    )

    out = np.zeros(
        len(values),
        dtype=np.float64,
    )

    valid = (
        np.isfinite(values)
        & (values >= 1)
        & (values <= 25)
    )

    out[valid] = (
        1.0
        / values[valid]
    )

    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "baseline",
        help="Baseline experiment directory suffix, e.g. meta_v1",
    )
    parser.add_argument(
        "candidate",
        help="Candidate experiment directory suffix, e.g. meta_entropy",
    )
    parser.add_argument(
        "--bootstraps",
        type=int,
        default=BOOTSTRAPS,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
    )
    return parser.parse_args()


def experiment_dir(name: str) -> Path:
    return (
        ROOT
        / f"learned_ranker_{name}"
    )


def load_predictions(name: str) -> pd.DataFrame:
    path = (
        experiment_dir(name)
        / "holdout_predictions.parquet"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Missing holdout predictions: {path}"
        )

    frame = pd.read_parquet(
        path
    )

    required = {
        "inchikey14",
        "truth_present",
        "learned_rank",
    }

    missing = (
        required
        - set(frame.columns)
    )

    if missing:
        raise ValueError(
            f"{path} is missing columns: {sorted(missing)}"
        )

    return frame[
        [
            "inchikey14",
            "truth_present",
            "learned_rank",
        ]
    ].rename(
        columns={
            "learned_rank":
                f"learned_rank_{name}",
            "truth_present":
                f"truth_present_{name}",
        }
    )


def paired_bootstrap(
    baseline: np.ndarray,
    candidate: np.ndarray,
    bootstraps: int,
    seed: int,
) -> tuple[float, float, float, float]:
    if len(baseline) != len(candidate):
        raise ValueError(
            "Paired samples must have equal length"
        )

    diff = (
        candidate
        - baseline
    )

    observed = float(
        diff.mean()
    )

    rng = np.random.default_rng(
        seed
    )

    n = len(diff)

    samples = np.empty(
        bootstraps,
        dtype=np.float64,
    )

    chunk = 1_000

    for start in range(
        0,
        bootstraps,
        chunk,
    ):
        stop = min(
            start + chunk,
            bootstraps,
        )

        indices = rng.integers(
            0,
            n,
            size=(
                stop - start,
                n,
            ),
        )

        samples[
            start:stop
        ] = diff[
            indices
        ].mean(
            axis=1
        )

    low, high = np.quantile(
        samples,
        [
            0.025,
            0.975,
        ],
    )

    probability_positive = float(
        np.mean(
            samples > 0
        )
    )

    return (
        observed,
        float(low),
        float(high),
        probability_positive,
    )


def summarize_subset(
    merged: pd.DataFrame,
    baseline_name: str,
    candidate_name: str,
    mask: pd.Series,
    bootstraps: int,
    seed: int,
) -> dict[str, object]:
    subset = merged.loc[
        mask
    ]

    baseline = rr25(
        subset[
            f"learned_rank_{baseline_name}"
        ]
    )

    candidate = rr25(
        subset[
            f"learned_rank_{candidate_name}"
        ]
    )

    (
        delta,
        low,
        high,
        probability_positive,
    ) = paired_bootstrap(
        baseline,
        candidate,
        bootstraps,
        seed,
    )

    return {
        "queries":
            len(subset),
        "baseline_mrr25":
            baseline.mean(),
        "candidate_mrr25":
            candidate.mean(),
        "delta_mrr25":
            delta,
        "relative_gain":
            (
                delta / baseline.mean()
                if baseline.mean() > 0
                else np.nan
            ),
        "ci95_low":
            low,
        "ci95_high":
            high,
        "bootstrap_p_gain":
            probability_positive,
        "queries_improved":
            int(
                np.sum(
                    candidate > baseline
                )
            ),
        "queries_worsened":
            int(
                np.sum(
                    candidate < baseline
                )
            ),
        "queries_unchanged":
            int(
                np.sum(
                    candidate == baseline
                )
            ),
    }


def main() -> None:
    args = parse_args()

    baseline = load_predictions(
        args.baseline
    )

    candidate = load_predictions(
        args.candidate
    )

    merged = baseline.merge(
        candidate,
        on="inchikey14",
        how="inner",
        validate="one_to_one",
    )

    if (
        len(merged) != len(baseline)
        or len(merged) != len(candidate)
    ):
        raise RuntimeError(
            "Experiments do not contain identical holdout queries"
        )

    base_truth = merged[
        f"truth_present_{args.baseline}"
    ].astype(bool)

    candidate_truth = merged[
        f"truth_present_{args.candidate}"
    ].astype(bool)

    if not np.array_equal(
        base_truth.to_numpy(),
        candidate_truth.to_numpy(),
    ):
        raise RuntimeError(
            "Experiments disagree on truth_present"
        )

    rows = []

    rows.append(
        {
            "subset":
                "all_holdout",
            **summarize_subset(
                merged,
                args.baseline,
                args.candidate,
                pd.Series(
                    True,
                    index=merged.index,
                ),
                args.bootstraps,
                args.seed,
            ),
        }
    )

    rows.append(
        {
            "subset":
                "truth_present",
            **summarize_subset(
                merged,
                args.baseline,
                args.candidate,
                base_truth,
                args.bootstraps,
                args.seed + 1,
            ),
        }
    )

    summary = pd.DataFrame(
        rows
    )

    print(
        summary.to_string(
            index=False,
            formatters={
                "baseline_mrr25":
                    lambda x: f"{x:.4f}",
                "candidate_mrr25":
                    lambda x: f"{x:.4f}",
                "delta_mrr25":
                    lambda x: f"{x:+.4f}",
                "relative_gain":
                    lambda x: f"{x:+.2%}",
                "ci95_low":
                    lambda x: f"{x:+.4f}",
                "ci95_high":
                    lambda x: f"{x:+.4f}",
                "bootstrap_p_gain":
                    lambda x: f"{x:.3%}",
            },
        )
    )


if __name__ == "__main__":
    main()
