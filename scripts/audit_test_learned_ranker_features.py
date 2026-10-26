from __future__ import annotations

import os
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(
    os.environ.get(
        "CASMI26_PROJECT_ROOT",
        Path(__file__).resolve().parent.parent,
    )
)

TEST_FEATURES_PATH = (
    PROJECT_ROOT
    / "data/processed/test/learned_ranker_features.parquet"
)

VALIDATION_FEATURES_PATH = (
    PROJECT_ROOT
    / "data/processed/diagnostics/large_validation_audit/"
      "true_formula_neighbor_benchmark/"
      "learned_ranker_top500/candidate_features.parquet"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "data/processed/diagnostics/large_validation_audit/"
      "true_formula_neighbor_benchmark/"
      "learned_ranker_top500/lgbm_ranker.txt"
)


def finite_summary(
    series: pd.Series,
) -> dict[str, float]:
    values = pd.to_numeric(
        series,
        errors="coerce",
    ).replace(
        [
            np.inf,
            -np.inf,
        ],
        np.nan,
    )

    finite = values.dropna()

    if finite.empty:
        return {
            "missing": float(values.isna().mean()),
            "median": np.nan,
            "p05": np.nan,
            "p95": np.nan,
        }

    return {
        "missing": float(values.isna().mean()),
        "median": float(finite.median()),
        "p05": float(finite.quantile(0.05)),
        "p95": float(finite.quantile(0.95)),
    }


def main() -> None:
    for path in [
        TEST_FEATURES_PATH,
        VALIDATION_FEATURES_PATH,
        MODEL_PATH,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"Missing required artifact: {path}"
            )

    test = pd.read_parquet(
        TEST_FEATURES_PATH
    )

    validation = pd.read_parquet(
        VALIDATION_FEATURES_PATH
    )

    holdout = validation.loc[
        validation["split"] == "holdout"
    ].copy()

    booster = lgb.Booster(
        model_file=str(MODEL_PATH)
    )

    feature_columns = list(
        booster.feature_name()
    )

    missing_test = [
        column
        for column in feature_columns
        if column not in test.columns
    ]

    missing_holdout = [
        column
        for column in feature_columns
        if column not in holdout.columns
    ]

    if missing_test or missing_holdout:
        raise ValueError(
            "Feature parity failure. "
            f"Missing from test={missing_test}; "
            f"missing from holdout={missing_holdout}"
        )

    rows = []

    for column in feature_columns:
        test_stats = finite_summary(
            test[column]
        )
        holdout_stats = finite_summary(
            holdout[column]
        )

        rows.append(
            {
                "feature": column,
                "holdout_missing": holdout_stats["missing"],
                "test_missing": test_stats["missing"],
                "holdout_median": holdout_stats["median"],
                "test_median": test_stats["median"],
                "holdout_p05": holdout_stats["p05"],
                "test_p05": test_stats["p05"],
                "holdout_p95": holdout_stats["p95"],
                "test_p95": test_stats["p95"],
            }
        )

    summary = pd.DataFrame(rows)

    print()
    print("Learned-ranker test/holdout feature audit")
    print("=" * 100)
    print("Model features:", len(feature_columns))
    print("Holdout rows:", f"{len(holdout):,}")
    print("Test rows:", f"{len(test):,}")
    print("Holdout queries:", f"{holdout['inchikey14'].nunique():,}")
    print("Test molecules:", f"{test['molecule_id'].nunique():,}")
    print()
    print(
        summary.to_string(
            index=False,
            formatters={
                "holdout_missing": lambda value: f"{value:.2%}",
                "test_missing": lambda value: f"{value:.2%}",
                "holdout_median": lambda value: f"{value:.5g}",
                "test_median": lambda value: f"{value:.5g}",
                "holdout_p05": lambda value: f"{value:.5g}",
                "test_p05": lambda value: f"{value:.5g}",
                "holdout_p95": lambda value: f"{value:.5g}",
                "test_p95": lambda value: f"{value:.5g}",
            },
        )
    )

    severe_missingness = summary.loc[
        (
            summary["test_missing"]
            - summary["holdout_missing"]
        )
        > 0.10
    ]

    if not severe_missingness.empty:
        print()
        print("WARNING: >10 percentage-point missingness shift:")
        print(
            severe_missingness[
                [
                    "feature",
                    "holdout_missing",
                    "test_missing",
                ]
            ].to_string(
                index=False,
                formatters={
                    "holdout_missing": lambda value: f"{value:.2%}",
                    "test_missing": lambda value: f"{value:.2%}",
                },
            )
        )

    print()
    print(
        "Audit complete. Inspect large distribution shifts before Kaggle."
    )


if __name__ == "__main__":
    main()
