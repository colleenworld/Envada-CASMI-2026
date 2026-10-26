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

TEST_ROOT = PROJECT_ROOT / "data/processed/test"

FEATURES_PATH = (
    TEST_ROOT
    / "learned_ranker_features.parquet"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "data/processed/diagnostics/large_validation_audit/"
      "true_formula_neighbor_benchmark/"
      "learned_ranker_top500/lgbm_ranker.txt"
)

OUTPUT_PATH = (
    TEST_ROOT
    / "positive_contrastive_ranked_candidates.parquet"
)


def main() -> None:
    for path in [
        FEATURES_PATH,
        MODEL_PATH,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"Missing required artifact: {path}"
            )

    frame = pd.read_parquet(
        FEATURES_PATH
    ).copy()

    required_identity = {
        "molecule_id",
        "candidate_inchikey14",
        "candidate_smiles",
        "candidate_formula",
        "hypothesis_spec",
    }

    missing = (
        required_identity
        - set(frame.columns)
    )

    if missing:
        raise ValueError(
            "Learned-ranker feature table is missing identity columns: "
            f"{sorted(missing)}"
        )

    booster = lgb.Booster(
        model_file=str(MODEL_PATH)
    )

    feature_columns = list(
        booster.feature_name()
    )

    missing_features = [
        column
        for column in feature_columns
        if column not in frame.columns
    ]

    if missing_features:
        raise ValueError(
            "Test feature table does not match the saved LightGBM model. "
            f"Missing: {missing_features}"
        )

    x = (
        frame[
            feature_columns
        ]
        .copy()
    )

    for column in [
        "in_full_coconut",
        "in_approved_coconut",
    ]:
        if column in x.columns:
            x[column] = (
                x[column]
                .astype(np.int8)
            )

    x = x.replace(
        [
            np.inf,
            -np.inf,
        ],
        np.nan,
    )

    scores = booster.predict(
        x.to_numpy(
            dtype=np.float32
        )
    )

    if len(scores) != len(frame):
        raise RuntimeError(
            "LightGBM prediction row count mismatch"
        )

    ranked = frame.copy()

    ranked[
        "learned_score"
    ] = np.asarray(
        scores,
        dtype=float,
    )

    ranked = (
        ranked
        .sort_values(
            [
                "molecule_id",
                "learned_score",
                "candidate_inchikey14",
            ],
            ascending=[
                True,
                False,
                True,
            ],
            kind="stable",
        )
        .reset_index(
            drop=True
        )
    )

    ranked[
        "rank"
    ] = (
        ranked.groupby(
            "molecule_id",
            sort=False,
        )
        .cumcount()
        + 1
    )

    keep = [
        "molecule_id",
        "hypothesis_spec",
        "candidate_formula",
        "candidate_inchikey14",
        "candidate_smiles",
        "prior_rank",
        "neighbor_prior_score",
        "learned_score",
        "rank",
    ]

    ranked[
        keep
    ].to_parquet(
        OUTPUT_PATH,
        index=False,
        compression="zstd",
    )

    print()
    print("Learned-ranker positive test ranking")
    print("=" * 76)
    print(
        "Model features:",
        len(feature_columns),
    )
    print(
        "Ranked molecules:",
        f"{ranked['molecule_id'].nunique():,}",
    )
    print(
        "Ranked rows:",
        f"{len(ranked):,}",
    )
    print()
    print("Model feature order:")
    for column in feature_columns:
        print(" ", column)
    print()
    print("Wrote:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
