from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(
    os.environ.get(
        "CASMI26_PROJECT_ROOT",
        Path(__file__).resolve().parent.parent,
    )
)

TEST_ROOT = PROJECT_ROOT / "data/processed/test"
CANDIDATES_PATH = TEST_ROOT / "guard5_neighbor_top500.parquet"
HYPOTHESES_PATH = TEST_ROOT / "multispectrum_formula_hypotheses.parquet"
FINGERPRINT_PATH = TEST_ROOT / "guard5_spectrum_fingerprint.parquet"
FULL_COCONUT_PATH = PROJECT_ROOT / "data/external/coconut/coconut_full.parquet"
APPROVED_COCONUT_PATH = PROJECT_ROOT / "data/external/coconut/coconut_approved.parquet"
OUTPUT_PATH = TEST_ROOT / "learned_ranker_features.parquet"

NEIGHBOR_FEATURE_COLUMNS = [
    "neighbor_tanimoto_max",
    "neighbor_tanimoto_mean",
    "neighbor_tanimoto_top3_mean",
    "neighbor_tanimoto_top5_mean",
    "neighbor_weighted_mean",
    "neighbor_weighted_top3_mean",
    "neighbor_count_ge_03",
    "neighbor_count_ge_05",
    "neighbor_count_ge_07",
    "best_structural_neighbor_spectral_score",
    "neighbor_feature_count",
]

SPECTRUM_FINGERPRINT_COLUMNS = [
    "predicted_fp_soft_jaccard",
    "predicted_fp_cosine",
    "predicted_fp_positive_mean",
    "predicted_fp_log_likelihood",
]


def main() -> None:
    for path in [
        CANDIDATES_PATH,
        HYPOTHESES_PATH,
        FINGERPRINT_PATH,
        FULL_COCONUT_PATH,
        APPROVED_COCONUT_PATH,
    ]:
        if not path.exists():
            raise FileNotFoundError(f"Missing required artifact: {path}")

    candidates = pd.read_parquet(CANDIDATES_PATH).copy()
    hypotheses = pd.read_parquet(HYPOTHESES_PATH).copy()
    fingerprint = pd.read_parquet(FINGERPRINT_PATH).copy()

    required_candidates = {
        "molecule_id",
        "hypothesis_spec",
        "candidate_formula",
        "candidate_inchikey14",
        "candidate_smiles",
        "prior_rank",
        "neighbor_prior_score",
        "usable_neighbor_fingerprints",
        *NEIGHBOR_FEATURE_COLUMNS,
    }

    missing = required_candidates - set(candidates.columns)
    if missing:
        raise ValueError(
            "Guard-5 candidate artifact missing learned-ranker columns: "
            f"{sorted(missing)}"
        )

    required_hypotheses = {
        "molecule_id",
        "hypothesis_spec",
        "candidate_formula",
        "min_abs_mass_error_ppm",
        "spectrum_support_count",
        "spectrum_support_fraction",
    }

    missing = required_hypotheses - set(hypotheses.columns)
    if missing:
        raise ValueError(
            "Formula hypothesis artifact missing learned-ranker columns: "
            f"{sorted(missing)}"
        )

    required_fp = {
        "hypothesis_spec",
        "candidate_formula",
        "candidate_inchikey14",
        *SPECTRUM_FINGERPRINT_COLUMNS,
    }

    missing = required_fp - set(fingerprint.columns)
    if missing:
        raise ValueError(
            "Fingerprint artifact missing learned-ranker columns: "
            f"{sorted(missing)}"
        )

    for frame in [candidates, hypotheses, fingerprint]:
        if "molecule_id" in frame.columns:
            frame["molecule_id"] = frame["molecule_id"].astype(str)
        if "hypothesis_spec" in frame.columns:
            frame["hypothesis_spec"] = frame["hypothesis_spec"].astype(str)
        if "candidate_formula" in frame.columns:
            frame["candidate_formula"] = frame["candidate_formula"].astype(str)
        if "candidate_inchikey14" in frame.columns:
            frame["candidate_inchikey14"] = frame["candidate_inchikey14"].astype(str)

    hypothesis_features = (
        hypotheses[
            [
                "molecule_id",
                "hypothesis_spec",
                "candidate_formula",
                "min_abs_mass_error_ppm",
                "spectrum_support_count",
                "spectrum_support_fraction",
            ]
        ]
        .sort_values(
            [
                "molecule_id",
                "min_abs_mass_error_ppm",
                "candidate_formula",
            ],
            ascending=[True, True, True],
            kind="stable",
        )
        .copy()
    )

    hypothesis_features["formula_rank"] = (
        hypothesis_features.groupby(
            "molecule_id",
            sort=False,
        ).cumcount()
        + 1
    )

    hypothesis_features = hypothesis_features.rename(
        columns={
            "min_abs_mass_error_ppm": "mass_error_best_abs_ppm",
            "spectrum_support_count": "formula_support_count",
            "spectrum_support_fraction": "formula_support_fraction",
        }
    )

    features = candidates.merge(
        hypothesis_features,
        on=[
            "molecule_id",
            "hypothesis_spec",
            "candidate_formula",
        ],
        how="left",
        validate="many_to_one",
    )

    if features[
        [
            "formula_rank",
            "mass_error_best_abs_ppm",
            "formula_support_count",
            "formula_support_fraction",
        ]
    ].isna().any(axis=None):
        raise RuntimeError(
            "Some guard-5 candidate rows did not match formula hypotheses"
        )

    fp_keys = [
        "hypothesis_spec",
        "candidate_formula",
        "candidate_inchikey14",
    ]

    features = features.merge(
        fingerprint[
            [
                *fp_keys,
                *SPECTRUM_FINGERPRINT_COLUMNS,
            ]
        ],
        on=fp_keys,
        how="left",
        validate="one_to_one",
    )

    if features[
        SPECTRUM_FINGERPRINT_COLUMNS
    ].isna().all(axis=None):
        raise RuntimeError(
            "Fingerprint merge produced no usable values"
        )

    query_counts = (
        features.groupby("molecule_id")["candidate_inchikey14"]
        .nunique()
        .rename("query_candidate_count")
    )

    formula_counts = (
        features.groupby(
            [
                "molecule_id",
                "candidate_formula",
            ]
        )["candidate_inchikey14"]
        .nunique()
        .rename("formula_candidate_count")
    )

    features = (
        features
        .merge(
            query_counts,
            on="molecule_id",
            how="left",
            validate="many_to_one",
        )
        .merge(
            formula_counts,
            on=[
                "molecule_id",
                "candidate_formula",
            ],
            how="left",
            validate="many_to_one",
        )
    )

    full_coconut = pd.read_parquet(
        FULL_COCONUT_PATH,
        columns=[
            "inchikey14",
            "np_likeness",
        ],
    )

    full_coconut["inchikey14"] = (
        full_coconut["inchikey14"].astype(str)
    )

    full_coconut = (
        full_coconut
        .drop_duplicates(
            subset=["inchikey14"]
        )
        .rename(
            columns={
                "inchikey14": "candidate_inchikey14",
                "np_likeness": "coconut_np_likeness",
            }
        )
    )

    approved = pd.read_parquet(
        APPROVED_COCONUT_PATH,
        columns=["inchikey14"],
    )

    approved_keys = set(
        approved["inchikey14"]
        .dropna()
        .astype(str)
    )

    features = features.merge(
        full_coconut,
        on="candidate_inchikey14",
        how="left",
        validate="many_to_one",
    )

    features["in_full_coconut"] = (
        features["coconut_np_likeness"].notna()
        | features["candidate_inchikey14"].isin(
            set(full_coconut["candidate_inchikey14"])
        )
    )

    features["in_approved_coconut"] = (
        features["candidate_inchikey14"].isin(
            approved_keys
        )
    )

    features["coconut_np_likeness"] = pd.to_numeric(
        features["coconut_np_likeness"],
        errors="coerce",
    )

    features["log_query_candidate_count"] = np.log1p(
        features["query_candidate_count"].astype(float)
    )

    features["log_formula_candidate_count"] = np.log1p(
        features["formula_candidate_count"].astype(float)
    )

    features["inverse_prior_rank"] = (
        1.0
        / features["prior_rank"].astype(float)
    )

    features["inverse_formula_rank"] = (
        1.0
        / features["formula_rank"].astype(float)
    )

    duplicate_count = int(
        features.duplicated(
            subset=[
                "molecule_id",
                "hypothesis_spec",
                "candidate_inchikey14",
            ]
        ).sum()
    )

    if duplicate_count:
        raise RuntimeError(
            f"Learned-ranker test features contain "
            f"{duplicate_count:,} duplicate candidate rows"
        )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    features.to_parquet(
        OUTPUT_PATH,
        index=False,
        compression="zstd",
    )

    print()
    print("Learned-ranker test feature table")
    print("=" * 76)
    print("Molecules:", f"{features['molecule_id'].nunique():,}")
    print("Candidate rows:", f"{len(features):,}")
    print(
        "Fingerprint coverage:",
        f"{features['predicted_fp_cosine'].notna().mean():.2%}",
    )
    print("Wrote:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
