from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(
    "data/processed/diagnostics/large_validation_audit"
)

BENCHMARK_DIR = (
    ROOT
    / "true_formula_neighbor_benchmark"
)


HYPOTHESES_PATH = (
    ROOT
    / "formula_hypotheses.parquet"
)

FULL_COCONUT_PATH = Path(
    "data/external/coconut/coconut_full.parquet"
)

APPROVED_COCONUT_PATH = Path(
    "data/external/coconut/coconut_approved.parquet"
)


SEED = 42
HOLDOUT_FRACTION = 0.20

NEIGHBOR_ENRICHMENT_COLUMNS = [
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidate-top-k",
        type=int,
        default=100,
        help="Per-formula neighbor candidate retention used by the source artifact.",
    )
    parser.add_argument(
        "--spectrum-fingerprint-name",
        default=None,
        help="Optional spectrum-fingerprint enrichment suffix to merge.",
    )
    parser.add_argument(
        "--experiment-name",
        default=None,
        help="Optional learned-ranker output directory suffix.",
    )
    args = parser.parse_args()

    if args.candidate_top_k <= 0:
        parser.error("--candidate-top-k must be positive")

    return args


def output_dir_for(
    top_k: int,
    experiment_name: str | None = None,
) -> Path:
    if experiment_name:
        return BENCHMARK_DIR / f"learned_ranker_{experiment_name}"

    if top_k == 100:
        return BENCHMARK_DIR / "learned_ranker"

    return BENCHMARK_DIR / f"learned_ranker_top{top_k}"


def main() -> None:
    args = parse_args()

    candidates_path = (
        BENCHMARK_DIR
        / "top3_neighbor_collapse"
        / f"top3_neighbor_top{args.candidate_top_k}.parquet"
    )

    output_dir = output_dir_for(
        args.candidate_top_k,
        args.experiment_name,
    )

    output_path = (
        output_dir
        / "candidate_features.parquet"
    )

    query_split_path = (
        output_dir
        / "query_split.parquet"
    )

    for path in [
        candidates_path,
        HYPOTHESES_PATH,
        FULL_COCONUT_PATH,
        APPROVED_COCONUT_PATH,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"Missing required artifact: {path}"
            )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    candidates = pd.read_parquet(
        candidates_path
    )

    hypotheses = pd.read_parquet(
        HYPOTHESES_PATH
    )

    full_coconut = pd.read_parquet(
        FULL_COCONUT_PATH,
        columns=[
            "inchikey14",
            "np_likeness",
        ],
    )

    approved_coconut = pd.read_parquet(
        APPROVED_COCONUT_PATH,
        columns=[
            "inchikey14",
        ],
    )

    required_candidates = {
        "inchikey14",
        "candidate_formula",
        "formula_rank",
        "mass_error_best_abs_ppm",
        "candidate_inchikey14",
        "prior_rank",
        "neighbor_prior_score",
        "usable_neighbor_fingerprints",
    }

    missing = (
        required_candidates
        - set(
            candidates.columns
        )
    )

    if missing:
        raise ValueError(
            "Candidate artifact missing columns: "
            f"{sorted(missing)}"
        )

    required_hypotheses = {
        "inchikey14",
        "candidate_formula",
        "formula_support_count",
        "formula_support_fraction",
        "mass_error_best_abs_ppm",
    }

    missing = (
        required_hypotheses
        - set(
            hypotheses.columns
        )
    )

    if missing:
        raise ValueError(
            "Hypothesis artifact missing columns: "
            f"{sorted(missing)}"
        )

    candidates[
        "inchikey14"
    ] = (
        candidates[
            "inchikey14"
        ]
        .astype(str)
    )

    candidates[
        "candidate_inchikey14"
    ] = (
        candidates[
            "candidate_inchikey14"
        ]
        .astype(str)
    )

    candidates[
        "candidate_formula"
    ] = (
        candidates[
            "candidate_formula"
        ]
        .astype(str)
    )

    benchmark_keys = set(
        candidates[
            "inchikey14"
        ]
    )

    hypotheses[
        "inchikey14"
    ] = (
        hypotheses[
            "inchikey14"
        ]
        .astype(str)
    )

    hypotheses[
        "candidate_formula"
    ] = (
        hypotheses[
            "candidate_formula"
        ]
        .astype(str)
    )

    hypothesis_features = (
        hypotheses.loc[
            hypotheses[
                "inchikey14"
            ]
            .isin(
                benchmark_keys
            ),
            [
                "inchikey14",
                "candidate_formula",
                "formula_support_count",
                "formula_support_fraction",
                "mass_error_best_abs_ppm",
            ],
        ]
        .drop_duplicates(
            subset=[
                "inchikey14",
                "candidate_formula",
            ]
        )
        .rename(
            columns={
                "mass_error_best_abs_ppm":
                    "hypothesis_mass_error_best_abs_ppm",
            }
        )
    )

    features = candidates.merge(
        hypothesis_features,
        on=[
            "inchikey14",
            "candidate_formula",
        ],
        how="left",
        validate="many_to_one",
    )

    if args.candidate_top_k >= 500:
        enrichment_path = (
            BENCHMARK_DIR
            / "learned_ranker_top500"
            / "neighbor_enrichment.parquet"
        )

        if not enrichment_path.exists():
            raise FileNotFoundError(
                "Missing top-500 neighbor enrichment. Run "
                "scripts/enrich_top500_neighbor_features.py first: "
                f"{enrichment_path}"
            )

        enrichment = pd.read_parquet(
            enrichment_path
        )

        required_enrichment = {
            "inchikey14",
            "candidate_formula",
            "candidate_inchikey14",
            *NEIGHBOR_ENRICHMENT_COLUMNS,
        }

        missing_enrichment = (
            required_enrichment
            - set(
                enrichment.columns
            )
        )

        if missing_enrichment:
            raise ValueError(
                "Neighbor enrichment is missing columns: "
                f"{sorted(missing_enrichment)}"
            )

        features = features.merge(
            enrichment[
                [
                    "inchikey14",
                    "candidate_formula",
                    "candidate_inchikey14",
                    *NEIGHBOR_ENRICHMENT_COLUMNS,
                ]
            ],
            on=[
                "inchikey14",
                "candidate_formula",
                "candidate_inchikey14",
            ],
            how="left",
            validate="one_to_one",
        )

        if features[
            NEIGHBOR_ENRICHMENT_COLUMNS
        ].isna().all(axis=None):
            raise RuntimeError(
                "Neighbor enrichment merge produced no usable values"
            )

        spectrum_fp_path = None

        if args.spectrum_fingerprint_name:
            spectrum_fp_path = (
                BENCHMARK_DIR
                / "learned_ranker_top500"
                / (
                    "spectrum_fingerprint_enrichment_"
                    f"{args.spectrum_fingerprint_name}.parquet"
                )
            )

            if not spectrum_fp_path.exists():
                raise FileNotFoundError(
                    "Missing requested spectrum-fingerprint enrichment: "
                    f"{spectrum_fp_path}"
                )

        if spectrum_fp_path is not None:
            spectrum_fp = pd.read_parquet(
                spectrum_fp_path
            )

            required_spectrum_fp = {
                "inchikey14",
                "candidate_formula",
                "candidate_inchikey14",
                *SPECTRUM_FINGERPRINT_COLUMNS,
            }

            missing_spectrum_fp = (
                required_spectrum_fp
                - set(
                    spectrum_fp.columns
                )
            )

            if missing_spectrum_fp:
                raise ValueError(
                    "Spectrum-fingerprint enrichment is missing columns: "
                    f"{sorted(missing_spectrum_fp)}"
                )

            features = features.merge(
                spectrum_fp[
                    [
                        "inchikey14",
                        "candidate_formula",
                        "candidate_inchikey14",
                        *SPECTRUM_FINGERPRINT_COLUMNS,
                    ]
                ],
                on=[
                    "inchikey14",
                    "candidate_formula",
                    "candidate_inchikey14",
                ],
                how="left",
                validate="one_to_one",
            )

            print(
                "Merged spectrum-fingerprint enrichment:",
                spectrum_fp_path,
            )

    if features[
        "formula_support_count"
    ].isna().any():
        raise RuntimeError(
            "Some candidate rows did not match a formula hypothesis"
        )

    features[
        "is_truth"
    ] = (
        features[
            "candidate_inchikey14"
        ]
        == features[
            "inchikey14"
        ]
    )

    #
    # Query- and formula-level burden features.
    #
    query_counts = (
        features.groupby(
            "inchikey14"
        )[
            "candidate_inchikey14"
        ]
        .nunique()
        .rename(
            "query_candidate_count"
        )
    )

    formula_counts = (
        features.groupby(
            [
                "inchikey14",
                "candidate_formula",
            ]
        )[
            "candidate_inchikey14"
        ]
        .nunique()
        .rename(
            "formula_candidate_count"
        )
    )

    features = (
        features
        .merge(
            query_counts,
            on="inchikey14",
            how="left",
            validate="many_to_one",
        )
        .merge(
            formula_counts,
            on=[
                "inchikey14",
                "candidate_formula",
            ],
            how="left",
            validate="many_to_one",
        )
    )

    #
    # COCONUT membership and NP-likeness.
    #
    full_coconut[
        "inchikey14"
    ] = (
        full_coconut[
            "inchikey14"
        ]
        .astype(str)
    )

    full_coconut = (
        full_coconut
        .drop_duplicates(
            subset=[
                "inchikey14"
            ]
        )
        .rename(
            columns={
                "inchikey14":
                    "candidate_inchikey14",
                "np_likeness":
                    "coconut_np_likeness",
            }
        )
    )

    approved_keys = set(
        approved_coconut[
            "inchikey14"
        ]
        .dropna()
        .astype(str)
    )

    features = features.merge(
        full_coconut,
        on="candidate_inchikey14",
        how="left",
        validate="many_to_one",
    )

    features[
        "in_full_coconut"
    ] = (
        features[
            "coconut_np_likeness"
        ].notna()
        | features[
            "candidate_inchikey14"
        ].isin(
            set(
                full_coconut[
                    "candidate_inchikey14"
                ]
            )
        )
    )

    features[
        "in_approved_coconut"
    ] = (
        features[
            "candidate_inchikey14"
        ]
        .isin(
            approved_keys
        )
    )

    features[
        "coconut_np_likeness"
    ] = pd.to_numeric(
        features[
            "coconut_np_likeness"
        ],
        errors="coerce",
    )

    #
    # A few scale-stabilized versions that tree models can use directly.
    #
    features[
        "log_query_candidate_count"
    ] = np.log1p(
        features[
            "query_candidate_count"
        ].astype(float)
    )

    features[
        "log_formula_candidate_count"
    ] = np.log1p(
        features[
            "formula_candidate_count"
        ].astype(float)
    )

    features[
        "inverse_prior_rank"
    ] = (
        1.0
        / features[
            "prior_rank"
        ].astype(float)
    )

    features[
        "inverse_formula_rank"
    ] = (
        1.0
        / features[
            "formula_rank"
        ].astype(float)
    )

    #
    # Deterministic structure-disjoint train/holdout split.
    #
    query_keys = (
        features[
            "inchikey14"
        ]
        .drop_duplicates()
        .sort_values()
        .to_numpy(
            dtype=object
        )
    )

    rng = np.random.default_rng(
        SEED
    )

    shuffled = query_keys.copy()

    rng.shuffle(
        shuffled
    )

    holdout_count = max(
        1,
        int(
            round(
                len(
                    shuffled
                )
                * HOLDOUT_FRACTION
            )
        ),
    )

    holdout_keys = set(
        shuffled[
            :holdout_count
        ]
    )

    query_split = pd.DataFrame(
        {
            "inchikey14":
                query_keys,
            "split":
                [
                    (
                        "holdout"
                        if key in holdout_keys
                        else "train"
                    )
                    for key
                    in query_keys
                ],
        }
    )

    features = features.merge(
        query_split,
        on="inchikey14",
        how="left",
        validate="many_to_one",
    )

    #
    # Queries where the truth was not retained remain in the evaluation
    # population, but cannot supply a positive example for training.
    #
    truth_by_query = (
        features.groupby(
            "inchikey14"
        )[
            "is_truth"
        ]
        .any()
        .rename(
            "truth_present"
        )
        .reset_index()
    )

    query_split = query_split.merge(
        truth_by_query,
        on="inchikey14",
        how="left",
        validate="one_to_one",
    )

    features = features.merge(
        truth_by_query,
        on="inchikey14",
        how="left",
        validate="many_to_one",
    )

    features.to_parquet(
        output_path,
        index=False,
        compression="zstd",
    )

    query_split.to_parquet(
        query_split_path,
        index=False,
        compression="zstd",
    )

    print()
    print(
        "Learned-ranker candidate dataset"
    )
    print(
        "=" * 72
    )
    print(
        "Queries:",
        f"{len(query_split):,}",
    )
    print(
        "Candidate rows:",
        f"{len(features):,}",
    )
    print(
        "Positive rows:",
        f"{int(features['is_truth'].sum()):,}",
    )
    print()
    print(
        "Train queries:",
        f"{int((query_split['split'] == 'train').sum()):,}",
    )
    print(
        "Train queries with truth retained:",
        f"{int(((query_split['split'] == 'train') & query_split['truth_present']).sum()):,}",
    )
    print(
        "Holdout queries:",
        f"{int((query_split['split'] == 'holdout').sum()):,}",
    )
    print(
        "Holdout queries with truth retained:",
        f"{int(((query_split['split'] == 'holdout') & query_split['truth_present']).sum()):,}",
    )
    print()
    print(
        "Candidate rows/query:"
    )
    print(
        features.groupby(
            "inchikey14"
        )
        .size()
        .describe(
            percentiles=[
                0.5,
                0.9,
                0.95,
                0.99,
            ]
        )
        .to_string()
    )
    print()
    print(
        "Full COCONUT candidate fraction:",
        f"{features['in_full_coconut'].mean():.2%}",
    )
    print(
        "Approved COCONUT candidate fraction:",
        f"{features['in_approved_coconut'].mean():.2%}",
    )
    print()
    print(
        "Wrote:",
        output_path,
    )
    print(
        "Wrote:",
        query_split_path,
    )


if __name__ == "__main__":
    main()
