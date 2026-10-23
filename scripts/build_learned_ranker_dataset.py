from __future__ import annotations

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

CANDIDATES_PATH = (
    BENCHMARK_DIR
    / "top3_neighbor_collapse"
    / "top3_neighbor_top100.parquet"
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

OUTPUT_DIR = (
    BENCHMARK_DIR
    / "learned_ranker"
)

OUTPUT_PATH = (
    OUTPUT_DIR
    / "candidate_features.parquet"
)

QUERY_SPLIT_PATH = (
    OUTPUT_DIR
    / "query_split.parquet"
)

SEED = 42
HOLDOUT_FRACTION = 0.20


def main() -> None:
    for path in [
        CANDIDATES_PATH,
        HYPOTHESES_PATH,
        FULL_COCONUT_PATH,
        APPROVED_COCONUT_PATH,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"Missing required artifact: {path}"
            )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    candidates = pd.read_parquet(
        CANDIDATES_PATH
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
        OUTPUT_PATH,
        index=False,
        compression="zstd",
    )

    query_split.to_parquet(
        QUERY_SPLIT_PATH,
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
        OUTPUT_PATH,
    )
    print(
        "Wrote:",
        QUERY_SPLIT_PATH,
    )


if __name__ == "__main__":
    main()
