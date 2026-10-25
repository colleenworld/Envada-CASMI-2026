from __future__ import annotations

from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd


BASE = Path(
    "data/processed/diagnostics/large_validation_audit/"
    "true_formula_neighbor_benchmark"
)

BEST_ROOT = (
    BASE
    / "learned_ranker_top500"
)

FEATURES_PATH = (
    BEST_ROOT
    / "candidate_features.parquet"
)

MODEL_PATH = (
    BEST_ROOT
    / "lgbm_ranker.txt"
)

SCALED_FP_PATH = (
    BEST_ROOT
    / "spectrum_fingerprint_enrichment_scaled150k.parquet"
)

OUTPUT_PATH = (
    BASE
    / "learned_ranker_top500_scaled150k_fusion.csv"
)

BASE_FEATURE_COLUMNS = [
    "formula_rank",
    "mass_error_best_abs_ppm",
    "formula_support_count",
    "formula_support_fraction",
    "prior_rank",
    "neighbor_prior_score",
    "usable_neighbor_fingerprints",
    "query_candidate_count",
    "formula_candidate_count",
    "coconut_np_likeness",
    "in_full_coconut",
    "in_approved_coconut",
    "log_query_candidate_count",
    "log_formula_candidate_count",
    "inverse_prior_rank",
    "inverse_formula_rank",
]

OPTIONAL_NEIGHBOR_FEATURE_COLUMNS = [
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

OPTIONAL_SPECTRUM_FINGERPRINT_COLUMNS = [
    "predicted_fp_soft_jaccard",
    "predicted_fp_cosine",
    "predicted_fp_positive_mean",
    "predicted_fp_log_likelihood",
]

SCALED_FEATURES = [
    "predicted_fp_soft_jaccard",
    "predicted_fp_cosine",
    "predicted_fp_positive_mean",
    "predicted_fp_log_likelihood",
]

FUSION_WEIGHTS = [
    0.00,
    0.10,
    0.20,
    0.30,
    0.40,
    0.50,
    0.75,
    1.00,
    1.50,
    2.00,
    3.00,
    5.00,
    10.00,
]


def rr25(rank) -> float:
    if rank is None or pd.isna(rank):
        return 0.0

    rank = int(rank)

    if rank < 1 or rank > 25:
        return 0.0

    return 1.0 / rank


def summarize(
    label: str,
    ranks: pd.Series,
) -> dict[str, object]:
    return {
        "policy":
            label,
        "queries":
            len(ranks),
        "mrr25":
            ranks.map(rr25).mean(),
        "top1":
            ranks.eq(1).mean(),
        "top5":
            ranks.between(1, 5).mean(),
        "top10":
            ranks.between(1, 10).mean(),
        "top25":
            ranks.between(1, 25).mean(),
        "top100":
            ranks.between(1, 100).mean(),
        "truth_returned":
            ranks.notna().mean(),
    }


def add_rank(
    frame: pd.DataFrame,
    *,
    score_column: str,
    rank_column: str,
    ascending: bool,
) -> pd.DataFrame:
    ranked = (
        frame
        .sort_values(
            [
                "inchikey14",
                score_column,
                "candidate_inchikey14",
            ],
            ascending=[
                True,
                ascending,
                True,
            ],
            kind="stable",
        )
        .copy()
    )

    ranked[
        rank_column
    ] = (
        ranked.groupby(
            "inchikey14",
            sort=False,
        )
        .cumcount()
        + 1
    )

    return ranked


def truth_ranks(
    ranked: pd.DataFrame,
    rank_column: str,
    query_keys: pd.Series,
) -> pd.Series:
    truth = (
        ranked.loc[
            ranked[
                "is_truth"
            ].astype(bool),
            [
                "inchikey14",
                rank_column,
            ],
        ]
        .drop_duplicates(
            subset=[
                "inchikey14"
            ]
        )
        .set_index(
            "inchikey14"
        )[
            rank_column
        ]
    )

    return (
        query_keys
        .astype(str)
        .map(
            truth
        )
    )


def main() -> None:
    for path in [
        FEATURES_PATH,
        MODEL_PATH,
        SCALED_FP_PATH,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"Missing required artifact: {path}"
            )

    frame = pd.read_parquet(
        FEATURES_PATH
    )

    holdout = frame.loc[
        frame[
            "split"
        ]
        == "holdout"
    ].copy()

    feature_columns = [
        *BASE_FEATURE_COLUMNS,
        *[
            column
            for column
            in OPTIONAL_NEIGHBOR_FEATURE_COLUMNS
            if column in holdout.columns
        ],
        *[
            column
            for column
            in OPTIONAL_SPECTRUM_FINGERPRINT_COLUMNS
            if column in holdout.columns
        ],
    ]

    booster = lgb.Booster(
        model_file=str(
            MODEL_PATH
        )
    )

    x = (
        holdout[
            feature_columns
        ]
        .copy()
    )

    for column in [
        "in_full_coconut",
        "in_approved_coconut",
    ]:
        x[
            column
        ] = (
            x[
                column
            ]
            .astype(np.int8)
        )

    x = x.replace(
        [
            np.inf,
            -np.inf,
        ],
        np.nan,
    )

    holdout[
        "best_lgbm_score"
    ] = booster.predict(
        x.to_numpy(
            dtype=np.float32
        )
    )

    holdout = add_rank(
        holdout,
        score_column="best_lgbm_score",
        rank_column="best_lgbm_rank",
        ascending=False,
    )

    scaled = pd.read_parquet(
        SCALED_FP_PATH
    )

    key_columns = [
        "inchikey14",
        "candidate_formula",
        "candidate_inchikey14",
    ]

    holdout = holdout.merge(
        scaled[
            [
                *key_columns,
                *SCALED_FEATURES,
            ]
        ],
        on=key_columns,
        how="left",
        validate="one_to_one",
        suffixes=(
            "",
            "_scaled",
        ),
    )

    # If canonical feature table already contains prototype columns,
    # the merge suffix gives us explicit scaled names.
    scaled_columns = {}

    for feature in SCALED_FEATURES:
        candidate = (
            f"{feature}_scaled"
            if f"{feature}_scaled" in holdout.columns
            else feature
        )

        scaled_columns[
            feature
        ] = candidate

    query_table = (
        holdout[
            [
                "inchikey14",
                "truth_present",
            ]
        ]
        .drop_duplicates(
            subset=[
                "inchikey14"
            ]
        )
        .sort_values(
            "inchikey14"
        )
        .reset_index(
            drop=True
        )
    )

    results = []

    baseline_ranks = truth_ranks(
        holdout,
        "best_lgbm_rank",
        query_table[
            "inchikey14"
        ],
    )

    results.append(
        summarize(
            "best_prototype_lgbm",
            baseline_ranks,
        )
    )

    for feature, column in scaled_columns.items():
        ranked = add_rank(
            holdout,
            score_column=column,
            rank_column="_scaled_rank",
            ascending=False,
        )

        fp_truth_ranks = truth_ranks(
            ranked,
            "_scaled_rank",
            query_table[
                "inchikey14"
            ],
        )

        results.append(
            summarize(
                f"scaled_{feature}",
                fp_truth_ranks,
            )
        )

        fusion_base = (
            ranked[
                [
                    "inchikey14",
                    "candidate_inchikey14",
                    "is_truth",
                    "best_lgbm_rank",
                    "_scaled_rank",
                ]
            ]
            .copy()
        )

        for weight in FUSION_WEIGHTS:
            fused = fusion_base.copy()

            fused[
                "_fusion_score"
            ] = (
                fused[
                    "best_lgbm_rank"
                ].astype(float)
                + weight
                * fused[
                    "_scaled_rank"
                ].astype(float)
            )

            fused = (
                fused
                .sort_values(
                    [
                        "inchikey14",
                        "_fusion_score",
                        "best_lgbm_rank",
                        "_scaled_rank",
                        "candidate_inchikey14",
                    ],
                    ascending=[
                        True,
                        True,
                        True,
                        True,
                        True,
                    ],
                    kind="stable",
                )
                .reset_index(
                    drop=True
                )
            )

            fused[
                "_fusion_rank"
            ] = (
                fused.groupby(
                    "inchikey14",
                    sort=False,
                )
                .cumcount()
                + 1
            )

            ranks = truth_ranks(
                fused,
                "_fusion_rank",
                query_table[
                    "inchikey14"
                ],
            )

            results.append(
                summarize(
                    (
                        f"rank_fusion_{feature}_"
                        f"weight_{weight:g}"
                    ),
                    ranks,
                )
            )

    summary = (
        pd.DataFrame(
            results
        )
        .sort_values(
            [
                "mrr25",
                "top1",
                "top25",
            ],
            ascending=[
                False,
                False,
                False,
            ],
        )
        .reset_index(
            drop=True
        )
    )

    summary.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print()
    print(
        "Prototype LightGBM + scaled fingerprint rank fusion"
    )
    print(
        "=" * 88
    )
    print(
        "Holdout queries:",
        f"{len(query_table):,}",
    )
    print()
    print(
        summary.head(
            20
        ).to_string(
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
                "truth_returned":
                    lambda value:
                    f"{value:.2%}",
            },
        )
    )
    print()
    print(
        "Wrote:",
        OUTPUT_PATH,
    )


if __name__ == "__main__":
    main()
