from __future__ import annotations

from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd


ROOT = Path(
    "data/processed/diagnostics/large_validation_audit/"
    "true_formula_neighbor_benchmark/learned_ranker"
)

FEATURES_PATH = (
    ROOT
    / "candidate_features.parquet"
)

MODEL_PATH = (
    ROOT
    / "lgbm_ranker.txt"
)

PREDICTIONS_PATH = (
    ROOT
    / "holdout_predictions.parquet"
)

SUMMARY_PATH = (
    ROOT
    / "holdout_summary.csv"
)

SEED = 42

FEATURE_COLUMNS = [
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


def rr25(
    rank: float | int | None,
) -> float:
    if (
        rank is None
        or pd.isna(
            rank
        )
    ):
        return 0.0

    rank = int(
        rank
    )

    if (
        rank < 1
        or rank > 25
    ):
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
            len(
                ranks
            ),
        "mrr25":
            ranks.map(
                rr25
            ).mean(),
        "top1":
            ranks.eq(
                1
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
        "truth_returned":
            ranks.notna().mean(),
    }


def prepare_ranker_frame(
    frame: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    np.ndarray,
    np.ndarray,
]:
    ordered = (
        frame
        .sort_values(
            [
                "inchikey14",
                "candidate_inchikey14",
            ],
            kind="stable",
        )
        .reset_index(
            drop=True
        )
    )

    groups = (
        ordered.groupby(
            "inchikey14",
            sort=False,
        )
        .size()
        .to_numpy(
            dtype=np.int32
        )
    )

    x = (
        ordered[
            FEATURE_COLUMNS
        ]
        .copy()
    )

    for column in [
        "in_full_coconut",
        "in_approved_coconut",
    ]:
        x[
            column
        ] = x[
            column
        ].astype(
            np.int8
        )

    x = x.replace(
        [
            np.inf,
            -np.inf,
        ],
        np.nan,
    )

    y = (
        ordered[
            "is_truth"
        ]
        .astype(
            np.int8
        )
        .to_numpy()
    )

    return (
        ordered,
        x.to_numpy(
            dtype=np.float32
        ),
        groups,
    )


def truth_ranks(
    ranked: pd.DataFrame,
    rank_column: str,
    all_query_keys: pd.Series,
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
        all_query_keys
        .astype(str)
        .map(
            truth
        )
    )


def rank_learned(
    frame: pd.DataFrame,
    scores: np.ndarray,
) -> pd.DataFrame:
    ranked = frame.copy()

    ranked[
        "learned_score"
    ] = scores

    ranked = (
        ranked
        .sort_values(
            [
                "inchikey14",
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
        "learned_rank"
    ] = (
        ranked.groupby(
            "inchikey14",
            sort=False,
        )
        .cumcount()
        + 1
    )

    return ranked


def rank_penalty_3(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    ranked = frame.copy()

    ranked[
        "_rank_penalty_score"
    ] = (
        ranked[
            "prior_rank"
        ].astype(float)
        + 3.0
        * (
            ranked[
                "formula_rank"
            ].astype(float)
            - 1.0
        )
    )

    ranked = (
        ranked
        .sort_values(
            [
                "inchikey14",
                "_rank_penalty_score",
                "formula_rank",
                "prior_rank",
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
        .drop_duplicates(
            subset=[
                "inchikey14",
                "candidate_inchikey14",
            ],
            keep="first",
        )
        .reset_index(
            drop=True
        )
    )

    ranked[
        "baseline_rank"
    ] = (
        ranked.groupby(
            "inchikey14",
            sort=False,
        )
        .cumcount()
        + 1
    )

    return ranked


def main() -> None:
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(
            "Missing learned-ranker dataset. Run "
            "scripts/build_learned_ranker_dataset.py first."
        )

    frame = pd.read_parquet(
        FEATURES_PATH
    )

    required = {
        "inchikey14",
        "candidate_inchikey14",
        "is_truth",
        "split",
        "truth_present",
        *FEATURE_COLUMNS,
    }

    missing = (
        required
        - set(
            frame.columns
        )
    )

    if missing:
        raise ValueError(
            "Ranker dataset is missing columns: "
            f"{sorted(missing)}"
        )

    train_queries = (
        frame.loc[
            (
                frame[
                    "split"
                ]
                == "train"
            )
            & frame[
                "truth_present"
            ].astype(bool),
            "inchikey14",
        ]
        .drop_duplicates()
        .sort_values()
        .to_numpy(
            dtype=object
        )
    )

    if len(
        train_queries
    ) < 20:
        raise RuntimeError(
            "Too few training queries with retained truth"
        )

    rng = np.random.default_rng(
        SEED
    )

    shuffled = train_queries.copy()

    rng.shuffle(
        shuffled
    )

    validation_count = max(
        1,
        int(
            round(
                len(
                    shuffled
                )
                * 0.15
            )
        ),
    )

    validation_keys = set(
        shuffled[
            :validation_count
        ]
    )

    fit_keys = set(
        shuffled[
            validation_count:
        ]
    )

    fit_frame = frame.loc[
        frame[
            "inchikey14"
        ]
        .isin(
            fit_keys
        )
    ].copy()

    validation_frame = frame.loc[
        frame[
            "inchikey14"
        ]
        .isin(
            validation_keys
        )
    ].copy()

    holdout_frame = frame.loc[
        frame[
            "split"
        ]
        == "holdout"
    ].copy()

    (
        fit_ordered,
        fit_x,
        fit_groups,
    ) = prepare_ranker_frame(
        fit_frame
    )

    (
        validation_ordered,
        validation_x,
        validation_groups,
    ) = prepare_ranker_frame(
        validation_frame
    )

    (
        holdout_ordered,
        holdout_x,
        _,
    ) = prepare_ranker_frame(
        holdout_frame
    )

    fit_y = (
        fit_ordered[
            "is_truth"
        ]
        .astype(
            np.int8
        )
        .to_numpy()
    )

    validation_y = (
        validation_ordered[
            "is_truth"
        ]
        .astype(
            np.int8
        )
        .to_numpy()
    )

    model = lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        ndcg_at=[
            1,
            5,
            10,
            25,
        ],
        learning_rate=0.03,
        n_estimators=2000,
        num_leaves=31,
        min_child_samples=20,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=SEED,
        n_jobs=-1,
        verbosity=-1,
    )

    model.fit(
        fit_x,
        fit_y,
        group=fit_groups,
        eval_set=[
            (
                validation_x,
                validation_y,
            )
        ],
        eval_group=[
            validation_groups
        ],
        eval_at=[
            1,
            5,
            10,
            25,
        ],
        callbacks=[
            lgb.early_stopping(
                100,
                verbose=True,
            ),
            lgb.log_evaluation(
                50
            ),
        ],
        feature_name=FEATURE_COLUMNS,
    )

    booster = model.booster_

    if booster is None:
        raise RuntimeError(
            "LightGBM did not produce a booster"
        )

    booster.save_model(
        str(
            MODEL_PATH
        )
    )

    scores = model.predict(
        holdout_x,
        num_iteration=(
            model.best_iteration_
            if model.best_iteration_
            else None
        ),
    )

    learned = rank_learned(
        holdout_ordered,
        np.asarray(
            scores,
            dtype=float,
        ),
    )

    baseline = rank_penalty_3(
        holdout_ordered
    )

    query_table = (
        holdout_ordered[
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

    learned_ranks = truth_ranks(
        learned,
        "learned_rank",
        query_table[
            "inchikey14"
        ],
    )

    baseline_ranks = truth_ranks(
        baseline,
        "baseline_rank",
        query_table[
            "inchikey14"
        ],
    )

    results = query_table.copy()

    results[
        "learned_rank"
    ] = learned_ranks.to_numpy()

    results[
        "baseline_rank"
    ] = baseline_ranks.to_numpy()

    results.to_parquet(
        PREDICTIONS_PATH,
        index=False,
        compression="zstd",
    )

    summary_rows = []

    summary_rows.append(
        summarize(
            "rank_penalty_3_all_holdout",
            results[
                "baseline_rank"
            ],
        )
    )

    summary_rows.append(
        summarize(
            "lightgbm_all_holdout",
            results[
                "learned_rank"
            ],
        )
    )

    truth_present = results.loc[
        results[
            "truth_present"
        ].astype(bool)
    ]

    summary_rows.append(
        summarize(
            "rank_penalty_3_truth_present",
            truth_present[
                "baseline_rank"
            ],
        )
    )

    summary_rows.append(
        summarize(
            "lightgbm_truth_present",
            truth_present[
                "learned_rank"
            ],
        )
    )

    summary = pd.DataFrame(
        summary_rows
    )

    summary.to_csv(
        SUMMARY_PATH,
        index=False,
    )

    print()
    print(
        "Learned candidate ranker"
    )
    print(
        "=" * 80
    )
    print(
        "Fit queries:",
        f"{len(fit_keys):,}",
    )
    print(
        "Early-stop validation queries:",
        f"{len(validation_keys):,}",
    )
    print(
        "Holdout queries:",
        f"{len(results):,}",
    )
    print(
        "Holdout queries with retained truth:",
        f"{int(results['truth_present'].sum()):,}",
    )
    print(
        "Best iteration:",
        model.best_iteration_,
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
                "truth_returned":
                    lambda value:
                    f"{value:.2%}",
            },
        )
    )

    importance = pd.DataFrame(
        {
            "feature":
                FEATURE_COLUMNS,
            "gain":
                booster.feature_importance(
                    importance_type="gain"
                ),
            "splits":
                booster.feature_importance(
                    importance_type="split"
                ),
        }
    ).sort_values(
        "gain",
        ascending=False,
    )

    print()
    print(
        "Feature importance"
    )
    print(
        "-" * 80
    )
    print(
        importance.to_string(
            index=False
        )
    )
    print()
    print(
        "Wrote:",
        MODEL_PATH,
    )
    print(
        "Wrote:",
        PREDICTIONS_PATH,
    )
    print(
        "Wrote:",
        SUMMARY_PATH,
    )


if __name__ == "__main__":
    main()
