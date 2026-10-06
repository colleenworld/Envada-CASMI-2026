from pathlib import Path
import pickle

import numpy as np
import pandas as pd


ROOT = Path(
    "data/processed/mist/validation_500"
)

RANKED_PATH = (
    ROOT
    / "mist_validation_ranked_candidates.parquet"
)

PRED_PATH = (
    ROOT
    / "mist_predictions/fp_preds"
    / "fp_preds_casmi_validation_500.p"
)

CANDIDATE_FP_PATH = (
    ROOT
    / "mist_validation_candidate_fingerprints.npz"
)

CANDIDATE_META_PATH = (
    ROOT
    / "mist_validation_candidate_fingerprint_metadata.parquet"
)

TOP_K = 25


def clamped_log_np(
    x: np.ndarray,
    floor: float = -5.0,
) -> np.ndarray:
    """
    Match MIST's utils.clamped_log_np behaviour closely enough
    for retrieval_fp.py BCE scoring.

    MIST then sums the per-bit loss across the fingerprint.
    """
    with np.errstate(
        divide="ignore",
        invalid="ignore",
    ):
        logged = np.log(x)

    return np.maximum(
        logged,
        floor,
    )


def bce_distance(
    pred: np.ndarray,
    targ: np.ndarray,
) -> np.ndarray:
    """
    Reproduce MIST retrieval_fp.py BCE distance.

    MIST:

        one_term = targ * clamped_log_np(pred, -5)
        zero_term = (1 - targ) * clamped_log_np(1 - pred, -5)
        return -(one_term + zero_term)

    rank_indices() then does:

        dist = dist_fn(...).sum(-1)

    Lower is better.
    """
    one_term = (
        targ
        * clamped_log_np(
            pred,
            -5,
        )
    )

    zero_term = (
        (1.0 - targ)
        * clamped_log_np(
            1.0 - pred,
            -5,
        )
    )

    return -(
        one_term
        + zero_term
    ).sum(
        axis=1
    )


def cosine_distance(
    pred: np.ndarray,
    targ: np.ndarray,
) -> np.ndarray:
    """
    Reproduce MIST retrieval_fp.py cosine distance.

    MIST returns:

        1 - cosine_similarity

    Lower is better.
    """
    numerator = (
        pred
        * targ
    ).sum(
        axis=1
    )

    pred_norm = np.linalg.norm(
        pred,
        axis=1,
    )

    targ_norm = np.linalg.norm(
        targ,
        axis=1,
    )

    denom = (
        pred_norm
        * targ_norm
    )

    result = np.full(
        len(pred),
        np.inf,
        dtype=np.float32,
    )

    valid = denom > 0

    result[valid] = (
        1.0
        - (
            numerator[valid]
            / denom[valid]
        )
    )

    return result


def l1_distance(
    pred: np.ndarray,
    targ: np.ndarray,
) -> np.ndarray:
    """
    Match MIST's L1 retrieval distance:

        abs(pred - targ)

    followed by sum(-1).

    Lower is better.
    """
    return np.abs(
        pred
        - targ
    ).sum(
        axis=1
    )


def l2_distance(
    pred: np.ndarray,
    targ: np.ndarray,
) -> np.ndarray:
    """
    Match MIST's L2 retrieval distance:

        square(pred - targ)

    followed by sum(-1).

    Lower is better.
    """
    return np.square(
        pred
        - targ
    ).sum(
        axis=1
    )


def reciprocal_rank(
    rank,
) -> float:
    if rank is None:
        return 0.0

    if rank > TOP_K:
        return 0.0

    return 1.0 / rank


def evaluate(
    candidates: pd.DataFrame,
    score_col: str,
    ascending: bool,
    label: str,
):
    """
    Evaluate one distance function on the same candidate universe.

    This function assumes candidates have already been restricted
    to the true molecular formula for same-formula analysis.
    """
    ranked = candidates.sort_values(
        by=[
            "query_inchikey14",
            score_col,
            "candidate_inchikey14",
        ],
        ascending=[
            True,
            ascending,
            True,
        ],
    ).copy()

    ranked[
        "eval_rank"
    ] = (
        ranked.groupby(
            "query_inchikey14"
        )
        .cumcount()
        + 1
    )

    truth = ranked[
        ranked[
            "is_truth"
        ]
    ][
        [
            "query_inchikey14",
            "eval_rank",
        ]
    ].copy()

    if truth[
        "query_inchikey14"
    ].duplicated().any():
        raise ValueError(
            f"{label}: more than one truth row "
            "for a query."
        )

    truth_rank = dict(
        zip(
            truth[
                "query_inchikey14"
            ],
            truth[
                "eval_rank"
            ],
        )
    )

    all_queries = sorted(
        candidates[
            "query_inchikey14"
        ].unique()
    )

    ranks = [
        int(
            truth_rank[query]
        )
        if query in truth_rank
        else None
        for query in all_queries
    ]

    rr = np.asarray(
        [
            reciprocal_rank(
                rank
            )
            for rank in ranks
        ],
        dtype=float,
    )

    def topk(k: int) -> float:
        return np.mean(
            [
                rank is not None
                and rank <= k
                for rank in ranks
            ]
        )

    rankable = [
        rank
        for rank in ranks
        if rank is not None
    ]

    print()
    print(label)
    print(
        "-" * len(label)
    )

    print(
        "Queries:",
        len(all_queries),
    )

    print(
        "Truth rankable:",
        len(rankable),
    )

    print(
        f"MRR@{TOP_K}:",
        f"{rr.mean():.4f}",
    )

    print(
        "Top-1:",
        f"{topk(1) * 100:.2f}%",
    )

    print(
        "Top-5:",
        f"{topk(5) * 100:.2f}%",
    )

    print(
        "Top-10:",
        f"{topk(10) * 100:.2f}%",
    )

    print(
        "Top-25:",
        f"{topk(25) * 100:.2f}%",
    )

    if rankable:
        rank_array = np.asarray(
            rankable,
            dtype=float,
        )

        conditional_rr = np.asarray(
            [
                reciprocal_rank(
                    rank
                )
                for rank in rankable
            ],
            dtype=float,
        )

        print(
            "Conditional MRR@25:",
            f"{conditional_rr.mean():.4f}",
        )

        print(
            "Conditional Top-1:",
            f"{np.mean(rank_array <= 1) * 100:.2f}%",
        )

        print(
            "Conditional Top-5:",
            f"{np.mean(rank_array <= 5) * 100:.2f}%",
        )

        print(
            "Conditional Top-10:",
            f"{np.mean(rank_array <= 10) * 100:.2f}%",
        )

        print(
            "Conditional Top-25:",
            f"{np.mean(rank_array <= 25) * 100:.2f}%",
        )

        print(
            "Median truth rank:",
            float(
                np.median(
                    rank_array
                )
            ),
        )

        print(
            "Mean truth rank:",
            float(
                np.mean(
                    rank_array
                )
            ),
        )

        print(
            "P90 truth rank:",
            float(
                np.percentile(
                    rank_array,
                    90,
                )
            ),
        )

        print(
            "P95 truth rank:",
            float(
                np.percentile(
                    rank_array,
                    95,
                )
            ),
        )

        print(
            "Max truth rank:",
            int(
                np.max(
                    rank_array
                )
            ),
        )

    return {
        "label":
            label,

        "queries":
            len(all_queries),

        "rankable":
            len(rankable),

        "mrr25":
            float(
                rr.mean()
            ),

        "top1":
            float(
                topk(1)
            ),

        "top5":
            float(
                topk(5)
            ),

        "top10":
            float(
                topk(10)
            ),

        "top25":
            float(
                topk(25)
            ),

        "median_rank":
            float(
                np.median(
                    rankable
                )
            )
            if rankable
            else np.nan,
    }


def main():
    #
    # Existing v1 ranked candidate metadata.
    #
    ranked = pd.read_parquet(
        RANKED_PATH
    )

    print(
        "Ranked candidate rows:",
        len(ranked),
    )

    required_ranked = {
        "query_inchikey14",
        "hypothesis_spec",
        "candidate_formula",
        "candidate_inchikey14",
        "is_truth",
    }

    missing_ranked = (
        required_ranked
        - set(
            ranked.columns
        )
    )

    if missing_ranked:
        raise ValueError(
            "Ranked candidate file missing columns: "
            f"{sorted(missing_ranked)}"
        )

    #
    # Load MIST predictions.
    #
    with PRED_PATH.open(
        "rb"
    ) as f:
        pred_obj = pickle.load(
            f
        )

    pred_names = list(
        pred_obj[
            "names"
        ]
    )

    pred_matrix = np.asarray(
        pred_obj[
            "preds"
        ],
        dtype=np.float32,
    )

    print(
        "MIST predictions:",
        pred_matrix.shape,
    )

    if (
        pred_matrix.ndim != 2
        or pred_matrix.shape[1]
        != 4096
    ):
        raise ValueError(
            "Unexpected MIST prediction shape: "
            f"{pred_matrix.shape}"
        )

    if len(
        pred_names
    ) != len(
        pred_matrix
    ):
        raise ValueError(
            "Prediction-name count does not "
            "match prediction matrix."
        )

    if len(
        set(
            pred_names
        )
    ) != len(
        pred_names
    ):
        raise ValueError(
            "Duplicate MIST prediction names."
        )

    pred_index = {
        name: i
        for i, name
        in enumerate(
            pred_names
        )
    }

    #
    # Load candidate fingerprints.
    #
    npz = np.load(
        CANDIDATE_FP_PATH
    )

    candidate_fp = np.asarray(
        npz[
            "fingerprints"
        ],
        dtype=np.float32,
    )

    candidate_meta = pd.read_parquet(
        CANDIDATE_META_PATH
    )

    print(
        "Candidate FP matrix:",
        candidate_fp.shape,
    )

    if (
        candidate_fp.ndim != 2
        or candidate_fp.shape[1]
        != 4096
    ):
        raise ValueError(
            "Unexpected candidate fingerprint "
            f"shape: {candidate_fp.shape}"
        )

    if len(
        candidate_meta
    ) != len(
        candidate_fp
    ):
        raise ValueError(
            "Candidate metadata and fingerprint "
            "matrix row counts differ."
        )

    #
    # The fingerprint metadata was saved in exactly the same row
    # order as the candidate fingerprint matrix.
    #
    candidate_meta = (
        candidate_meta
        .reset_index(
            drop=True
        )
        .reset_index()
        .rename(
            columns={
                "index":
                    "candidate_fp_row"
            }
        )
    )

    join_cols = [
        "hypothesis_spec",
        "candidate_inchikey14",
    ]

    #
    # Confirm candidate metadata has unique hypothesis/structure
    # pairs before joining.
    #
    if candidate_meta.duplicated(
        subset=join_cols
    ).any():
        raise ValueError(
            "Duplicate hypothesis/structure rows "
            "in candidate fingerprint metadata."
        )

    if ranked.duplicated(
        subset=join_cols
    ).any():
        raise ValueError(
            "Duplicate hypothesis/structure rows "
            "in ranked candidate metadata."
        )

    ranked = ranked.merge(
        candidate_meta[
            join_cols
            + [
                "candidate_fp_row"
            ]
        ],
        on=join_cols,
        how="left",
        validate="one_to_one",
    )

    if ranked[
        "candidate_fp_row"
    ].isna().any():
        missing_count = int(
            ranked[
                "candidate_fp_row"
            ]
            .isna()
            .sum()
        )

        raise ValueError(
            "Missing candidate fingerprints "
            f"after join: {missing_count}"
        )

    #
    # Ensure every formula hypothesis has a MIST prediction.
    #
    missing_predictions = sorted(
        set(
            ranked[
                "hypothesis_spec"
            ]
        )
        - set(
            pred_index
        )
    )

    if missing_predictions:
        raise ValueError(
            "Missing MIST predictions for "
            f"{len(missing_predictions)} hypotheses. "
            f"Examples: "
            f"{missing_predictions[:10]}"
        )

    pred_rows = np.asarray(
        [
            pred_index[
                name
            ]
            for name in ranked[
                "hypothesis_spec"
            ]
        ],
        dtype=np.int64,
    )

    fp_rows = (
        ranked[
            "candidate_fp_row"
        ]
        .astype(
            np.int64
        )
        .to_numpy()
    )

    pred = pred_matrix[
        pred_rows
    ]

    targ = candidate_fp[
        fp_rows
    ]

    print(
        "Scoring rows:",
        len(pred),
    )

    #
    # Compute all four retrieval distances supported by MIST's
    # retrieval_fp.py.
    #
    ranked[
        "cosine_dist"
    ] = cosine_distance(
        pred,
        targ,
    )

    ranked[
        "bce_dist"
    ] = bce_distance(
        pred,
        targ,
    )

    ranked[
        "l1_dist"
    ] = l1_distance(
        pred,
        targ,
    )

    ranked[
        "l2_dist"
    ] = l2_distance(
        pred,
        targ,
    )

    #
    # Restrict to the true formula for each rankable query.
    #
    # This deliberately isolates STRUCTURE ranking from formula
    # ranking.
    #
    truth_formula_rows = (
        ranked.loc[
            ranked[
                "is_truth"
            ],
            [
                "query_inchikey14",
                "candidate_formula",
            ],
        ]
        .drop_duplicates()
    )

    if truth_formula_rows[
        "query_inchikey14"
    ].duplicated().any():
        raise ValueError(
            "A query has more than one truth formula."
        )

    truth_formula_by_query = dict(
        zip(
            truth_formula_rows[
                "query_inchikey14"
            ],
            truth_formula_rows[
                "candidate_formula"
            ],
        )
    )

    true_formula = ranked[
        "query_inchikey14"
    ].map(
        truth_formula_by_query
    )

    same_formula = ranked[
        ranked[
            "candidate_formula"
        ]
        == true_formula
    ].copy()

    print()
    print(
        "Same-formula candidates:",
        len(
            same_formula
        ),
    )

    print(
        "Same-formula queries:",
        same_formula[
            "query_inchikey14"
        ].nunique(),
    )

    #
    # Evaluate MIST-supported distances.
    #
    results = []

    results.append(
        evaluate(
            same_formula,
            score_col="cosine_dist",
            ascending=True,
            label="V1 same-formula cosine",
        )
    )

    results.append(
        evaluate(
            same_formula,
            score_col="bce_dist",
            ascending=True,
            label="V1 same-formula BCE",
        )
    )

    results.append(
        evaluate(
            same_formula,
            score_col="l1_dist",
            ascending=True,
            label="V1 same-formula L1",
        )
    )

    results.append(
        evaluate(
            same_formula,
            score_col="l2_dist",
            ascending=True,
            label="V1 same-formula L2",
        )
    )

    #
    # Compact comparison table.
    #
    result_df = pd.DataFrame(
        results
    )

    result_df = result_df.sort_values(
        by=[
            "mrr25",
            "top1",
        ],
        ascending=[
            False,
            False,
        ],
    )

    print()
    print(
        "Distance comparison"
    )
    print(
        "-------------------"
    )

    display = result_df.copy()

    for col in [
        "mrr25",
        "top1",
        "top5",
        "top10",
        "top25",
    ]:
        if col == "mrr25":
            display[col] = (
                display[col]
                .map(
                    lambda x:
                        f"{x:.4f}"
                )
            )
        else:
            display[col] = (
                display[col]
                .map(
                    lambda x:
                        f"{x * 100:.2f}%"
                )
            )

    print(
        display[
            [
                "label",
                "mrr25",
                "top1",
                "top5",
                "top10",
                "top25",
                "median_rank",
            ]
        ]
        .to_string(
            index=False
        )
    )

    best = result_df.iloc[0]

    print()
    print(
        "Best by MRR@25:",
        best[
            "label"
        ],
        f"({best['mrr25']:.4f})",
    )


if __name__ == "__main__":
    main()