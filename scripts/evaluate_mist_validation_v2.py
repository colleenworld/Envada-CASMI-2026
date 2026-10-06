from pathlib import Path
import pickle

import numpy as np
import pandas as pd


ROOT = Path(
    "/home/colleen/PycharmProjects/Envada-CASMI-2026/"
    "data/processed/mist/validation_500_v2"
)

PRED_PATH = (
    ROOT
    / "mist_predictions/fp_preds/"
    / "fp_preds_casmi_validation_500_v2_multispectrum.p"
)

CANDIDATE_FP_PATH = (
    ROOT / "mist_structure_candidate_fingerprints.npz"
)

CANDIDATE_META_PATH = (
    ROOT / "mist_structure_candidate_fingerprint_metadata.parquet"
)

MANIFEST_PATH = ROOT / "manifest.parquet"

OUTPUT_RANKED_PATH = (
    ROOT / "mist_ranked_candidates.parquet"
)

MASS_PENALTY = 0.010
TOP_K = 25


def cosine_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Row-wise cosine similarity between arrays with identical shape.
    """
    dot = np.sum(a * b, axis=1)

    a_norm = np.linalg.norm(a, axis=1)
    b_norm = np.linalg.norm(b, axis=1)

    denom = a_norm * b_norm

    out = np.zeros(
        len(a),
        dtype=np.float32,
    )

    valid = denom > 0

    out[valid] = (
        dot[valid] / denom[valid]
    )

    return out


def reciprocal_rank(rank):
    if rank is None:
        return 0.0

    if rank > TOP_K:
        return 0.0

    return 1.0 / rank


def main():
    #
    # Load MIST predictions.
    #
    with PRED_PATH.open("rb") as f:
        pred_obj = pickle.load(f)

    pred_names = list(
        pred_obj["names"]
    )

    pred_matrix = np.asarray(
        pred_obj["preds"],
        dtype=np.float32,
    )

    if pred_matrix.shape[1] != 4096:
        raise ValueError(
            f"Unexpected MIST prediction shape: "
            f"{pred_matrix.shape}"
        )

    if len(pred_names) != len(pred_matrix):
        raise ValueError(
            "Prediction names and prediction rows differ."
        )

    if len(set(pred_names)) != len(pred_names):
        raise ValueError(
            "Duplicate prediction names found."
        )

    pred_index = {
        name: i
        for i, name in enumerate(
            pred_names
        )
    }

    #
    # Load candidate fingerprints + metadata.
    #
    candidate_npz = np.load(
        CANDIDATE_FP_PATH
    )

    candidate_fp = candidate_npz[
        "fingerprints"
    ].astype(
        np.float32,
        copy=False,
    )

    candidates = pd.read_parquet(
        CANDIDATE_META_PATH
    )

    if len(candidates) != len(candidate_fp):
        raise ValueError(
            "Candidate metadata and fingerprint "
            "matrix row counts differ."
        )

    if candidate_fp.shape[1] != 4096:
        raise ValueError(
            f"Unexpected candidate fingerprint "
            f"shape: {candidate_fp.shape}"
        )

    manifest = pd.read_parquet(
        MANIFEST_PATH
    )

    all_queries = manifest[
        "query_inchikey14"
    ].tolist()

    #
    # Map every candidate row to its hypothesis prediction.
    #
    missing_predictions = sorted(
        set(
            candidates[
                "hypothesis_spec"
            ]
        )
        - set(pred_index)
    )

    if missing_predictions:
        raise ValueError(
            "Missing MIST predictions for "
            f"{len(missing_predictions)} hypotheses. "
            f"Examples: {missing_predictions[:10]}"
        )

    pred_rows = np.fromiter(
        (
            pred_index[name]
            for name in candidates[
                "hypothesis_spec"
            ]
        ),
        dtype=np.int64,
        count=len(candidates),
    )

    query_pred_fp = pred_matrix[
        pred_rows
    ]

    #
    # Raw MIST cosine similarity.
    #
    candidates[
        "mist_cosine"
    ] = cosine_rows(
        query_pred_fp,
        candidate_fp,
    )

    #
    # Frozen cross-formula mass penalty.
    #
    candidates[
        "mass_penalty"
    ] = (
        MASS_PENALTY
        * candidates[
            "selected_mass_error_ppm"
        ].abs()
    )

    candidates[
        "score"
    ] = (
        candidates[
            "mist_cosine"
        ]
        - candidates[
            "mass_penalty"
        ]
    )

    #
    # A structure can theoretically be reached through more than
    # one formula hypothesis. Collapse to one row per
    # query/structure using the best score.
    #
    candidates = candidates.sort_values(
        by=[
            "query_inchikey14",
            "candidate_inchikey14",
            "score",
            "mist_cosine",
        ],
        ascending=[
            True,
            True,
            False,
            False,
        ],
    )

    collapsed = candidates.drop_duplicates(
        subset=[
            "query_inchikey14",
            "candidate_inchikey14",
        ],
        keep="first",
    ).copy()

    #
    # Rank structures within each query.
    #
    collapsed = collapsed.sort_values(
        by=[
            "query_inchikey14",
            "score",
            "mist_cosine",
            "candidate_inchikey14",
        ],
        ascending=[
            True,
            False,
            False,
            True,
        ],
    )

    collapsed["rank"] = (
        collapsed.groupby(
            "query_inchikey14"
        )
        .cumcount()
        + 1
    )

    collapsed.to_parquet(
        OUTPUT_RANKED_PATH,
        index=False,
    )

    #
    # Evaluate across ALL 500 queries.
    #
    truth_rows = collapsed[
        collapsed[
            "is_truth_structure"
        ]
    ][
        [
            "query_inchikey14",
            "rank",
            "score",
            "mist_cosine",
            "selected_mass_error_ppm",
        ]
    ].copy()

    if truth_rows[
        "query_inchikey14"
    ].duplicated().any():
        raise ValueError(
            "More than one truth structure row "
            "for a query after collapsing."
        )

    truth_rank = dict(
        zip(
            truth_rows[
                "query_inchikey14"
            ],
            truth_rows[
                "rank"
            ],
        )
    )

    ranks = [
        int(truth_rank[q])
        if q in truth_rank
        else None
        for q in all_queries
    ]

    rr = np.array(
        [
            reciprocal_rank(rank)
            for rank in ranks
        ],
        dtype=float,
    )

    def topk(k):
        return np.mean(
            [
                rank is not None
                and rank <= k
                for rank in ranks
            ]
        )

    rankable_ranks = [
        rank
        for rank in ranks
        if rank is not None
    ]

    print("\nFrozen MIST multi-spectrum validation v2")
    print("----------------------------------------")
    print("Queries evaluated:", len(all_queries))
    print(
        "Truth structures ranked:",
        len(rankable_ranks),
        "/",
        len(all_queries),
    )

    print(
        f"MRR@{TOP_K}:",
        f"{rr.mean():.4f}",
    )

    print(
        "Top-1 accuracy:",
        f"{topk(1) * 100:.2f}%",
    )
    print(
        "Top-5 accuracy:",
        f"{topk(5) * 100:.2f}%",
    )
    print(
        "Top-10 accuracy:",
        f"{topk(10) * 100:.2f}%",
    )
    print(
        "Top-25 recall:",
        f"{topk(25) * 100:.2f}%",
    )

    if rankable_ranks:
        rank_array = np.asarray(
            rankable_ranks,
            dtype=float,
        )

        print("\nTruth-rank distribution")
        print("-----------------------")
        print(
            "Median:",
            float(
                np.median(
                    rank_array
                )
            ),
        )
        print(
            "Mean:",
            float(
                np.mean(
                    rank_array
                )
            ),
        )
        print(
            "P75:",
            float(
                np.percentile(
                    rank_array,
                    75,
                )
            ),
        )
        print(
            "P90:",
            float(
                np.percentile(
                    rank_array,
                    90,
                )
            ),
        )
        print(
            "P95:",
            float(
                np.percentile(
                    rank_array,
                    95,
                )
            ),
        )
        print(
            "Max:",
            int(
                np.max(
                    rank_array
                )
            ),
        )

        conditional_rr = np.array(
            [
                reciprocal_rank(rank)
                for rank in rankable_ranks
            ],
            dtype=float,
        )

        print(
            "\nConditional on truth being rankable"
        )
        print(
            "MRR@25:",
            f"{conditional_rr.mean():.4f}",
        )

        print(
            "Top-1:",
            f"{np.mean(np.array(rankable_ranks) <= 1) * 100:.2f}%",
        )

        print(
            "Top-25:",
            f"{np.mean(np.array(rankable_ranks) <= 25) * 100:.2f}%",
        )

    print(
        f"\nWrote {OUTPUT_RANKED_PATH}"
    )


if __name__ == "__main__":
    main()