from pathlib import Path
import pickle

import numpy as np
import pandas as pd


ROOT = Path("data/processed/mist/validation_500_v2")

RANKED_PATH = ROOT / "mist_ranked_candidates.parquet"

CONTRASTIVE_PATH = (
    ROOT
    / "contrastive_predictions"
    / "retrieval_contrastive_casmi_validation_500_v2_morgan4096_"
      "with_morgan4096_retrieval_db_"
      "casmi_validation_500_v2_contrastive_cosine.p"
)

OUTPUT_PATH = (
    ROOT
    / "contrastive_multiformula_ranked_candidates.parquet"
)

LAMBDA = 0.020
TOP_K = 25
TOTAL_QUERIES = 500


def to_str(value):
    if isinstance(value, bytes):
        return value.decode()
    return str(value)


def reciprocal_rank(rank):
    if rank is None or rank > TOP_K:
        return 0.0
    return 1.0 / rank


def metrics(ranks):
    rr = np.asarray(
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

    rankable = [
        rank
        for rank in ranks
        if rank is not None
    ]

    return {
        "mrr25": float(rr.mean()),
        "top1": float(topk(1)),
        "top5": float(topk(5)),
        "top10": float(topk(10)),
        "top25": float(topk(25)),
        "median_rank":
            float(np.median(rankable))
            if rankable else np.nan,
        "mean_rank":
            float(np.mean(rankable))
            if rankable else np.nan,
        "p90_rank":
            float(np.percentile(rankable, 90))
            if rankable else np.nan,
        "p95_rank":
            float(np.percentile(rankable, 95))
            if rankable else np.nan,
        "max_rank":
            int(np.max(rankable))
            if rankable else None,
    }


def main():
    ranked = pd.read_parquet(RANKED_PATH)

    required = {
    "query_inchikey14",
    "hypothesis_spec",
    "candidate_formula",
    "candidate_inchikey14",
    "selected_mass_error_ppm",
    }

    missing = required - set(ranked.columns)

    if missing:
        raise ValueError(
            f"Missing columns: {sorted(missing)}"
        )

    with CONTRASTIVE_PATH.open("rb") as f:
        contrastive = pickle.load(f)

    names = [
        to_str(x)
        for x in contrastive["names"]
    ]

    ikey_lists = [
        [to_str(x) for x in row]
        for row in contrastive["ikeys"]
    ]

    dist_lists = [
        np.asarray(row, dtype=float)
        for row in contrastive["dists"]
    ]

    contrastive_by_name = {
        name: {
            "ikeys": ikeys,
            "dists": dists,
        }
        for name, ikeys, dists
        in zip(
            names,
            ikey_lists,
            dist_lists,
        )
    }

    rows = []

    for hypothesis, group in ranked.groupby(
        "hypothesis_spec",
        sort=False,
    ):
        retrieval = contrastive_by_name.get(
            hypothesis
        )

        if retrieval is None:
            continue

        dist_by_ikey = {
            ikey: float(dist)
            for ikey, dist
            in zip(
                retrieval["ikeys"],
                retrieval["dists"],
            )
        }

        for row in group.itertuples(
            index=False
        ):
            distance = dist_by_ikey.get(
                row.candidate_inchikey14
            )

            if distance is None:
                continue

            rows.append(
                {
                    "query_inchikey14":
                        row.query_inchikey14,

                    "hypothesis_spec":
                        row.hypothesis_spec,

                    "candidate_formula":
                        row.candidate_formula,

                    "candidate_inchikey14":
                        row.candidate_inchikey14,

                    "formula_mass_error_ppm":
                        float(
                            row.selected_mass_error_ppm
                        ),
                    "contrastive_distance":
                        distance,

                    "is_truth":
                        (
                            row.candidate_inchikey14
                            == row.query_inchikey14
                        ),
                }
            )

    df = pd.DataFrame(rows)

    print(
        "Candidate rows:",
        len(df),
    )

    print(
        "Queries with candidates:",
        df["query_inchikey14"].nunique(),
    )

    print(
        "Truth-containing queries:",
        df.loc[
            df["is_truth"],
            "query_inchikey14",
        ].nunique(),
    )

    #
    # Frozen v1-selected scoring rule.
    #
    df["final_score"] = (
        -df["contrastive_distance"]
        - LAMBDA
        * df["formula_mass_error_ppm"].abs()
    )

    #
    # A structure can occur through multiple formula hypotheses.
    # Keep its best-scoring occurrence before assigning final ranks.
    #
    df = df.sort_values(
        [
            "query_inchikey14",
            "candidate_inchikey14",
            "final_score",
        ],
        ascending=[
            True,
            True,
            False,
        ],
    )

    df = df.drop_duplicates(
        subset=[
            "query_inchikey14",
            "candidate_inchikey14",
        ],
        keep="first",
    )

    df = df.sort_values(
        [
            "query_inchikey14",
            "final_score",
        ],
        ascending=[
            True,
            False,
        ],
    )

    df["rank"] = (
        df.groupby(
            "query_inchikey14"
        )
        .cumcount()
        + 1
    )

    truth_ranks = (
        df.loc[
            df["is_truth"],
            [
                "query_inchikey14",
                "rank",
            ],
        ]
        .set_index(
            "query_inchikey14"
        )["rank"]
        .to_dict()
    )

    #
    # The validation set contains 500 queries. Queries with no
    # candidates or without the truth structure count as failures.
    #
    all_known_queries = (
        ranked[
            "query_inchikey14"
        ]
        .drop_duplicates()
        .tolist()
    )

    ranks = [
        int(
            truth_ranks[q]
        )
        if q in truth_ranks
        else None
        for q in all_known_queries
    ]

    #
    # ranked may contain only queries that reached candidate
    # generation. Pad to the complete 500-query validation set.
    #
    if len(ranks) > TOTAL_QUERIES:
        raise ValueError(
            f"Found {len(ranks)} queries, expected <= {TOTAL_QUERIES}"
        )

    ranks.extend(
        [None]
        * (
            TOTAL_QUERIES
            - len(ranks)
        )
    )

    result = metrics(ranks)

    print()
    print(
        "V2 frozen contrastive multi-formula ranking"
    )
    print(
        "-------------------------------------------"
    )

    print(
        f"lambda:  {LAMBDA:.3f}"
    )
    print(
        f"MRR@25:  {result['mrr25']:.4f}"
    )
    print(
        f"Top-1:   {result['top1'] * 100:.2f}%"
    )
    print(
        f"Top-5:   {result['top5'] * 100:.2f}%"
    )
    print(
        f"Top-10:  {result['top10'] * 100:.2f}%"
    )
    print(
        f"Top-25:  {result['top25'] * 100:.2f}%"
    )
    print(
        "Median truth rank:",
        result["median_rank"],
    )
    print(
        "Mean truth rank:",
        result["mean_rank"],
    )
    print(
        "P90 truth rank:",
        result["p90_rank"],
    )
    print(
        "P95 truth rank:",
        result["p95_rank"],
    )
    print(
        "Max truth rank:",
        result["max_rank"],
    )

    print()
    print(
        "Current frozen fingerprint pipeline"
    )
    print(
        "-----------------------------------"
    )
    print("MRR@25: 0.4838")
    print("Top-1:  37.40%")
    print("Top-5:  60.00%")
    print("Top-10: 71.60%")
    print("Top-25: 83.20%")

    df.to_parquet(
        OUTPUT_PATH,
        index=False,
    )

    print()
    print(
        f"Wrote {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()