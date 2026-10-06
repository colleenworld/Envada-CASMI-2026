from pathlib import Path
import pickle

import numpy as np
import pandas as pd


ROOT = Path("data/processed/mist/validation_500")

RANKED_PATH = ROOT / "mist_validation_ranked_candidates.parquet"

CONTRASTIVE_PATH = (
    ROOT
    / "contrastive_predictions"
    / "retrieval_contrastive_casmi_validation_500_morgan4096_"
      "with_morgan4096_retrieval_db_"
      "casmi_validation_500_contrastive_cosine.p"
)

OUTPUT_PATH = ROOT / "contrastive_multiformula_results.parquet"

TOP_K = 25

# Small, predefined v1-only grid.
LAMBDAS = [
    0.0,
    0.0025,
    0.005,
    0.010,
    0.020,
]


def to_str(value):
    if isinstance(value, bytes):
        return value.decode()
    return str(value)


def reciprocal_rank(rank):
    if rank is None or rank > TOP_K:
        return 0.0
    return 1.0 / rank


def metric_summary(ranks):
    ranks_clean = [
        r for r in ranks
        if r is not None
    ]

    rr = np.array(
        [
            reciprocal_rank(r)
            for r in ranks
        ],
        dtype=float,
    )

    def topk(k):
        return np.mean(
            [
                r is not None and r <= k
                for r in ranks
            ]
        )

    return {
        "mrr25": float(rr.mean()),
        "top1": float(topk(1)),
        "top5": float(topk(5)),
        "top10": float(topk(10)),
        "top25": float(topk(25)),
        "median_rank":
            float(np.median(ranks_clean))
            if ranks_clean else np.nan,
        "mean_rank":
            float(np.mean(ranks_clean))
            if ranks_clean else np.nan,
    }


def main():
    ranked = pd.read_parquet(RANKED_PATH)

    required = {
        "query_inchikey14",
        "hypothesis_spec",
        "candidate_formula",
        "candidate_inchikey14",
        "formula_mass_error_ppm",
    }

    missing = required - set(ranked.columns)

    if missing:
        raise ValueError(
            f"Missing ranked columns: {sorted(missing)}"
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
        in zip(names, ikey_lists, dist_lists)
    }

    rows = []

    #
    # Expand each formula hypothesis back into candidate rows,
    # attaching its contrastive distance.
    #
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
            dist = dist_by_ikey.get(
                row.candidate_inchikey14
            )

            if dist is None:
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
                            row.formula_mass_error_ppm
                        ),

                    "contrastive_distance":
                        dist,

                    "is_truth":
                        (
                            row.candidate_inchikey14
                            == row.query_inchikey14
                        ),
                }
            )

    df = pd.DataFrame(rows)

    print(
        "Contrastive candidate rows:",
        len(df),
    )
    print(
        "Queries:",
        df["query_inchikey14"].nunique(),
    )
    print(
        "Truth-containing queries:",
        df.loc[
            df["is_truth"],
            "query_inchikey14",
        ].nunique(),
    )

    print()
    print("Contrastive distance distribution")
    print("---------------------------------")
    print(
        df["contrastive_distance"]
        .describe(
            percentiles=[
                0.10,
                0.25,
                0.50,
                0.75,
                0.90,
                0.95,
                0.99,
            ]
        )
        .to_string()
    )

    truth = df[df["is_truth"]]

    print()
    print("Truth contrastive distance")
    print("--------------------------")
    print(
        truth["contrastive_distance"]
        .describe(
            percentiles=[
                0.10,
                0.25,
                0.50,
                0.75,
                0.90,
                0.95,
                0.99,
            ]
        )
        .to_string()
    )

    print()
    print("Truth absolute ppm error")
    print("------------------------")
    print(
        truth["formula_mass_error_ppm"]
        .abs()
        .describe(
            percentiles=[
                0.10,
                0.25,
                0.50,
                0.75,
                0.90,
                0.95,
                0.99,
            ]
        )
        .to_string()
    )

    summaries = []

    for lam in LAMBDAS:
        working = df.copy()

        #
        # Lower contrastive distance is better, so convert to a
        # descending score by negating it.
        #
        working["final_score"] = (
            -working["contrastive_distance"]
            - lam
            * working[
                "formula_mass_error_ppm"
            ].abs()
        )

        working = working.sort_values(
            [
                "query_inchikey14",
                "final_score",
            ],
            ascending=[
                True,
                False,
            ],
        )

        working["rank"] = (
            working.groupby(
                "query_inchikey14"
            )
            .cumcount()
            + 1
        )

        truth_ranks = (
            working.loc[
                working["is_truth"],
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

        all_queries = (
            working[
                "query_inchikey14"
            ]
            .drop_duplicates()
            .tolist()
        )

        ranks = [
            int(truth_ranks[q])
            if q in truth_ranks
            else None
            for q in all_queries
        ]

        summary = metric_summary(
            ranks
        )

        summary["lambda"] = lam

        summaries.append(summary)

    summary_df = pd.DataFrame(
        summaries
    )

    summary_df = summary_df[
        [
            "lambda",
            "mrr25",
            "top1",
            "top5",
            "top10",
            "top25",
            "median_rank",
            "mean_rank",
        ]
    ]

    print()
    print("V1 contrastive multi-formula penalty sweep")
    print("------------------------------------------")
    print(
        summary_df.to_string(
            index=False,
            formatters={
                "mrr25":
                    lambda x: f"{x:.4f}",
                "top1":
                    lambda x: f"{x * 100:.2f}%",
                "top5":
                    lambda x: f"{x * 100:.2f}%",
                "top10":
                    lambda x: f"{x * 100:.2f}%",
                "top25":
                    lambda x: f"{x * 100:.2f}%",
            },
        )
    )

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