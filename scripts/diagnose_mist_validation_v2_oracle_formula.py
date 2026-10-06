from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(
    "data/processed/mist/validation_500_v2"
)

RANKED_PATH = (
    ROOT / "mist_ranked_candidates.parquet"
)

MANIFEST_PATH = (
    ROOT / "manifest.parquet"
)


def reciprocal_rank(rank, top_k=25):
    if rank is None or rank > top_k:
        return 0.0

    return 1.0 / rank


def main():
    ranked = pd.read_parquet(
        RANKED_PATH
    )

    manifest = pd.read_parquet(
        MANIFEST_PATH
    )

    all_queries = manifest[
        "query_inchikey14"
    ].tolist()

    oracle_ranks = {}
    candidate_counts = {}

    for query, group in ranked.groupby(
        "query_inchikey14"
    ):
        truth = group[
            group["is_truth_structure"]
        ]

        if truth.empty:
            continue

        if len(truth) != 1:
            raise ValueError(
                f"{query}: expected one truth row, "
                f"found {len(truth)}"
            )

        truth_row = truth.iloc[0]

        truth_formula = (
            truth_row["candidate_formula"]
        )

        #
        # Oracle formula:
        # retain only structures belonging to the
        # true molecular formula.
        #
        same_formula = group[
            group["candidate_formula"]
            == truth_formula
        ].copy()

        #
        # Formula-level mass penalties are identical
        # for every structure in this subset, so rank
        # purely by MIST fingerprint similarity.
        #
        same_formula = (
            same_formula.sort_values(
                by=[
                    "mist_cosine",
                    "candidate_inchikey14",
                ],
                ascending=[
                    False,
                    True,
                ],
            )
            .reset_index(drop=True)
        )

        same_formula["oracle_rank"] = (
            np.arange(
                len(same_formula)
            )
            + 1
        )

        truth_oracle = same_formula[
            same_formula[
                "is_truth_structure"
            ]
        ]

        if len(truth_oracle) != 1:
            raise ValueError(
                f"{query}: truth missing from "
                "oracle formula subset"
            )

        oracle_ranks[query] = int(
            truth_oracle.iloc[0][
                "oracle_rank"
            ]
        )

        candidate_counts[query] = len(
            same_formula
        )

    ranks = [
        oracle_ranks.get(query)
        for query in all_queries
    ]

    rankable = [
        rank
        for rank in ranks
        if rank is not None
    ]

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

    print(
        "Oracle-formula MIST validation v2"
    )
    print(
        "---------------------------------"
    )

    print(
        "Queries evaluated:",
        len(all_queries),
    )

    print(
        "Truth structures rankable:",
        len(rankable),
        "/",
        len(all_queries),
    )

    print(
        "MRR@25:",
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
        arr = np.asarray(
            rankable,
            dtype=float,
        )

        conditional_rr = np.asarray(
            [
                reciprocal_rank(rank)
                for rank in rankable
            ]
        )

        print(
            "\nConditional on truth rankable"
        )
        print(
            "-----------------------------"
        )

        print(
            "MRR@25:",
            f"{conditional_rr.mean():.4f}",
        )

        print(
            "Top-1:",
            f"{np.mean(arr <= 1) * 100:.2f}%",
        )

        print(
            "Top-5:",
            f"{np.mean(arr <= 5) * 100:.2f}%",
        )

        print(
            "Top-10:",
            f"{np.mean(arr <= 10) * 100:.2f}%",
        )

        print(
            "Top-25:",
            f"{np.mean(arr <= 25) * 100:.2f}%",
        )

        print(
            "\nTruth rank distribution"
        )
        print(
            "-----------------------"
        )

        print(
            "Median:",
            float(np.median(arr)),
        )

        print(
            "Mean:",
            float(np.mean(arr)),
        )

        print(
            "P75:",
            float(
                np.percentile(arr, 75)
            ),
        )

        print(
            "P90:",
            float(
                np.percentile(arr, 90)
            ),
        )

        print(
            "P95:",
            float(
                np.percentile(arr, 95)
            ),
        )

        print(
            "Max:",
            int(np.max(arr)),
        )

        counts = np.asarray(
            [
                candidate_counts[q]
                for q in oracle_ranks
            ],
            dtype=float,
        )

        print(
            "\nTrue-formula structure candidates"
        )
        print(
            "---------------------------------"
        )

        print(
            "Median:",
            float(np.median(counts)),
        )

        print(
            "Mean:",
            float(np.mean(counts)),
        )

        print(
            "P90:",
            float(
                np.percentile(counts, 90)
            ),
        )

        print(
            "P95:",
            float(
                np.percentile(counts, 95)
            ),
        )

        print(
            "Max:",
            int(np.max(counts)),
        )


if __name__ == "__main__":
    main()