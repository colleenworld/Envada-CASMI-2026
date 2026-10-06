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

CONTRASTIVE_PATH = (
    ROOT
    / "contrastive_predictions"
    / (
        "retrieval_contrastive_"
        "casmi_validation_500_morgan4096_"
        "with_morgan4096_retrieval_db_"
        "casmi_validation_500_contrastive_"
        "cosine.p"
    )
)

TOP_K = 25


def to_str(value):
    if isinstance(value, bytes):
        return value.decode()

    return str(value)


def reciprocal_rank(rank):
    if rank is None:
        return 0.0

    if rank > TOP_K:
        return 0.0

    return 1.0 / rank


def main():
    #
    # Existing v1 candidate/ranking artifact.
    #
    ranked = pd.read_parquet(
        RANKED_PATH
    )

    print(
        "Existing ranked candidate rows:",
        len(ranked),
    )

    #
    # Determine the truth structure and, critically, the exact
    # TRUE-FORMULA hypothesis for every rankable query.
    #
    truth_rows = ranked[
        ranked["is_truth"]
    ].copy()

    print(
        "Truth-containing queries:",
        truth_rows[
            "query_inchikey14"
        ].nunique(),
    )

    if truth_rows[
        "query_inchikey14"
    ].duplicated().any():
        duplicates = (
            truth_rows.loc[
                truth_rows[
                    "query_inchikey14"
                ].duplicated(
                    keep=False
                ),
                "query_inchikey14",
            ]
            .unique()
            .tolist()
        )

        raise ValueError(
            "Expected exactly one truth candidate "
            "per query. Duplicates: "
            f"{duplicates[:10]}"
        )

    truth_by_query = (
        truth_rows.set_index(
            "query_inchikey14"
        )
    )

    #
    # Load contrastive retrieval output.
    #
    with CONTRASTIVE_PATH.open(
        "rb"
    ) as f:
        contrastive = pickle.load(f)

    print(
        "Contrastive keys:",
        list(
            contrastive.keys()
        ),
    )

    names = [
        to_str(name)
        for name in contrastive[
            "names"
        ]
    ]

    ikey_lists = (
        contrastive[
            "ikeys"
        ]
    )

    dist_lists = (
        contrastive[
            "dists"
        ]
    )

    print(
        "Contrastive hypotheses:",
        len(names),
    )

    if len(names) != len(
        ikey_lists
    ):
        raise ValueError(
            "Contrastive names/ikeys counts differ."
        )

    if len(names) != len(
        dist_lists
    ):
        raise ValueError(
            "Contrastive names/dists counts differ."
        )

    if len(
        set(names)
    ) != len(names):
        raise ValueError(
            "Duplicate contrastive hypothesis names."
        )

    contrastive_by_name = {
        name: {
            "ikeys": [
                to_str(x)
                for x in ikeys
            ],
            "dists":
                np.asarray(
                    dists,
                    dtype=float,
                ),
        }
        for name, ikeys, dists
        in zip(
            names,
            ikey_lists,
            dist_lists,
        )
    }

    #
    # Evaluate only queries whose truth structure exists in the
    # leakage-safe candidate universe.
    #
    result_rows = []

    for query, truth in (
        truth_by_query.iterrows()
    ):
        hypothesis = (
            truth[
                "hypothesis_spec"
            ]
        )

        truth_ikey = (
            truth[
                "candidate_inchikey14"
            ]
        )

        candidate_formula = (
            truth[
                "candidate_formula"
            ]
        )

        existing_cosine_rank = int(
            truth[
                "rank"
            ]
        )

        retrieval = (
            contrastive_by_name.get(
                hypothesis
            )
        )

        if retrieval is None:
            result_rows.append(
                {
                    "query_inchikey14":
                        query,

                    "hypothesis_spec":
                        hypothesis,

                    "truth_formula":
                        candidate_formula,

                    "truth_inchikey14":
                        truth_ikey,

                    "baseline_rank":
                        existing_cosine_rank,

                    "contrastive_rank":
                        np.nan,

                    "truth_contrastive_distance":
                        np.nan,

                    "num_contrastive_candidates":
                        0,

                    "status":
                        "missing_hypothesis",
                }
            )

            continue

        ikeys = retrieval[
            "ikeys"
        ]

        dists = retrieval[
            "dists"
        ]

        try:
            zero_based_rank = (
                ikeys.index(
                    truth_ikey
                )
            )

            contrastive_rank = (
                zero_based_rank
                + 1
            )

            truth_dist = float(
                dists[
                    zero_based_rank
                ]
            )

            status = "ranked"

        except ValueError:
            contrastive_rank = None
            truth_dist = np.nan
            status = "truth_missing"

        result_rows.append(
            {
                "query_inchikey14":
                    query,

                "hypothesis_spec":
                    hypothesis,

                "truth_formula":
                    candidate_formula,

                "truth_inchikey14":
                    truth_ikey,

                "baseline_rank":
                    existing_cosine_rank,

                "contrastive_rank":
                    contrastive_rank,

                "truth_contrastive_distance":
                    truth_dist,

                "num_contrastive_candidates":
                    len(ikeys),

                "status":
                    status,
            }
        )

    results = pd.DataFrame(
        result_rows
    )

    output_path = (
        ROOT
        / "contrastive_validation_results.parquet"
    )

    results.to_parquet(
        output_path,
        index=False,
    )

    #
    # Diagnostics.
    #
    print()
    print(
        "Contrastive retrieval coverage"
    )
    print(
        "------------------------------"
    )

    print(
        results[
            "status"
        ]
        .value_counts(
            dropna=False
        )
        .to_string()
    )

    print(
        "\nQueries evaluated:",
        len(results),
    )

    #
    # For a fair comparison, every one of these 429 truth
    # structures should be present in the HDF ranking.
    #
    missing = results[
        results[
            "status"
        ]
        != "ranked"
    ]

    if not missing.empty:
        print(
            "\nWARNING: contrastive retrieval "
            "did not rank all truth structures."
        )

        print(
            missing[
                [
                    "query_inchikey14",
                    "hypothesis_spec",
                    "truth_formula",
                    "status",
                ]
            ]
            .head(20)
            .to_string(
                index=False
            )
        )

    #
    # Treat missing truth ranks as failures, so the metric remains
    # conservative.
    #
    contrastive_ranks = []

    baseline_ranks = []

    for row in results.itertuples(
        index=False
    ):
        baseline_ranks.append(
            int(
                row.baseline_rank
            )
        )

        if pd.isna(
            row.contrastive_rank
        ):
            contrastive_ranks.append(
                None
            )
        else:
            contrastive_ranks.append(
                int(
                    row.contrastive_rank
                )
            )

    def metrics(ranks):
        rr = np.asarray(
            [
                reciprocal_rank(
                    rank
                )
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

            "mean_rank":
                float(
                    np.mean(
                        rankable
                    )
                )
                if rankable
                else np.nan,

            "p90_rank":
                float(
                    np.percentile(
                        rankable,
                        90,
                    )
                )
                if rankable
                else np.nan,

            "p95_rank":
                float(
                    np.percentile(
                        rankable,
                        95,
                    )
                )
                if rankable
                else np.nan,

            "max_rank":
                int(
                    np.max(
                        rankable
                    )
                )
                if rankable
                else None,
        }

    baseline = metrics(
        baseline_ranks
    )

    contrast = metrics(
        contrastive_ranks
    )

    def print_metrics(
        label,
        values,
    ):
        print()
        print(label)
        print(
            "-" * len(label)
        )

        print(
            "MRR@25:",
            f"{values['mrr25']:.4f}",
        )

        print(
            "Top-1:",
            f"{values['top1'] * 100:.2f}%",
        )

        print(
            "Top-5:",
            f"{values['top5'] * 100:.2f}%",
        )

        print(
            "Top-10:",
            f"{values['top10'] * 100:.2f}%",
        )

        print(
            "Top-25:",
            f"{values['top25'] * 100:.2f}%",
        )

        print(
            "Median truth rank:",
            values[
                "median_rank"
            ],
        )

        print(
            "Mean truth rank:",
            values[
                "mean_rank"
            ],
        )

        print(
            "P90 truth rank:",
            values[
                "p90_rank"
            ],
        )

        print(
            "P95 truth rank:",
            values[
                "p95_rank"
            ],
        )

        print(
            "Max truth rank:",
            values[
                "max_rank"
            ],
        )

    #
    # The baseline ranks from mist_validation_ranked_candidates
    # include our full multi-formula final ranking, not necessarily
    # oracle-formula cosine. Therefore also report them mainly as
    # an integrity/reference check.
    #
    print_metrics(
        "Existing v1 ranking reference",
        baseline,
    )

    print_metrics(
        "V1 contrastive same-formula cosine",
        contrast,
    )

    #
    # Direct query-level comparison.
    #
    comparison = results[
        results[
            "status"
        ]
        == "ranked"
    ].copy()

    comparison[
        "rank_delta"
    ] = (
        comparison[
            "contrastive_rank"
        ]
        - comparison[
            "baseline_rank"
        ]
    )

    print()
    print(
        "Query-level rank movement"
    )
    print(
        "-------------------------"
    )

    print(
        "Improved:",
        int(
            (
                comparison[
                    "rank_delta"
                ]
                < 0
            ).sum()
        ),
    )

    print(
        "Unchanged:",
        int(
            (
                comparison[
                    "rank_delta"
                ]
                == 0
            ).sum()
        ),
    )

    print(
        "Worsened:",
        int(
            (
                comparison[
                    "rank_delta"
                ]
                > 0
            ).sum()
        ),
    )

    print(
        "Median rank delta:",
        float(
            comparison[
                "rank_delta"
            ].median()
        ),
    )

    print(
        f"\nWrote {output_path}"
    )


if __name__ == "__main__":
    main()