from pathlib import Path

import pandas as pd


ROOT = Path("data/processed/mist/benchmark_100")

ORACLE_PATH = ROOT / "mist_ranked_candidates.parquet"
MULTI_PATH = ROOT / "mist_multiformula_ranked_candidates.parquet"
MAPPING_PATH = ROOT / "mist_multiformula_mapping.parquet"
MANIFEST_PATH = ROOT / "manifest.parquet"


def main():
    oracle = pd.read_parquet(ORACLE_PATH)
    multi = pd.read_parquet(MULTI_PATH)
    mapping = pd.read_parquet(MAPPING_PATH)
    manifest = pd.read_parquet(MANIFEST_PATH)

    # Oracle truth rank.
    oracle_truth = (
        oracle.loc[
            oracle["candidate_inchikey14"]
            == oracle["query_inchikey14"],
            [
                "query_inchikey14",
                "mist_rank",
            ],
        ]
        .groupby("query_inchikey14", as_index=False)
        ["mist_rank"]
        .min()
        .rename(columns={"mist_rank": "oracle_rank"})
    )

    # Multi-formula truth candidate row.
    multi_truth = (
        multi.loc[
            multi["candidate_inchikey14"]
            == multi["query_inchikey14"]
        ]
        .sort_values("rank")
        .drop_duplicates(
            "query_inchikey14",
            keep="first",
        )
        [
            [
                "query_inchikey14",
                "candidate_formula",
                "mist_cosine",
                "formula_mass_error_ppm",
                "rank",
            ]
        ]
        .rename(
            columns={
                "candidate_formula": "truth_formula",
                "mist_cosine": "truth_mist_cosine",
                "formula_mass_error_ppm":
                    "truth_formula_mass_error_ppm",
                "rank": "multi_rank",
            }
        )
    )

    comparison = (
        manifest[
            [
                "query_inchikey14",
                "truth_formula",
            ]
        ]
        .drop_duplicates()
        .merge(
            oracle_truth,
            on="query_inchikey14",
            how="left",
        )
        .merge(
            multi_truth.drop(
                columns=["truth_formula"]
            ),
            on="query_inchikey14",
            how="left",
        )
    )

    comparison["rank_delta"] = (
        comparison["multi_rank"]
        - comparison["oracle_rank"]
    )

    worsened = comparison[
        comparison["rank_delta"] > 0
    ].copy()

    print("Worsened queries:", len(worsened))

    summary_rows = []
    intrusion_rows = []

    for q in worsened.itertuples(index=False):
        query_rows = multi[
            multi["query_inchikey14"]
            == q.query_inchikey14
        ].copy()

        truth_rank = int(q.multi_rank)

        #
        # These are all candidates that rank ahead of the truth in
        # the multi-formula result.
        #
        above_truth = query_rows[
            query_rows["rank"] < truth_rank
        ].copy()

        #
        # Only candidates from a different formula are genuine
        # cross-formula intrusions.
        #
        intrusions = above_truth[
            above_truth["candidate_formula"]
            != q.truth_formula
        ].copy()

        summary_rows.append(
            {
                "query_inchikey14":
                    q.query_inchikey14,
                "truth_formula":
                    q.truth_formula,
                "oracle_rank":
                    q.oracle_rank,
                "multi_rank":
                    q.multi_rank,
                "rank_delta":
                    q.rank_delta,
                "truth_mist_cosine":
                    q.truth_mist_cosine,
                "truth_formula_mass_error_ppm":
                    q.truth_formula_mass_error_ppm,
                "cross_formula_intrusions":
                    len(intrusions),
                "better_mass_than_truth":
                    int(
                        (
                            intrusions[
                                "formula_mass_error_ppm"
                            ].abs()
                            <
                            abs(
                                q.truth_formula_mass_error_ppm
                            )
                        ).sum()
                    ),
            }
        )

        for row in intrusions.itertuples(index=False):
            intrusion_rows.append(
                {
                    "query_inchikey14":
                        q.query_inchikey14,
                    "truth_formula":
                        q.truth_formula,
                    "truth_rank":
                        truth_rank,
                    "truth_mist_cosine":
                        q.truth_mist_cosine,
                    "truth_formula_mass_error_ppm":
                        q.truth_formula_mass_error_ppm,

                    "intruder_rank":
                        row.rank,
                    "intruder_formula":
                        row.candidate_formula,
                    "intruder_inchikey14":
                        row.candidate_inchikey14,
                    "intruder_mist_cosine":
                        row.mist_cosine,
                    "intruder_formula_mass_error_ppm":
                        row.formula_mass_error_ppm,

                    "mist_advantage":
                        row.mist_cosine
                        - q.truth_mist_cosine,

                    "abs_mass_error_advantage":
                        abs(q.truth_formula_mass_error_ppm)
                        - abs(row.formula_mass_error_ppm),
                }
            )

    summary = pd.DataFrame(summary_rows)
    intrusions = pd.DataFrame(intrusion_rows)

    print("\nPer-query summary")
    print("-----------------")
    print(
        summary.sort_values(
            "rank_delta",
            ascending=False,
        ).to_string(index=False)
    )

    if intrusions.empty:
        print("\nNo cross-formula intrusions found.")
        return

    print("\nCross-formula intrusions")
    print("------------------------")
    print("Total:", len(intrusions))

    print(
        "Intruders with better absolute mass error "
        "than truth:",
        int(
            (
                intrusions[
                    "abs_mass_error_advantage"
                ] > 0
            ).sum()
        ),
        "/",
        len(intrusions),
    )

    print("\nMass-error comparison:")
    print(
        intrusions[
            [
                "truth_formula_mass_error_ppm",
                "intruder_formula_mass_error_ppm",
                "abs_mass_error_advantage",
            ]
        ]
        .describe(
            percentiles=[
                0.25,
                0.5,
                0.75,
                0.9,
            ]
        )
        .to_string()
    )

    print("\nLargest intrusions by MIST advantage:")
    print(
        intrusions.sort_values(
            "mist_advantage",
            ascending=False,
        )
        .head(30)
        .to_string(index=False)
    )

    out = ROOT / "mist_cross_formula_intrusions.parquet"
    intrusions.to_parquet(out, index=False)

    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()