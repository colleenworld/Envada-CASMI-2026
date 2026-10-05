from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path("data/processed/mist/benchmark_100")

ORACLE_PATH = ROOT / "mist_ranked_candidates.parquet"
MULTI_PATH = ROOT / "mist_multiformula_ranked_candidates.parquet"
MAPPING_PATH = ROOT / "mist_multiformula_mapping.parquet"
MANIFEST_PATH = ROOT / "manifest.parquet"


def truth_ranks(
    ranked: pd.DataFrame,
    manifest: pd.DataFrame,
    rank_name: str,
    rank_column: str,
) -> pd.DataFrame:
    truth = ranked[
        ranked["candidate_inchikey14"]
        == ranked["query_inchikey14"]
    ][
        ["query_inchikey14", rank_column]
    ].copy()

    truth = (
        truth.groupby("query_inchikey14", as_index=False)[rank_column]
        .min()
        .rename(columns={rank_column: rank_name})
    )

    return (
        manifest[["query_inchikey14", "truth_formula"]]
        .drop_duplicates()
        .merge(
            truth,
            on="query_inchikey14",
            how="left",
        )
    )


def main():
    oracle = pd.read_parquet(ORACLE_PATH)
    multi = pd.read_parquet(MULTI_PATH)
    mapping = pd.read_parquet(MAPPING_PATH)
    manifest = pd.read_parquet(MANIFEST_PATH)

    oracle_ranks = truth_ranks(
        oracle,
        manifest,
        "oracle_rank",
        "mist_rank",
    )

    multi_ranks = truth_ranks(
        multi,
        manifest,
        "multi_rank",
        "rank",
    )

    comparison = oracle_ranks.merge(
        multi_ranks[
            ["query_inchikey14", "multi_rank"]
        ],
        on="query_inchikey14",
        how="left",
    )

    formula_available = (
        mapping.groupby("query_inchikey14")["is_truth_formula"]
        .any()
        .rename("truth_formula_available")
    )

    comparison = comparison.merge(
        formula_available,
        on="query_inchikey14",
        how="left",
    )

    comparison["truth_formula_available"] = (
        comparison["truth_formula_available"]
        .fillna(False)
    )

    surviving = comparison[
        comparison["truth_formula_available"]
        & comparison["multi_rank"].notna()
    ].copy()

    surviving["rank_delta"] = (
        surviving["multi_rank"]
        - surviving["oracle_rank"]
    )

    print("Queries:", len(comparison))
    print(
        "Truth formula available:",
        int(comparison["truth_formula_available"].sum()),
    )

    print("\nAmong truth-formula survivors:")
    print("Queries:", len(surviving))

    print(
        "Improved:",
        int((surviving["rank_delta"] < 0).sum()),
    )
    print(
        "Unchanged:",
        int((surviving["rank_delta"] == 0).sum()),
    )
    print(
        "Worsened:",
        int((surviving["rank_delta"] > 0).sum()),
    )

    print("\nRank delta:")
    print(
        surviving["rank_delta"]
        .describe(
            percentiles=[0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
        )
        .to_string()
    )

    print("\nLargest degradations:")
    print(
        surviving.sort_values(
            "rank_delta",
            ascending=False,
        )
        .head(15)
        [
            [
                "query_inchikey14",
                "truth_formula",
                "oracle_rank",
                "multi_rank",
                "rank_delta",
            ]
        ]
        .to_string(index=False)
    )

    print("\nLargest improvements:")
    print(
        surviving.sort_values(
            "rank_delta",
            ascending=True,
        )
        .head(10)
        [
            [
                "query_inchikey14",
                "truth_formula",
                "oracle_rank",
                "multi_rank",
                "rank_delta",
            ]
        ]
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()