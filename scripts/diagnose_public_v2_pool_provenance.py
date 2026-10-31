from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent

POOL = (
    ROOT
    / "data/external/casmi26-public-assets/"
      "casmi26-v2-pool/pool_meta.parquet"
)

TRUTH = (
    ROOT
    / "data/processed/blind_pseudotest/truth.parquet"
)

COCONUT = (
    ROOT
    / "data/external/coconut/coconut_approved.parquet"
)


def main() -> None:
    pool = pd.read_parquet(
        POOL,
        columns=[
            "key",
            "formula",
            "src",
            "train_sid",
        ],
    )

    truth = pd.read_parquet(
        TRUTH,
        columns=[
            "molecule_id",
            "inchikey14",
            "molecular_formula",
        ],
    )

    coconut = pd.read_parquet(
        COCONUT,
        columns=[
            "inchikey14",
        ],
    )

    coconut_keys = set(
        coconut["inchikey14"]
        .dropna()
        .astype(str)
    )

    pool["key"] = pool["key"].astype(str)
    pool["formula"] = pool["formula"].astype(str)
    pool["in_local_coconut"] = (
        pool["key"].isin(coconut_keys)
    )
    pool["train_sid_nonnegative"] = (
        pd.to_numeric(
            pool["train_sid"],
            errors="coerce",
        ).fillna(-1) >= 0
    )

    truth_keys = set(
        truth["inchikey14"]
        .astype(str)
    )
    pool["is_blind_truth"] = (
        pool["key"].isin(truth_keys)
    )

    print()
    print("Pool row count:", f"{len(pool):,}")
    print("Unique keys:", f"{pool['key'].nunique():,}")

    print()
    print("src distribution")
    print("=" * 72)
    print(
        pool.groupby(
            "src",
            dropna=False,
        )
        .agg(
            rows=("key", "size"),
            unique_keys=("key", "nunique"),
            train_sid_nonnegative=(
                "train_sid_nonnegative",
                "sum",
            ),
            local_coconut_overlap=(
                "in_local_coconut",
                "sum",
            ),
            blind_truth_matches=(
                "is_blind_truth",
                "sum",
            ),
        )
        .sort_index()
        .to_string()
    )

    print()
    print("train_sid summary")
    print("=" * 72)
    print(
        pool["train_sid"]
        .describe()
        .to_string()
    )

    print()
    print("Blind truth matches in public v2 pool")
    print("=" * 72)

    matched = pool.loc[
        pool["is_blind_truth"],
        [
            "key",
            "formula",
            "src",
            "train_sid",
            "in_local_coconut",
        ],
    ].copy()

    print("Rows:", len(matched))
    print(
        matched.groupby(
            [
                "src",
                "in_local_coconut",
            ],
            dropna=False,
        )
        .size()
        .rename("count")
        .to_string()
    )

    print()
    print(
        "Blind matches with train_sid < 0:",
        int(
            (
                pd.to_numeric(
                    matched["train_sid"],
                    errors="coerce",
                ).fillna(-1)
                < 0
            ).sum()
        ),
    )

    print(
        "Blind matches also in local COCONUT:",
        int(
            matched[
                "in_local_coconut"
            ].sum()
        ),
    )


if __name__ == "__main__":
    main()
