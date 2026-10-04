from __future__ import annotations

from pathlib import Path

import pandas as pd


COCONUT_PATH = Path(
    "data/external/coconut/coconut_approved.parquet"
)

TRAIN_INDEX_PATH = Path(
    "data/processed/structure_index/structures.parquet"
)

VALIDATION_PATH = Path(
    "data/processed/splits/validation_structures.parquet"
)


def pct(numerator: int, denominator: int) -> str:
    if denominator == 0:
        return "n/a"

    return f"{numerator / denominator:.3%}"


def main() -> None:
    coconut = pd.read_parquet(COCONUT_PATH)
    train = pd.read_parquet(TRAIN_INDEX_PATH)
    validation = pd.read_parquet(VALIDATION_PATH)

    print("Loaded")
    print("=" * 70)
    print(f"COCONUT structures:   {len(coconut):,}")
    print(f"Train index:          {len(train):,}")
    print(f"Validation structures:{len(validation):,}")

    print()
    print("Columns")
    print("=" * 70)
    print("COCONUT:")
    print(list(coconut.columns))
    print()
    print("Train index:")
    print(list(train.columns))
    print()
    print("Validation:")
    print(list(validation.columns))

    coconut_keys = set(
        coconut["inchikey14"]
        .dropna()
        .astype(str)
    )

    train_keys = set(
        train["inchikey14"]
        .dropna()
        .astype(str)
    )

    validation_keys = set(
        validation["inchikey14"]
        .dropna()
        .astype(str)
    )

    overlap_train = coconut_keys & train_keys

    coconut_only = coconut_keys - train_keys

    validation_in_coconut = (
        validation_keys & coconut_keys
    )

    validation_in_train = (
        validation_keys & train_keys
    )

    # This is the particularly interesting set:
    # validation structures externally available through
    # COCONUT but absent from the train-derived catalog.
    validation_coconut_only = (
        validation_keys
        & coconut_keys
        - train_keys
    )

    print()
    print("Structure overlap")
    print("=" * 70)

    print(
        f"COCONUT unique keys:             "
        f"{len(coconut_keys):,}"
    )

    print(
        f"Train unique keys:               "
        f"{len(train_keys):,}"
    )

    print(
        f"COCONUT ∩ train:                 "
        f"{len(overlap_train):,} "
        f"({pct(len(overlap_train), len(coconut_keys))} "
        f"of COCONUT)"
    )

    print(
        f"COCONUT not in train:            "
        f"{len(coconut_only):,} "
        f"({pct(len(coconut_only), len(coconut_keys))})"
    )

    print()
    print("Validation coverage")
    print("=" * 70)

    print(
        f"Validation unique keys:          "
        f"{len(validation_keys):,}"
    )

    print(
        f"Validation found in COCONUT:     "
        f"{len(validation_in_coconut):,} "
        f"({pct(len(validation_in_coconut), len(validation_keys))})"
    )

    print(
        f"Validation found in train index: "
        f"{len(validation_in_train):,} "
        f"({pct(len(validation_in_train), len(validation_keys))})"
    )

    print(
        f"Validation in COCONUT, not train:"
        f" {len(validation_coconut_only):,} "
        f"({pct(len(validation_coconut_only), len(validation_keys))})"
    )

    print()
    print("Formula coverage")
    print("=" * 70)

    coconut_formulas = set(
        coconut["molecular_formula"]
        .dropna()
        .astype(str)
    )

    train_formulas = set(
        train["molecular_formula"]
        .dropna()
        .astype(str)
    )

    print(
        f"COCONUT formulas:                "
        f"{len(coconut_formulas):,}"
    )

    print(
        f"Train formulas:                  "
        f"{len(train_formulas):,}"
    )

    print(
        f"COCONUT formulas not in train:   "
        f"{len(coconut_formulas - train_formulas):,}"
    )

    print(
        f"Combined formulas:               "
        f"{len(coconut_formulas | train_formulas):,}"
    )

    print()
    print("Candidate density")
    print("=" * 70)

    coconut_per_formula = (
        coconut.groupby("molecular_formula")[
            "inchikey14"
        ]
        .nunique()
    )

    print(
        coconut_per_formula.describe(
            percentiles=[
                0.50,
                0.90,
                0.95,
                0.99,
            ]
        )
    )


if __name__ == "__main__":
    main()