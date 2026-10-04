from __future__ import annotations

from pathlib import Path

import pandas as pd


INDEX_PATH = Path(
    "data/processed/formula_index/"
    "training_coconut_formulas.parquet"
)

TRAINING_PATH = Path(
    "data/processed/structure_index/"
    "training_structures.parquet"
)

COCONUT_PATH = Path(
    "data/external/coconut/"
    "coconut_approved.parquet"
)

VALIDATION_KEYS_PATH = Path(
    "data/processed/splits/"
    "validation_structures.parquet"
)

ALL_STRUCTURES_PATH = Path(
    "data/processed/structure_index/"
    "structures.parquet"
)


def main() -> None:
    index = pd.read_parquet(INDEX_PATH)

    training = pd.read_parquet(
        TRAINING_PATH,
        columns=[
            "inchikey14",
            "molecular_formula",
        ],
    )

    coconut = pd.read_parquet(
        COCONUT_PATH,
        columns=[
            "inchikey14",
            "molecular_formula",
        ],
    )

    validation_keys = set(
        pd.read_parquet(
            VALIDATION_KEYS_PATH,
            columns=["inchikey14"],
        )["inchikey14"]
        .dropna()
        .astype(str)
    )

    all_structures = pd.read_parquet(
        ALL_STRUCTURES_PATH,
        columns=[
            "inchikey14",
            "molecular_formula",
        ],
    )

    validation_rows = all_structures[
        all_structures["inchikey14"]
        .astype(str)
        .isin(validation_keys)
    ]

    validation_formulas = set(
        validation_rows["molecular_formula"]
        .dropna()
        .astype(str)
    )

    training_formulas = set(
        training["molecular_formula"]
        .dropna()
        .astype(str)
    )

    coconut_formulas = set(
        coconut["molecular_formula"]
        .dropna()
        .astype(str)
    )

    index_formulas = set(
        index["molecular_formula"]
        .dropna()
        .astype(str)
    )

    training_keys = set(
        training["inchikey14"]
        .dropna()
        .astype(str)
    )

    print("External formula index validation")
    print("=" * 70)

    print(
        f"Index formulas:                 "
        f"{len(index_formulas):,}"
    )

    print(
        f"Validation structures:          "
        f"{len(validation_keys):,}"
    )

    print(
        f"Validation structures indexable:"
        f" {len(validation_rows):,}"
    )

    print(
        f"Validation formulas:            "
        f"{len(validation_formulas):,}"
    )

    print()
    print("Validation formula availability")
    print("-" * 70)

    in_training = (
        validation_formulas & training_formulas
    )

    in_coconut = (
        validation_formulas & coconut_formulas
    )

    in_combined = (
        validation_formulas & index_formulas
    )

    print(
        f"In training formula universe:   "
        f"{len(in_training):,} / "
        f"{len(validation_formulas):,} "
        f"({len(in_training) / len(validation_formulas):.3%})"
    )

    print(
        f"In COCONUT formula universe:    "
        f"{len(in_coconut):,} / "
        f"{len(validation_formulas):,} "
        f"({len(in_coconut) / len(validation_formulas):.3%})"
    )

    print(
        f"In combined formula universe:   "
        f"{len(in_combined):,} / "
        f"{len(validation_formulas):,} "
        f"({len(in_combined) / len(validation_formulas):.3%})"
    )

    print()
    print(
        f"Validation formulas supplied only by COCONUT: "
        f"{len((validation_formulas & coconut_formulas) - training_formulas):,}"
    )

    print(
        f"Validation formulas absent from combined index: "
        f"{len(validation_formulas - index_formulas):,}"
    )

    print()
    print("Leakage checks")
    print("-" * 70)

    overlap = training_keys & validation_keys

    print(
        f"Training/validation structure overlap: "
        f"{len(overlap):,}"
    )

    if overlap:
        raise RuntimeError(
            "Training structure index contains validation structures"
        )

    duplicates = int(
        index["molecular_formula"].duplicated().sum()
    )

    print(
        f"Duplicate formulas in index:           "
        f"{duplicates:,}"
    )

    if duplicates:
        raise RuntimeError(
            "Formula index contains duplicate formulas"
        )

    sorted_masses = (
        index["monoisotopic_mass"]
        .is_monotonic_increasing
    )

    print(
        f"Masses sorted:                         "
        f"{sorted_masses}"
    )

    if not sorted_masses:
        raise RuntimeError(
            "Formula index is not sorted by mass"
        )

    print()
    print("Validation passed.")


if __name__ == "__main__":
    main()