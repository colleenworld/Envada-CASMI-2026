from __future__ import annotations

from pathlib import Path

import pandas as pd


SOURCE_PATH = Path(
    "data/processed/structure_index/structures.parquet"
)

VALIDATION_PATH = Path(
    "data/processed/splits/validation_structures.parquet"
)

OUTPUT_PATH = Path(
    "data/processed/structure_index/training_structures.parquet"
)


def main() -> None:
    structures = pd.read_parquet(SOURCE_PATH)
    validation = pd.read_parquet(VALIDATION_PATH)

    validation_keys = set(
        validation["inchikey14"]
        .dropna()
        .astype(str)
    )

    is_validation = (
        structures["inchikey14"]
        .astype(str)
        .isin(validation_keys)
    )

    training = (
        structures.loc[~is_validation]
        .copy()
        .reset_index(drop=True)
    )

    removed = int(is_validation.sum())

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    training.to_parquet(
        OUTPUT_PATH,
        index=False,
    )

    training_keys = set(
        training["inchikey14"]
        .dropna()
        .astype(str)
    )

    remaining_overlap = (
        training_keys & validation_keys
    )

    print("Training-only structure index")
    print("=" * 70)

    print(
        f"Original structures:       "
        f"{len(structures):,}"
    )

    print(
        f"Validation keys:            "
        f"{len(validation_keys):,}"
    )

    print(
        f"Validation structures removed: "
        f"{removed:,}"
    )

    print(
        f"Training structures:       "
        f"{len(training):,}"
    )

    print(
        f"Remaining validation overlap: "
        f"{len(remaining_overlap):,}"
    )

    if remaining_overlap:
        raise RuntimeError(
            "Training index still contains validation structures"
        )

    print()
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()