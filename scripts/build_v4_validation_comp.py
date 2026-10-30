from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


DEFAULT_TRAIN = Path("data/raw/train.parquet")
DEFAULT_OUTPUT = Path("data/processed/v4_validation")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a structure-disjoint v4 validation competition directory "
            "from the competition training parquet."
        )
    )
    parser.add_argument(
        "--train",
        type=Path,
        default=DEFAULT_TRAIN,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
    )
    parser.add_argument(
        "--structures",
        type=int,
        default=250,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20261030,
    )
    parser.add_argument(
        "--max-spectra-per-structure",
        type=int,
        default=16,
    )
    return parser.parse_args()


def required_columns(path: Path) -> list[str]:
    schema = pq.ParquetFile(path).schema_arrow
    names = set(schema.names)

    candidates = [
        "inchikey14",
        "normalized_smiles",
        "spectrum_id",
        "precursor_mz",
        "adduct",
        "ionization_mode",
        "ms2_mzs",
        "ms2_normalized_intensities",
        "collision_energy_ev",
        "instrument_type",
        "molecular_formula",
    ]

    required = {
        "inchikey14",
        "normalized_smiles",
        "precursor_mz",
        "adduct",
        "ionization_mode",
        "ms2_mzs",
        "ms2_normalized_intensities",
    }

    missing = sorted(required - names)

    if missing:
        raise ValueError(
            "Training parquet is missing required columns: "
            + ", ".join(missing)
        )

    return [
        column
        for column in candidates
        if column in names
    ]


def choose_structures(
    train: pd.DataFrame,
    count: int,
    seed: int,
) -> pd.DataFrame:
    structures = (
        train[
            [
                "inchikey14",
                "normalized_smiles",
            ]
        ]
        .dropna()
        .astype(str)
    )

    structures = structures.loc[
        (structures["inchikey14"] != "")
        & (structures["normalized_smiles"] != "")
        & (structures["normalized_smiles"] != "nan")
    ]

    consistency = (
        structures.groupby(
            "inchikey14",
            sort=False,
        )["normalized_smiles"]
        .nunique()
    )

    valid_keys = consistency.loc[
        consistency == 1
    ].index

    unique_structures = (
        structures.loc[
            structures["inchikey14"].isin(
                valid_keys
            )
        ]
        .drop_duplicates(
            "inchikey14"
        )
        .reset_index(
            drop=True
        )
    )

    if len(unique_structures) < count:
        raise ValueError(
            f"Requested {count:,} structures but only "
            f"{len(unique_structures):,} usable unique structures exist."
        )

    rng = np.random.default_rng(
        seed
    )

    indices = rng.choice(
        len(unique_structures),
        size=count,
        replace=False,
    )

    return (
        unique_structures.iloc[
            np.sort(indices)
        ]
        .reset_index(
            drop=True
        )
    )


def main() -> None:
    args = parse_args()

    if not args.train.exists():
        raise FileNotFoundError(
            args.train
        )

    columns = required_columns(
        args.train
    )

    print(
        "Reading:",
        args.train,
    )

    train = pd.read_parquet(
        args.train,
        columns=columns,
    )

    train["inchikey14"] = (
        train["inchikey14"]
        .astype(str)
    )

    selected = choose_structures(
        train,
        args.structures,
        args.seed,
    )

    selected_keys = set(
        selected[
            "inchikey14"
        ]
    )

    validation_rows = (
        train.loc[
            train[
                "inchikey14"
            ].isin(
                selected_keys
            )
        ]
        .copy()
    )

    validation_rows["_source_row"] = np.arange(
        len(validation_rows),
        dtype=np.int64,
    )

    sort_columns = [
        "inchikey14",
        "spectrum_id",
    ] if "spectrum_id" in validation_rows.columns else [
        "inchikey14",
        "_source_row",
    ]

    validation_rows = (
        validation_rows
        .sort_values(
            sort_columns
        )
        .groupby(
            "inchikey14",
            sort=False,
            group_keys=False,
        )
        .head(
            args.max_spectra_per_structure
        )
        .copy()
    )

    molecule_ids = {
        key: f"val_{i:04d}_{key}"
        for i, key in enumerate(
            selected[
                "inchikey14"
            ],
            start=1,
        )
    }

    validation_rows[
        "molecule_id"
    ] = validation_rows[
        "inchikey14"
    ].map(
        molecule_ids
    )

    if "spectrum_id" not in validation_rows.columns:
        validation_rows[
            "spectrum_id"
        ] = [
            f"val_spec_{i:07d}"
            for i in range(
                1,
                len(validation_rows) + 1,
            )
        ]

    validation_rows = validation_rows.drop(
        columns=[
            "_source_row",
        ],
        errors="ignore",
    )

    truth = selected.copy()
    truth[
        "molecule_id"
    ] = truth[
        "inchikey14"
    ].map(
        molecule_ids
    )

    truth = truth[
        [
            "molecule_id",
            "inchikey14",
            "normalized_smiles",
        ]
    ].sort_values(
        "molecule_id"
    )

    # The validation test set must not expose truth-only columns.
    test_drop = [
        column
        for column in [
            "inchikey14",
            "normalized_smiles",
            "molecular_formula",
        ]
        if column in validation_rows.columns
    ]

    test = validation_rows.drop(
        columns=test_drop
    )

    # Purge every spectrum belonging to a selected validation structure.
    purged_train = train.loc[
        ~train[
            "inchikey14"
        ].isin(
            selected_keys
        )
    ].copy()

    sample_submission = pd.DataFrame(
        {
            "molecule_id":
                truth[
                    "molecule_id"
                ].tolist(),
            "smiles":
                ["CCO"]
                * len(
                    truth
                ),
        }
    )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    purged_train.to_parquet(
        args.output_dir
        / "train.parquet",
        index=False,
    )

    test.to_parquet(
        args.output_dir
        / "test.parquet",
        index=False,
    )

    truth.to_parquet(
        args.output_dir
        / "truth.parquet",
        index=False,
    )

    sample_submission.to_csv(
        args.output_dir
        / "sample_submission.csv",
        index=False,
    )

    print()
    print(
        "V4 validation split"
    )
    print(
        "=" * 72
    )
    print(
        "Structures:",
        f"{len(truth):,}",
    )
    print(
        "Validation spectra:",
        f"{len(test):,}",
    )
    print(
        "Purged training spectra:",
        f"{len(purged_train):,}",
    )
    print(
        "Validation structures still in train:",
        int(
            purged_train[
                "inchikey14"
            ].isin(
                selected_keys
            ).sum()
        ),
    )
    print()
    print(
        "Wrote:",
        args.output_dir
        / "train.parquet",
    )
    print(
        "Wrote:",
        args.output_dir
        / "test.parquet",
    )
    print(
        "Wrote:",
        args.output_dir
        / "sample_submission.csv",
    )
    print(
        "Wrote:",
        args.output_dir
        / "truth.parquet",
    )


if __name__ == "__main__":
    main()
