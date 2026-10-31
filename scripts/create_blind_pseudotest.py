from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


TRAIN_PATH = Path("data/raw/train.parquet")
VALIDATION_PATH = Path(
    "data/processed/splits/validation_structures.parquet"
)
OUTPUT_ROOT = Path(
    "data/processed/blind_pseudotest"
)

LEARNED_RANKER_FEATURES_PATH = Path(
    "data/processed/diagnostics/large_validation_audit/"
    "true_formula_neighbor_benchmark/"
    "learned_ranker_top500/candidate_features.parquet"
)

TEST_COLUMNS = [
    "molecule_id",
    "spectrum_id",
    "ms2_mzs",
    "ms2_normalized_intensities",
    "base_peak_intensity",
    "adduct",
    "ionization_mode",
    "instrument_type",
    "precursor_mz",
    "collision_energy_orig",
    "collision_energy_ev",
    "collision_energy_orig_units",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a truth-blind pseudo-test directly from the "
            "structure-disjoint validation split."
        )
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=1000,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20261026,
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=OUTPUT_ROOT,
    )
    args = parser.parse_args()

    if args.sample_size <= 0:
        parser.error("--sample-size must be positive")

    return args


def main() -> None:
    args = parse_args()

    for path in [
        TRAIN_PATH,
        VALIDATION_PATH,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"Missing required artifact: {path}"
            )

    validation = pd.read_parquet(
        VALIDATION_PATH,
        columns=["inchikey14"],
    )

    validation_keys = (
        validation["inchikey14"]
        .dropna()
        .astype(str)
        .drop_duplicates()
        .sort_values()
        .to_numpy(dtype=object)
    )

    excluded_ranker_keys: set[str] = set()

    if LEARNED_RANKER_FEATURES_PATH.exists():
        ranker_population = pd.read_parquet(
            LEARNED_RANKER_FEATURES_PATH,
            columns=["inchikey14"],
        )

        excluded_ranker_keys = set(
            ranker_population["inchikey14"]
            .dropna()
            .astype(str)
        )

    eligible_keys = np.asarray(
        [
            key
            for key in validation_keys
            if key not in excluded_ranker_keys
        ],
        dtype=object,
    )

    if len(eligible_keys) < args.sample_size:
        raise RuntimeError(
            "Not enough validation structures remain after excluding "
            "the learned-ranker benchmark population: "
            f"requested={args.sample_size:,}, eligible={len(eligible_keys):,}"
        )

    sample_size = args.sample_size

    rng = np.random.default_rng(
        args.seed
    )

    selected = np.sort(
        rng.choice(
            eligible_keys,
            size=sample_size,
            replace=False,
        )
    )

    key_to_molecule = {
        key: f"pseudo_{index:06d}"
        for index, key in enumerate(
            selected,
            start=1,
        )
    }

    selected_keys = set(
        key_to_molecule
    )

    parquet = pq.ParquetFile(
        TRAIN_PATH
    )

    source_columns = [
        "inchikey14",
        "normalized_smiles",
        "molecular_formula",
        "spectrum_id",
        "ms2_mzs",
        "ms2_normalized_intensities",
        "base_peak_intensity",
        "adduct",
        "ionization_mode",
        "instrument_type",
        "precursor_mz",
        "collision_energy_orig",
        "collision_energy_ev",
        "collision_energy_orig_units",
    ]

    parts = []
    scanned = 0

    for batch in parquet.iter_batches(
        batch_size=100_000,
        columns=source_columns,
    ):
        frame = batch.to_pandas()
        scanned += len(frame)

        matched = frame.loc[
            frame["inchikey14"]
            .astype(str)
            .isin(selected_keys)
        ].copy()

        if not matched.empty:
            parts.append(
                matched
            )

        print(
            f"\rRows scanned: {scanned:,}",
            end="",
            flush=True,
        )

    print()

    if not parts:
        raise RuntimeError(
            "No spectra found for selected validation structures"
        )

    spectra = pd.concat(
        parts,
        ignore_index=True,
    )

    found_keys = set(
        spectra["inchikey14"]
        .astype(str)
    )

    missing_keys = sorted(
        selected_keys
        - found_keys
    )

    if missing_keys:
        raise RuntimeError(
            "Selected validation structures were missing from train.parquet. "
            f"Examples: {missing_keys[:20]}"
        )

    spectra["molecule_id"] = (
        spectra["inchikey14"]
        .astype(str)
        .map(key_to_molecule)
    )

    # Preserve the deterministic scan order from train.parquet.
    # The original source spectrum identifier is not required for the
    # blind harness; assign anonymous per-molecule IDs instead.
    spectra = spectra.sort_values(
        [
            "molecule_id",
        ],
        kind="stable",
    ).reset_index(
        drop=True
    )

    spectra["spectrum_id"] = (
        spectra.groupby(
            "molecule_id",
            sort=False,
        )
        .cumcount()
        .map(
            lambda value:
                f"spectrum_{int(value) + 1:04d}"
        )
    )

    test = spectra[
        TEST_COLUMNS
    ].copy()

    # Resolve truth metadata directly from the raw validation spectra.
    # This avoids depending on a secondary structure index that may omit
    # otherwise valid validation structures.
    truth_observations = (
        spectra[
            [
                "inchikey14",
                "normalized_smiles",
                "molecular_formula",
            ]
        ]
        .dropna(
            subset=[
                "normalized_smiles",
                "molecular_formula",
            ]
        )
        .copy()
    )

    truth_observations[
        "inchikey14"
    ] = (
        truth_observations[
            "inchikey14"
        ]
        .astype(str)
    )

    truth_counts = (
        truth_observations
        .groupby(
            [
                "inchikey14",
                "normalized_smiles",
                "molecular_formula",
            ],
            dropna=False,
        )
        .size()
        .rename(
            "observation_count"
        )
        .reset_index()
        .sort_values(
            [
                "inchikey14",
                "observation_count",
                "normalized_smiles",
                "molecular_formula",
            ],
            ascending=[
                True,
                False,
                True,
                True,
            ],
            kind="stable",
        )
    )

    truth_metadata = (
        truth_counts
        .drop_duplicates(
            subset=[
                "inchikey14",
            ],
            keep="first",
        )
        [
            [
                "inchikey14",
                "normalized_smiles",
                "molecular_formula",
            ]
        ]
    )

    truth = (
        pd.DataFrame(
            {
                "inchikey14": selected,
                "molecule_id": [
                    key_to_molecule[
                        key
                    ]
                    for key in selected
                ],
            }
        )
        .merge(
            truth_metadata,
            on="inchikey14",
            how="left",
            validate="one_to_one",
        )
    )

    incomplete_truth = truth[
        truth[
            [
                "normalized_smiles",
                "molecular_formula",
            ]
        ].isna().any(axis=1)
    ]

    if not incomplete_truth.empty:
        raise RuntimeError(
            "Truth metadata is incomplete even in raw train.parquet for "
            f"{len(incomplete_truth):,} selected structures. "
            f"Examples: {incomplete_truth['inchikey14'].head(20).tolist()}"
        )

    sample_submission = pd.DataFrame(
        {
            "molecule_id": truth[
                "molecule_id"
            ],
            "smiles": "",
        }
    )

    args.output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    test_path = (
        args.output_root
        / "test.parquet"
    )
    truth_path = (
        args.output_root
        / "truth.parquet"
    )
    sample_path = (
        args.output_root
        / "sample_submission.csv"
    )

    test.to_parquet(
        test_path,
        index=False,
        compression="zstd",
    )

    truth.to_parquet(
        truth_path,
        index=False,
        compression="zstd",
    )

    sample_submission.to_csv(
        sample_path,
        index=False,
    )

    print()
    print("Blind pseudo-test")
    print("=" * 76)
    print(
        "Validation structures available:",
        f"{len(validation_keys):,}",
    )
    print(
        "Excluded learned-ranker benchmark structures:",
        f"{len(excluded_ranker_keys):,}",
    )
    print(
        "Eligible blind structures:",
        f"{len(eligible_keys):,}",
    )
    print(
        "Selected structures:",
        f"{len(truth):,}",
    )
    print(
        "Pseudo-test spectra:",
        f"{len(test):,}",
    )
    print(
        "Mean spectra/molecule:",
        f"{len(test) / len(truth):.2f}",
    )
    print()
    print(
        "Ionization modes:"
    )
    print(
        test[
            "ionization_mode"
        ]
        .value_counts(
            dropna=False
        )
        .to_string()
    )
    print()
    print(
        "Adducts:"
    )
    print(
        test[
            "adduct"
        ]
        .value_counts(
            dropna=False
        )
        .to_string()
    )
    print()
    print(
        "Truth is sealed separately from inference inputs."
    )
    print(
        "Test:",
        test_path,
    )
    print(
        "Sample submission:",
        sample_path,
    )
    print(
        "Truth:",
        truth_path,
    )


if __name__ == "__main__":
    main()
