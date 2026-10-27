from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


TRAIN_PATH = Path(
    "data/raw/train.parquet"
)

OUTPUT_ROOT = Path(
    "data/processed/domain_pseudotest"
)

LEARNED_RANKER_FEATURES_PATH = Path(
    "data/processed/diagnostics/large_validation_audit/"
    "true_formula_neighbor_benchmark/"
    "learned_ranker_top500/candidate_features.parquet"
)

DOMAIN_LIBRARY = "enveda-np-examples"

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
            "Create an instrument/domain-matched pseudo-test from "
            "enveda-np-examples while keeping truth sealed."
        )
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=OUTPUT_ROOT,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not TRAIN_PATH.exists():
        raise FileNotFoundError(
            f"Missing required artifact: {TRAIN_PATH}"
        )

    parquet = pq.ParquetFile(
        TRAIN_PATH
    )

    source_columns = [
        "inchikey14",
        "normalized_smiles",
        "molecular_formula",
        "ingest_lib",
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
        scanned += len(
            frame
        )

        matched = frame.loc[
            frame[
                "ingest_lib"
            ].astype(str)
            == DOMAIN_LIBRARY
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
            f"No spectra found for {DOMAIN_LIBRARY!r}"
        )

    domain = pd.concat(
        parts,
        ignore_index=True,
    )

    domain[
        "inchikey14"
    ] = (
        domain[
            "inchikey14"
        ]
        .astype(str)
    )

    all_domain_keys = sorted(
        set(
            domain[
                "inchikey14"
            ]
        )
    )

    ranker_keys: set[str] = set()

    if LEARNED_RANKER_FEATURES_PATH.exists():
        ranker_population = pd.read_parquet(
            LEARNED_RANKER_FEATURES_PATH,
            columns=[
                "inchikey14",
            ],
        )

        ranker_keys = set(
            ranker_population[
                "inchikey14"
            ]
            .dropna()
            .astype(str)
        )

    overlap = (
        set(
            all_domain_keys
        )
        & ranker_keys
    )

    evaluation_keys = sorted(
        set(
            all_domain_keys
        )
        - overlap
    )

    if not evaluation_keys:
        raise RuntimeError(
            "No domain structures remain after excluding "
            "learned-ranker benchmark overlap"
        )

    evaluation = domain.loc[
        domain[
            "inchikey14"
        ].isin(
            evaluation_keys
        )
    ].copy()

    key_to_molecule = {
        key:
            f"domain_{index:04d}"
        for index, key
        in enumerate(
            evaluation_keys,
            start=1,
        )
    }

    evaluation[
        "molecule_id"
    ] = (
        evaluation[
            "inchikey14"
        ]
        .map(
            key_to_molecule
        )
    )

    evaluation = (
        evaluation
        .sort_values(
            [
                "molecule_id",
            ],
            kind="stable",
        )
        .reset_index(
            drop=True
        )
    )

    evaluation[
        "spectrum_id"
    ] = (
        evaluation
        .groupby(
            "molecule_id",
            sort=False,
        )
        .cumcount()
        .map(
            lambda value:
                f"spectrum_{int(value) + 1:04d}"
        )
    )

    test = evaluation[
        TEST_COLUMNS
    ].copy()

    truth_counts = (
        evaluation[
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
                "inchikey14":
                    evaluation_keys,
                "molecule_id":
                    [
                        key_to_molecule[
                            key
                        ]
                        for key
                        in evaluation_keys
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

    if truth[
        [
            "normalized_smiles",
            "molecular_formula",
        ]
    ].isna().any(
        axis=None
    ):
        raise RuntimeError(
            "Domain truth metadata is incomplete"
        )

    exclusion = pd.DataFrame(
        {
            "inchikey14":
                all_domain_keys,
        }
    )

    sample_submission = (
        pd.DataFrame(
            {
                "molecule_id":
                    truth[
                        "molecule_id"
                    ],
                "smiles":
                    "",
            }
        )
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

    exclusion_path = (
        args.output_root
        / "excluded_structure_keys.parquet"
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

    exclusion.to_parquet(
        exclusion_path,
        index=False,
        compression="zstd",
    )

    sample_submission.to_csv(
        sample_path,
        index=False,
    )

    print()
    print(
        "Domain-matched blind pseudo-test"
    )
    print(
        "=" * 76
    )
    print(
        "Domain library:",
        DOMAIN_LIBRARY,
    )
    print(
        "Domain structures:",
        f"{len(all_domain_keys):,}",
    )
    print(
        "Excluded for learned-ranker overlap:",
        f"{len(overlap):,}",
    )
    print(
        "Evaluation structures:",
        f"{len(truth):,}",
    )
    print(
        "Evaluation spectra:",
        f"{len(test):,}",
    )
    print()
    print(
        "Instrument types:"
    )
    print(
        test[
            "instrument_type"
        ]
        .value_counts(
            dropna=False
        )
        .to_string()
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
        "All domain identities are excluded from retrieval/reference assets."
    )
    print(
        "Test:",
        test_path,
    )
    print(
        "Truth:",
        truth_path,
    )
    print(
        "Exclusion list:",
        exclusion_path,
    )


if __name__ == "__main__":
    main()
