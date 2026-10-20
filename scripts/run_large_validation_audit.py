from __future__ import annotations

from collections import defaultdict
import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from casmi26.chemistry.adducts import neutral_mass


SEED = 42
DEFAULT_SAMPLE_SIZE = 10_000
TOLERANCE_PPM = 10.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a large structure-disjoint audit of formula and "
            "candidate recovery without MIST."
        )
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=DEFAULT_SAMPLE_SIZE,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
    )
    return parser.parse_args()


def percentile_summary(values: pd.Series) -> str:
    if values.empty:
        return "n=0"

    return (
        values.describe(
            percentiles=[
                0.25,
                0.5,
                0.75,
                0.9,
                0.95,
                0.99,
            ]
        ).to_string()
    )


def main() -> None:
    args = parse_args()

    project_root = Path(
        os.environ.get(
            "CASMI26_PROJECT_ROOT",
            Path(__file__).resolve().parent.parent,
        )
    )

    train_path = project_root / "data/raw/train.parquet"
    validation_path = (
        project_root
        / "data/processed/splits/validation_structures.parquet"
    )
    formula_index_path = (
        project_root
        / "data/processed/formula_index/training_coconut_formulas.parquet"
    )
    training_structures_path = (
        project_root
        / "data/processed/structure_index/training_structures.parquet"
    )
    coconut_path = (
        project_root
        / "data/external/coconut/coconut_approved.parquet"
    )

    output_dir = (
        project_root
        / "data/processed/diagnostics/large_validation_audit"
    )
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest_path = output_dir / "manifest.parquet"
    results_path = output_dir / "results.parquet"
    by_library_path = output_dir / "by_library.csv"
    by_adduct_path = output_dir / "by_adduct.csv"
    by_instrument_path = output_dir / "by_instrument.csv"

    validation = pd.read_parquet(
        validation_path,
        columns=["inchikey14"],
    )

    validation_keys = (
        validation["inchikey14"]
        .dropna()
        .astype(str)
        .drop_duplicates()
        .sort_values()
        .to_numpy()
    )

    if args.sample_size <= 0:
        raise ValueError(
            "--sample-size must be positive"
        )

    sample_size = min(
        args.sample_size,
        len(validation_keys),
    )

    rng = np.random.default_rng(
        args.seed
    )

    selected = np.sort(
        rng.choice(
            validation_keys,
            size=sample_size,
            replace=False,
        )
    )

    sample_keys = set(
        selected.tolist()
    )

    manifest = pd.DataFrame(
        {
            "inchikey14": selected,
        }
    )

    manifest.to_parquet(
        manifest_path,
        index=False,
    )

    print(
        "Validation structures available:",
        f"{len(validation_keys):,}",
    )
    print(
        "Audit sample structures:",
        f"{len(selected):,}",
    )
    print(
        "Seed:",
        args.seed,
    )

    #
    # Formula catalog used by the production-style inference path.
    #
    formulas = pd.read_parquet(
        formula_index_path
    )

    required_formula_columns = {
        "molecular_formula",
        "monoisotopic_mass",
    }

    missing = (
        required_formula_columns
        - set(formulas.columns)
    )

    if missing:
        raise ValueError(
            "Formula index missing columns: "
            f"{sorted(missing)}"
        )

    formulas = (
        formulas[
            [
                "molecular_formula",
                "monoisotopic_mass",
            ]
        ]
        .dropna()
        .drop_duplicates(
            subset=["molecular_formula"]
        )
        .sort_values(
            "monoisotopic_mass"
        )
        .reset_index(drop=True)
    )

    formula_names = (
        formulas["molecular_formula"]
        .astype(str)
        .to_numpy()
    )
    formula_masses = (
        formulas["monoisotopic_mass"]
        .astype(float)
        .to_numpy()
    )

    #
    # Build the leakage-safe candidate structure catalog:
    # training split + COCONUT, preferring training on overlap and
    # excluding cross-catalog formula conflicts.
    #
    training = pd.read_parquet(
        training_structures_path
    )[
        [
            "inchikey14",
            "molecular_formula",
        ]
    ].copy()

    training["source"] = "training"

    coconut = pd.read_parquet(
        coconut_path
    )[
        [
            "inchikey14",
            "molecular_formula",
        ]
    ].copy()

    coconut["source"] = "coconut"

    overlap = training[
        [
            "inchikey14",
            "molecular_formula",
        ]
    ].merge(
        coconut[
            [
                "inchikey14",
                "molecular_formula",
            ]
        ],
        on="inchikey14",
        how="inner",
        suffixes=(
            "_training",
            "_coconut",
        ),
    )

    conflicts = set(
        overlap.loc[
            overlap[
                "molecular_formula_training"
            ]
            != overlap[
                "molecular_formula_coconut"
            ],
            "inchikey14",
        ].astype(str)
    )

    training = training.loc[
        ~training["inchikey14"]
        .astype(str)
        .isin(conflicts)
    ].copy()

    coconut = coconut.loc[
        ~coconut["inchikey14"]
        .astype(str)
        .isin(conflicts)
    ].copy()

    catalog = pd.concat(
        [
            training,
            coconut,
        ],
        ignore_index=True,
    )

    catalog["_prefer_training"] = (
        catalog["source"] == "training"
    )

    catalog = (
        catalog
        .sort_values(
            [
                "inchikey14",
                "_prefer_training",
            ],
            ascending=[
                True,
                False,
            ],
        )
        .drop_duplicates(
            subset=["inchikey14"],
            keep="first",
        )
    )

    formula_to_structures: dict[str, set[str]] = defaultdict(set)

    for row in catalog.itertuples(
        index=False
    ):
        formula_to_structures[
            str(row.molecular_formula)
        ].add(
            str(row.inchikey14)
        )

    print(
        "Leakage-safe candidate structures:",
        f"{len(catalog):,}",
    )
    print(
        "Candidate formulas represented:",
        f"{len(formula_to_structures):,}",
    )

    #
    # Collect every spectrum belonging to the 10k sampled structures.
    #
    parquet = pq.ParquetFile(
        train_path
    )

    columns = [
        "inchikey14",
        "molecular_formula",
        "ingest_lib",
        "ionization_mode",
        "instrument_type",
        "adduct",
        "precursor_mz",
        "num_peaks",
    ]

    chunks: list[pd.DataFrame] = []
    scanned = 0

    print(
        "Collecting audit spectra..."
    )

    for batch in parquet.iter_batches(
        batch_size=100_000,
        columns=columns,
    ):
        frame = batch.to_pandas()
        scanned += len(frame)

        matched = frame.loc[
            frame[
                "inchikey14"
            ]
            .astype(str)
            .isin(
                sample_keys
            )
        ]

        if not matched.empty:
            chunks.append(
                matched.copy()
            )

        print(
            f"\rRows scanned: {scanned:,}",
            end="",
            flush=True,
        )

    print()

    if not chunks:
        raise RuntimeError(
            "No spectra found for audit sample"
        )

    spectra = pd.concat(
        chunks,
        ignore_index=True,
    )

    print(
        "Audit spectra:",
        f"{len(spectra):,}",
    )

    rows: list[dict[str, object]] = []

    for index, (
        structure_key,
        group,
    ) in enumerate(
        spectra.groupby(
            "inchikey14",
            sort=False,
        ),
        start=1,
    ):
        structure_key = str(
            structure_key
        )

        truth_formula_counts = (
            group[
                "molecular_formula"
            ]
            .dropna()
            .astype(str)
            .value_counts()
        )

        truth_formula = (
            str(
                truth_formula_counts.index[0]
            )
            if not truth_formula_counts.empty
            else None
        )

        formula_support: dict[
            str,
            list[float],
        ] = defaultdict(list)

        supported_spectrum_count = 0
        unsupported_adduct_count = 0

        for spectrum in group.itertuples(
            index=False
        ):
            try:
                precursor_mz = float(
                    spectrum.precursor_mz
                )
            except (
                TypeError,
                ValueError,
            ):
                continue

            if not np.isfinite(
                precursor_mz
            ):
                continue

            mass = neutral_mass(
                precursor_mz,
                str(
                    spectrum.adduct
                ),
            )

            if mass is None:
                unsupported_adduct_count += 1
                continue

            if not np.isfinite(
                mass
            ):
                continue

            supported_spectrum_count += 1

            tolerance = (
                mass
                * TOLERANCE_PPM
                / 1e6
            )

            lo = np.searchsorted(
                formula_masses,
                mass - tolerance,
                side="left",
            )
            hi = np.searchsorted(
                formula_masses,
                mass + tolerance,
                side="right",
            )

            for formula_index in range(
                lo,
                hi,
            ):
                formula = str(
                    formula_names[
                        formula_index
                    ]
                )
                candidate_mass = float(
                    formula_masses[
                        formula_index
                    ]
                )
                ppm_error = (
                    (
                        candidate_mass
                        - mass
                    )
                    / mass
                    * 1e6
                )
                formula_support[
                    formula
                ].append(
                    ppm_error
                )

        candidate_formulas = set(
            formula_support
        )

        truth_formula_recovered = (
            truth_formula is not None
            and truth_formula
            in candidate_formulas
        )

        candidate_structures: set[str] = set()

        for formula in candidate_formulas:
            candidate_structures.update(
                formula_to_structures.get(
                    formula,
                    ()
                )
            )

        truth_structure_recovered = (
            structure_key
            in candidate_structures
        )

        truth_formula_candidate_count = (
            len(
                formula_to_structures.get(
                    truth_formula,
                    (),
                )
            )
            if truth_formula is not None
            else 0
        )

        truth_support_fraction = np.nan
        truth_best_abs_ppm = np.nan

        if (
            truth_formula is not None
            and truth_formula
            in formula_support
            and supported_spectrum_count
        ):
            errors = np.asarray(
                formula_support[
                    truth_formula
                ],
                dtype=float,
            )
            truth_support_fraction = (
                len(errors)
                / supported_spectrum_count
            )
            truth_best_abs_ppm = float(
                np.abs(
                    errors
                ).min()
            )

        libraries = (
            group[
                "ingest_lib"
            ]
            .dropna()
            .astype(str)
        )
        adducts = (
            group[
                "adduct"
            ]
            .dropna()
            .astype(str)
        )
        instruments = (
            group[
                "instrument_type"
            ]
            .dropna()
            .astype(str)
        )
        modes = (
            group[
                "ionization_mode"
            ]
            .dropna()
            .astype(str)
        )

        dominant_library = (
            str(
                libraries.value_counts().index[0]
            )
            if not libraries.empty
            else ""
        )
        dominant_adduct = (
            str(
                adducts.value_counts().index[0]
            )
            if not adducts.empty
            else ""
        )
        dominant_instrument = (
            str(
                instruments.value_counts().index[0]
            )
            if not instruments.empty
            else ""
        )

        rows.append(
            {
                "inchikey14":
                    structure_key,
                "truth_formula":
                    truth_formula,
                "spectrum_count":
                    len(group),
                "supported_spectrum_count":
                    supported_spectrum_count,
                "unsupported_adduct_count":
                    unsupported_adduct_count,
                "library_count":
                    libraries.nunique(),
                "adduct_count":
                    adducts.nunique(),
                "instrument_count":
                    instruments.nunique(),
                "ionization_mode_count":
                    modes.nunique(),
                "dominant_library":
                    dominant_library,
                "dominant_adduct":
                    dominant_adduct,
                "dominant_instrument":
                    dominant_instrument,
                "candidate_formula_count":
                    len(
                        candidate_formulas
                    ),
                "truth_formula_recovered":
                    truth_formula_recovered,
                "truth_formula_support_fraction":
                    truth_support_fraction,
                "truth_formula_best_abs_ppm":
                    truth_best_abs_ppm,
                "candidate_structure_count":
                    len(
                        candidate_structures
                    ),
                "truth_structure_recovered":
                    truth_structure_recovered,
                "truth_formula_candidate_count":
                    truth_formula_candidate_count,
            }
        )

        if (
            index % 1000 == 0
            or index == sample_size
        ):
            print(
                f"Audited structures: "
                f"{index:,}/"
                f"{sample_size:,}"
            )

    results = (
        pd.DataFrame(
            rows
        )
        .merge(
            manifest,
            on="inchikey14",
            how="right",
            validate="one_to_one",
        )
    )

    results.to_parquet(
        results_path,
        index=False,
    )

    evaluated = results.loc[
        results[
            "spectrum_count"
        ].notna()
    ].copy()

    print()
    print(
        "Large validation audit"
    )
    print(
        "======================"
    )
    print(
        "Sample structures:",
        f"{len(results):,}",
    )
    print(
        "Structures evaluated:",
        f"{len(evaluated):,}",
    )
    print(
        "Spectra:",
        f"{int(evaluated['spectrum_count'].sum()):,}",
    )

    formula_recall = float(
        evaluated[
            "truth_formula_recovered"
        ].fillna(False).mean()
    )
    structure_recall = float(
        evaluated[
            "truth_structure_recovered"
        ].fillna(False).mean()
    )

    print()
    print(
        "Truth formula recovered:",
        f"{formula_recall:.2%}",
    )
    print(
        "Truth structure recovered:",
        f"{structure_recall:.2%}",
    )

    print()
    print(
        "Candidate formulas per structure"
    )
    print(
        "--------------------------------"
    )
    print(
        percentile_summary(
            evaluated[
                "candidate_formula_count"
            ]
        )
    )

    print()
    print(
        "Candidate structures per structure"
    )
    print(
        "----------------------------------"
    )
    print(
        percentile_summary(
            evaluated[
                "candidate_structure_count"
            ]
        )
    )

    print()
    print(
        "Structures sharing the truth formula"
    )
    print(
        "------------------------------------"
    )
    print(
        percentile_summary(
            evaluated[
                "truth_formula_candidate_count"
            ]
        )
    )

    print()
    print(
        "Spectra per structure"
    )
    print(
        "---------------------"
    )
    print(
        percentile_summary(
            evaluated[
                "spectrum_count"
            ]
        )
    )

    def grouped_report(
        column: str,
        minimum_n: int = 25,
    ) -> pd.DataFrame:
        frame = (
            evaluated.groupby(
                column,
                dropna=False,
            )
            .agg(
                structures=(
                    "inchikey14",
                    "size",
                ),
                spectra=(
                    "spectrum_count",
                    "sum",
                ),
                formula_recall=(
                    "truth_formula_recovered",
                    "mean",
                ),
                structure_recall=(
                    "truth_structure_recovered",
                    "mean",
                ),
                median_formula_candidates=(
                    "candidate_formula_count",
                    "median",
                ),
                median_structure_candidates=(
                    "candidate_structure_count",
                    "median",
                ),
            )
            .reset_index()
        )

        return (
            frame.loc[
                frame[
                    "structures"
                ]
                >= minimum_n
            ]
            .sort_values(
                [
                    "structures",
                    column,
                ],
                ascending=[
                    False,
                    True,
                ],
            )
            .reset_index(
                drop=True
            )
        )

    by_library = grouped_report(
        "dominant_library"
    )
    by_adduct = grouped_report(
        "dominant_adduct"
    )
    by_instrument = grouped_report(
        "dominant_instrument"
    )

    by_library.to_csv(
        by_library_path,
        index=False,
    )
    by_adduct.to_csv(
        by_adduct_path,
        index=False,
    )
    by_instrument.to_csv(
        by_instrument_path,
        index=False,
    )

    print()
    print(
        "By dominant library"
    )
    print(
        "-------------------"
    )
    print(
        by_library.to_string(
            index=False
        )
    )

    print()
    print(
        "By dominant adduct"
    )
    print(
        "------------------"
    )
    print(
        by_adduct.to_string(
            index=False
        )
    )

    print()
    print(
        "By dominant instrument"
    )
    print(
        "----------------------"
    )
    print(
        by_instrument.to_string(
            index=False
        )
    )

    print()
    print(
        "Outputs"
    )
    print(
        "-------"
    )
    print(
        manifest_path
    )
    print(
        results_path
    )
    print(
        by_library_path
    )
    print(
        by_adduct_path
    )
    print(
        by_instrument_path
    )


if __name__ == "__main__":
    main()
