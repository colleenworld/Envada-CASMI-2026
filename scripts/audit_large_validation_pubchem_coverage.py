from __future__ import annotations

from collections import defaultdict
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from casmi26.chemistry.adducts import neutral_mass


TOLERANCE_PPM = 10.0


def describe(values: pd.Series) -> str:
    if values.empty:
        return "n=0"

    return values.describe(
        percentiles=[
            0.25,
            0.5,
            0.75,
            0.9,
            0.95,
            0.99,
        ]
    ).to_string()


def main() -> None:
    project_root = Path(
        os.environ.get(
            "CASMI26_PROJECT_ROOT",
            Path(__file__).resolve().parent.parent,
        )
    )

    audit_dir = (
        project_root
        / "data/processed/diagnostics/large_validation_audit"
    )

    manifest_path = audit_dir / "manifest.parquet"
    audit_results_path = audit_dir / "results.parquet"
    hypotheses_path = audit_dir / "formula_hypotheses.parquet"

    pubchem_root = (
        project_root
        / "data/processed/pubchem_formula_index"
    )
    formula_buckets_path = (
        pubchem_root / "formula_buckets.parquet"
    )

    output_path = (
        audit_dir / "pubchem_coverage.parquet"
    )
    by_library_path = (
        audit_dir / "pubchem_by_library.csv"
    )
    by_adduct_path = (
        audit_dir / "pubchem_by_adduct.csv"
    )
    by_instrument_path = (
        audit_dir / "pubchem_by_instrument.csv"
    )

    train_path = (
        project_root / "data/raw/train.parquet"
    )
    formula_index_path = (
        project_root
        / "data/processed/formula_index/training_coconut_formulas.parquet"
    )

    for required in [
        manifest_path,
        audit_results_path,
        formula_buckets_path,
        train_path,
        formula_index_path,
    ]:
        if not required.exists():
            raise FileNotFoundError(
                f"Missing required artifact: {required}"
            )

    manifest = pd.read_parquet(
        manifest_path
    )

    audit = pd.read_parquet(
        audit_results_path
    )

    if len(manifest) != 10_000:
        print(
            "WARNING: audit manifest contains "
            f"{len(manifest):,} structures, not 10,000."
        )

    sample_keys = set(
        manifest[
            "inchikey14"
        ]
        .astype(str)
    )

    formulas = pd.read_parquet(
        formula_index_path
    )[
        [
            "molecular_formula",
            "monoisotopic_mass",
        ]
    ]

    formulas = (
        formulas
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
        formulas[
            "molecular_formula"
        ]
        .astype(str)
        .to_numpy()
    )
    formula_masses = (
        formulas[
            "monoisotopic_mass"
        ]
        .astype(float)
        .to_numpy()
    )

    #
    # Reconstruct the exact production-style 10 ppm formula hypotheses
    # for the frozen large-audit sample.
    #
    if hypotheses_path.exists():
        hypotheses = pd.read_parquet(
            hypotheses_path
        )

        print(
            "Using cached formula hypotheses:",
            f"{len(hypotheses):,}",
        )
    else:
        parquet = pq.ParquetFile(
            train_path
        )

        columns = [
            "inchikey14",
            "molecular_formula",
            "adduct",
            "precursor_mz",
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
                .isin(sample_keys)
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

        spectra = pd.concat(
            chunks,
            ignore_index=True,
        )

        print(
            "Audit spectra:",
            f"{len(spectra):,}",
        )

        hypothesis_rows: list[
            dict[str, object]
        ] = []

        for number, (
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

            support: dict[
                str,
                list[float],
            ] = defaultdict(list)

            supported_spectra = 0

            for row in group.itertuples(
                index=False
            ):
                try:
                    precursor_mz = float(
                        row.precursor_mz
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
                        row.adduct
                    ),
                )

                if (
                    mass is None
                    or not np.isfinite(mass)
                ):
                    continue

                supported_spectra += 1

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

                    support[
                        formula
                    ].append(
                        ppm_error
                    )

            for formula, errors in support.items():
                errors_array = np.asarray(
                    errors,
                    dtype=float,
                )

                hypothesis_rows.append(
                    {
                        "inchikey14":
                            structure_key,
                        "truth_formula":
                            truth_formula,
                        "candidate_formula":
                            formula,
                        "is_truth_formula":
                            (
                                formula
                                == truth_formula
                            ),
                        "supported_spectrum_count":
                            supported_spectra,
                        "formula_support_count":
                            len(errors_array),
                        "formula_support_fraction":
                            (
                                len(errors_array)
                                / supported_spectra
                                if supported_spectra
                                else np.nan
                            ),
                        "mass_error_best_abs_ppm":
                            float(
                                np.abs(
                                    errors_array
                                ).min()
                            ),
                    }
                )

            if (
                number % 1000 == 0
                or number == len(sample_keys)
            ):
                print(
                    "Built hypotheses:",
                    f"{number:,}/"
                    f"{len(sample_keys):,}",
                )

        hypotheses = pd.DataFrame(
            hypothesis_rows
        )

        hypotheses.to_parquet(
            hypotheses_path,
            index=False,
        )

        print(
            "Formula hypotheses:",
            f"{len(hypotheses):,}",
        )
        print(
            "Unique candidate formulas:",
            f"{hypotheses['candidate_formula'].nunique():,}",
        )
        print(
            f"Wrote {hypotheses_path}"
        )

    required_hypothesis_columns = {
        "inchikey14",
        "truth_formula",
        "candidate_formula",
        "is_truth_formula",
    }

    missing = (
        required_hypothesis_columns
        - set(hypotheses.columns)
    )

    if missing:
        raise ValueError(
            "Formula hypotheses missing columns: "
            f"{sorted(missing)}"
        )

    requested_formulas = set(
        hypotheses[
            "candidate_formula"
        ]
        .dropna()
        .astype(str)
    )

    formula_buckets = pd.read_parquet(
        formula_buckets_path
    )

    formula_buckets[
        "molecular_formula"
    ] = (
        formula_buckets[
            "molecular_formula"
        ]
        .astype(str)
    )

    indexed_formulas = set(
        formula_buckets[
            "molecular_formula"
        ]
    )

    available_formulas = (
        requested_formulas
        & indexed_formulas
    )

    missing_formulas = (
        requested_formulas
        - indexed_formulas
    )

    print()
    print(
        "Requested formulas:",
        f"{len(requested_formulas):,}",
    )
    print(
        "Present in PubChem inference asset:",
        f"{len(available_formulas):,}",
    )
    print(
        "Missing from PubChem inference asset:",
        f"{len(missing_formulas):,}",
    )

    #
    # Map formula -> audit query keys needing it. This lets us test
    # truth-structure membership while streaming PubChem buckets.
    #
    hypothesis_pairs = (
        hypotheses[
            [
                "inchikey14",
                "candidate_formula",
            ]
        ]
        .drop_duplicates()
    )

    queries_by_formula: dict[
        str,
        set[str],
    ] = defaultdict(set)

    for row in hypothesis_pairs.itertuples(
        index=False
    ):
        queries_by_formula[
            str(
                row.candidate_formula
            )
        ].add(
            str(
                row.inchikey14
            )
        )

    pubchem_count_by_formula: dict[
        str,
        int,
    ] = {}

    truth_structure_found: set[str] = set()

    selected_bucket_rows = formula_buckets.loc[
        formula_buckets[
            "molecular_formula"
        ].isin(
            available_formulas
        ),
        [
            "molecular_formula",
            "bucket",
        ],
    ]

    buckets = sorted(
        selected_bucket_rows[
            "bucket"
        ]
        .astype(int)
        .unique()
    )

    print()
    print(
        "Scanning PubChem buckets:",
        len(buckets),
    )

    for bucket_number, bucket in enumerate(
        buckets,
        start=1,
    ):
        bucket_formulas = set(
            selected_bucket_rows.loc[
                selected_bucket_rows[
                    "bucket"
                ].astype(int)
                == bucket,
                "molecular_formula",
            ].astype(str)
        )

        path = (
            pubchem_root
            / "candidates"
            / f"bucket-{bucket:03d}.parquet"
        )

        if not path.exists():
            raise FileNotFoundError(
                f"Missing PubChem bucket: {path}"
            )

        frame = pd.read_parquet(
            path,
            columns=[
                "molecular_formula",
                "inchikey14",
            ],
        )

        frame[
            "molecular_formula"
        ] = (
            frame[
                "molecular_formula"
            ]
            .astype(str)
        )
        frame[
            "inchikey14"
        ] = (
            frame[
                "inchikey14"
            ]
            .astype(str)
        )

        frame = frame.loc[
            frame[
                "molecular_formula"
            ].isin(
                bucket_formulas
            )
        ]

        if not frame.empty:
            frame = frame.drop_duplicates(
                subset=[
                    "molecular_formula",
                    "inchikey14",
                ]
            )

            counts = (
                frame.groupby(
                    "molecular_formula"
                )[
                    "inchikey14"
                ]
                .nunique()
            )

            for formula, count in counts.items():
                pubchem_count_by_formula[
                    str(formula)
                ] = int(
                    count
                )

            #
            # Check only the actual audit truth keys associated with each
            # formula, rather than constructing a query x candidate join.
            #
            for formula, formula_frame in frame.groupby(
                "molecular_formula",
                sort=False,
            ):
                expected_truth_keys = (
                    queries_by_formula.get(
                        str(formula),
                        set(),
                    )
                )

                if not expected_truth_keys:
                    continue

                present_keys = set(
                    formula_frame[
                        "inchikey14"
                    ]
                    .astype(str)
                )

                truth_structure_found.update(
                    expected_truth_keys
                    & present_keys
                )

        if (
            bucket_number % 25 == 0
            or bucket_number == len(buckets)
        ):
            print(
                "PubChem buckets:",
                f"{bucket_number:,}/"
                f"{len(buckets):,}",
            )

    #
    # Attach PubChem burden to every formula hypothesis.
    #
    hypotheses[
        "pubchem_candidate_count"
    ] = (
        hypotheses[
            "candidate_formula"
        ]
        .astype(str)
        .map(
            pubchem_count_by_formula
        )
        .fillna(0)
        .astype(np.int64)
    )

    hypotheses[
        "formula_in_pubchem_asset"
    ] = (
        hypotheses[
            "candidate_formula"
        ]
        .astype(str)
        .isin(
            available_formulas
        )
    )

    per_query = (
        hypotheses.groupby(
            "inchikey14"
        )
        .agg(
            pubchem_candidate_count=(
                "pubchem_candidate_count",
                "sum",
            ),
            pubchem_formula_count=(
                "formula_in_pubchem_asset",
                "sum",
            ),
        )
        .reset_index()
    )

    truth_formula_rows = (
        hypotheses.loc[
            hypotheses[
                "is_truth_formula"
            ].astype(bool),
            [
                "inchikey14",
                "candidate_formula",
                "formula_in_pubchem_asset",
                "pubchem_candidate_count",
            ],
        ]
        .drop_duplicates(
            subset=[
                "inchikey14",
            ]
        )
        .rename(
            columns={
                "candidate_formula":
                    "recovered_truth_formula",
                "formula_in_pubchem_asset":
                    "truth_formula_in_pubchem_asset",
                "pubchem_candidate_count":
                    "truth_formula_pubchem_candidate_count",
            }
        )
    )

    result = (
        audit.merge(
            per_query,
            on="inchikey14",
            how="left",
            validate="one_to_one",
        )
        .merge(
            truth_formula_rows,
            on="inchikey14",
            how="left",
            validate="one_to_one",
        )
    )

    result[
        "pubchem_candidate_count"
    ] = (
        result[
            "pubchem_candidate_count"
        ]
        .fillna(0)
        .astype(np.int64)
    )

    result[
        "pubchem_formula_count"
    ] = (
        result[
            "pubchem_formula_count"
        ]
        .fillna(0)
        .astype(np.int64)
    )

    result[
        "truth_formula_in_pubchem_asset"
    ] = (
        result[
            "truth_formula_in_pubchem_asset"
        ]
        .fillna(False)
        .astype(bool)
    )

    result[
        "truth_formula_pubchem_candidate_count"
    ] = (
        result[
            "truth_formula_pubchem_candidate_count"
        ]
        .fillna(0)
        .astype(np.int64)
    )

    result[
        "truth_structure_in_pubchem_hypotheses"
    ] = (
        result[
            "inchikey14"
        ]
        .astype(str)
        .isin(
            truth_structure_found
        )
    )

    result.to_parquet(
        output_path,
        index=False,
    )

    total = len(
        result
    )

    formula_recovered = (
        result[
            "truth_formula_recovered"
        ]
        .fillna(False)
        .astype(bool)
    )

    truth_formula_asset = (
        result[
            "truth_formula_in_pubchem_asset"
        ]
    )

    truth_structure_asset = (
        result[
            "truth_structure_in_pubchem_hypotheses"
        ]
    )

    print()
    print(
        "Large-validation PubChem coverage"
    )
    print(
        "================================="
    )
    print(
        "Structures:",
        f"{total:,}",
    )
    print(
        "Truth formula recovered:",
        f"{formula_recovered.mean():.2%}",
    )
    print(
        "Recovered truth formula present in PubChem asset:",
        f"{truth_formula_asset.mean():.2%}",
    )
    print(
        "Truth structure present under any generated "
        "PubChem formula hypothesis:",
        f"{truth_structure_asset.mean():.2%}",
    )

    conditional = result.loc[
        formula_recovered
    ]

    if not conditional.empty:
        print()
        print(
            "Conditional on truth formula being recovered"
        )
        print(
            "--------------------------------------------"
        )
        print(
            "Truth formula present in PubChem asset:",
            f"{conditional['truth_formula_in_pubchem_asset'].mean():.2%}",
        )
        print(
            "Truth structure present:",
            f"{conditional['truth_structure_in_pubchem_hypotheses'].mean():.2%}",
        )

    print()
    print(
        "PubChem candidates across all generated formulas/query"
    )
    print(
        "------------------------------------------------------"
    )
    print(
        describe(
            result[
                "pubchem_candidate_count"
            ]
        )
    )

    truth_formula_recovered_rows = result.loc[
        formula_recovered
    ]

    print()
    print(
        "PubChem candidates for recovered truth formula"
    )
    print(
        "----------------------------------------------"
    )
    print(
        describe(
            truth_formula_recovered_rows[
                "truth_formula_pubchem_candidate_count"
            ]
        )
    )

    def grouped_report(
        column: str,
        minimum_n: int = 25,
    ) -> pd.DataFrame:
        grouped = (
            result.groupby(
                column,
                dropna=False,
            )
            .agg(
                structures=(
                    "inchikey14",
                    "size",
                ),
                formula_recall=(
                    "truth_formula_recovered",
                    "mean",
                ),
                truth_formula_pubchem=(
                    "truth_formula_in_pubchem_asset",
                    "mean",
                ),
                truth_structure_pubchem=(
                    "truth_structure_in_pubchem_hypotheses",
                    "mean",
                ),
                median_pubchem_candidates=(
                    "pubchem_candidate_count",
                    "median",
                ),
                median_truth_formula_candidates=(
                    "truth_formula_pubchem_candidate_count",
                    "median",
                ),
            )
            .reset_index()
        )

        return (
            grouped.loc[
                grouped[
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

    for label, frame in [
        (
            "By dominant library",
            by_library,
        ),
        (
            "By dominant adduct",
            by_adduct,
        ),
        (
            "By dominant instrument",
            by_instrument,
        ),
    ]:
        print()
        print(label)
        print("-" * len(label))
        print(
            frame.to_string(
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
    for path in [
        hypotheses_path,
        output_path,
        by_library_path,
        by_adduct_path,
        by_instrument_path,
    ]:
        print(path)


if __name__ == "__main__":
    main()
