from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from casmi26.chemistry.formula import monoisotopic_mass
from scripts.build_test_multispectrum_formula_hypotheses import (
    FORMULA_TOLERANCE_PPM,
    is_mist_processable,
    neutral_mass_from_adduct,
    ppm_error,
    select_query_spectra,
)


PROJECT_ROOT = Path(
    os.environ.get(
        "CASMI26_PROJECT_ROOT",
        Path(__file__).resolve().parent.parent,
    )
)

TEST_PATH = PROJECT_ROOT / "data/raw/test.parquet"

PRODUCTION_HYPOTHESES_PATH = (
    PROJECT_ROOT
    / "data/processed/test/multispectrum_formula_hypotheses.parquet"
)

FUSED_FORMULAS_PATH = (
    PROJECT_ROOT
    / "data/processed/blind_pseudotest/"
      "production_msbuddy_fused_formulas.parquet"
)

OUTPUT_PATH = PRODUCTION_HYPOTHESES_PATH

TOP_K = 3


def main() -> None:
    for path in [
        TEST_PATH,
        PRODUCTION_HYPOTHESES_PATH,
        FUSED_FORMULAS_PATH,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"Missing required artifact: {path}"
            )

    test = (
        pd.read_parquet(TEST_PATH)
        .reset_index(drop=True)
        .copy()
    )

    test["molecule_id"] = test["molecule_id"].astype(str)
    test["spectrum_id"] = test["spectrum_id"].astype(str)
    test["source_test_row_id"] = np.arange(
        len(test),
        dtype=np.int64,
    )

    selected = select_query_spectra(test).copy()

    selected["neutral_mass"] = [
        neutral_mass_from_adduct(
            precursor_mz,
            adduct,
        )
        for precursor_mz, adduct
        in zip(
            selected["precursor_mz"],
            selected["adduct"],
            strict=True,
        )
    ]

    processability = [
        is_mist_processable(
            mzs,
            intensities,
            precursor_mz,
        )
        for mzs, intensities, precursor_mz
        in zip(
            selected["ms2_mzs"],
            selected["ms2_normalized_intensities"],
            selected["precursor_mz"],
            strict=True,
        )
    ]

    selected["mist_processable"] = [
        item[0]
        for item in processability
    ]
    selected["mist_usable_peak_count"] = [
        item[1]
        for item in processability
    ]

    production = pd.read_parquet(
        PRODUCTION_HYPOTHESES_PATH
    ).copy()

    production["molecule_id"] = (
        production["molecule_id"]
        .astype(str)
    )
    production["candidate_formula"] = (
        production["candidate_formula"]
        .astype(str)
    )

    production_lookup = {
        (
            str(row.molecule_id),
            str(row.candidate_formula),
        ): row
        for row
        in production.itertuples(index=False)
    }

    fused = pd.read_parquet(
        FUSED_FORMULAS_PATH
    ).copy()

    required_fused = {
        "molecule_id",
        "candidate_formula",
        "fusion_rank",
    }

    missing = required_fused - set(
        fused.columns
    )

    if missing:
        raise ValueError(
            "Fused formula artifact missing columns: "
            f"{sorted(missing)}"
        )

    fused["molecule_id"] = (
        fused["molecule_id"]
        .astype(str)
    )
    fused["candidate_formula"] = (
        fused["candidate_formula"]
        .astype(str)
    )

    fused = (
        fused.loc[
            fused["fusion_rank"] <= TOP_K
        ]
        .sort_values(
            [
                "molecule_id",
                "fusion_rank",
                "candidate_formula",
            ],
            kind="stable",
        )
        .reset_index(drop=True)
    )

    selected_by_molecule = {
        str(molecule_id):
            group.copy()
        for molecule_id, group
        in selected.groupby(
            "molecule_id",
            sort=False,
        )
    }

    rows = []
    reused = 0
    reconstructed = 0
    skipped = 0

    for fused_row in fused.itertuples(
        index=False
    ):
        molecule_id = str(
            fused_row.molecule_id
        )
        formula = str(
            fused_row.candidate_formula
        )

        existing = production_lookup.get(
            (
                molecule_id,
                formula,
            )
        )

        if existing is not None:
            row = existing._asdict()
            row["hypothesis_spec"] = (
                f"{molecule_id}"
                f"__fused_{int(fused_row.fusion_rank):04d}"
            )
            rows.append(row)
            reused += 1
            continue

        spectra = selected_by_molecule.get(
            molecule_id
        )

        if (
            spectra is None
            or spectra.empty
        ):
            skipped += 1
            continue

        try:
            theoretical_mass = float(
                monoisotopic_mass(
                    formula
                )
            )
        except Exception:
            skipped += 1
            continue

        candidates = spectra.copy()

        candidates[
            "selected_mass_error_ppm"
        ] = [
            ppm_error(
                observed,
                theoretical_mass,
            )
            for observed
            in candidates[
                "neutral_mass"
            ]
        ]

        candidates[
            "selected_abs_mass_error_ppm"
        ] = candidates[
            "selected_mass_error_ppm"
        ].abs()

        within_tolerance = candidates.loc[
            candidates[
                "selected_abs_mass_error_ppm"
            ]
            <= FORMULA_TOLERANCE_PPM
        ].copy()

        if within_tolerance.empty:
            skipped += 1
            continue

        support_count = int(
            within_tolerance[
                "spectrum_id"
            ].nunique()
        )

        selected_count = int(
            candidates[
                "spectrum_id"
            ].nunique()
        )

        representative = (
            within_tolerance
            .sort_values(
                [
                    "mist_processable",
                    "selected_abs_mass_error_ppm",
                    "mist_usable_peak_count",
                    "source_test_row_id",
                ],
                ascending=[
                    False,
                    True,
                    False,
                    True,
                ],
                kind="stable",
            )
            .iloc[0]
        )

        rows.append(
            {
                "molecule_id":
                    molecule_id,
                "source_test_row_id":
                    int(
                        representative[
                            "source_test_row_id"
                        ]
                    ),
                "source_spectrum_id":
                    str(
                        representative[
                            "spectrum_id"
                        ]
                    ),
                "selected_spectrum_id":
                    str(
                        representative[
                            "spectrum_id"
                        ]
                    ),
                "selected_adduct":
                    str(
                        representative[
                            "adduct"
                        ]
                    ),
                "selected_ionization_mode":
                    str(
                        representative[
                            "ionization_mode"
                        ]
                    ),
                "selection_reason":
                    str(
                        representative[
                            "selection_reason"
                        ]
                    ),
                "precursor_mz":
                    float(
                        representative[
                            "precursor_mz"
                        ]
                    ),
                "selected_precursor_mz":
                    float(
                        representative[
                            "precursor_mz"
                        ]
                    ),
                "neutral_mass":
                    float(
                        representative[
                            "neutral_mass"
                        ]
                    ),
                "molecular_formula":
                    formula,
                "candidate_formula":
                    formula,
                "formula_monoisotopic_mass":
                    theoretical_mass,
                "selected_mass_error_ppm":
                    float(
                        representative[
                            "selected_mass_error_ppm"
                        ]
                    ),
                "selected_abs_mass_error_ppm":
                    float(
                        representative[
                            "selected_abs_mass_error_ppm"
                        ]
                    ),
                "mist_processable":
                    bool(
                        representative[
                            "mist_processable"
                        ]
                    ),
                "mist_usable_peak_count":
                    int(
                        representative[
                            "mist_usable_peak_count"
                        ]
                    ),
                "spectrum_support_count":
                    support_count,
                "selected_spectrum_count":
                    selected_count,
                "spectrum_support_fraction":
                    (
                        support_count
                        / selected_count
                    ),
                "min_abs_mass_error_ppm":
                    float(
                        within_tolerance[
                            "selected_abs_mass_error_ppm"
                        ].min()
                    ),
                "hypothesis_spec":
                    (
                        f"{molecule_id}"
                        f"__fused_{int(fused_row.fusion_rank):04d}"
                    ),
            }
        )

        reconstructed += 1

    output = pd.DataFrame(
        rows
    )

    if output.empty:
        raise RuntimeError(
            "Fused formula gate produced no usable hypotheses"
        )

    duplicated = output[
        "hypothesis_spec"
    ].duplicated()

    if duplicated.any():
        raise RuntimeError(
            "Fused hypothesis_spec values are not unique"
        )

    expected_molecules = set(
        fused[
            "molecule_id"
        ]
    )

    actual_molecules = set(
        output[
            "molecule_id"
        ]
        .astype(str)
    )

    missing_molecules = sorted(
        expected_molecules
        - actual_molecules
    )

    print()
    print("Fused formula hypotheses")
    print("=" * 72)
    print(
        "Requested fused formulas:",
        f"{len(fused):,}",
    )
    print(
        "Output hypotheses:",
        f"{len(output):,}",
    )
    print(
        "Production hypotheses reused:",
        f"{reused:,}",
    )
    print(
        "MSBuddy-only hypotheses reconstructed:",
        f"{reconstructed:,}",
    )
    print(
        "Fused formulas skipped:",
        f"{skipped:,}",
    )
    print(
        "Molecules represented:",
        f"{len(actual_molecules):,}",
    )
    print(
        "Molecules missing entirely:",
        f"{len(missing_molecules):,}",
    )

    if missing_molecules:
        print(
            "First missing molecules:",
            missing_molecules[:20],
        )

    output.to_parquet(
        OUTPUT_PATH,
        index=False,
        compression="zstd",
    )

    print()
    print("Wrote:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
