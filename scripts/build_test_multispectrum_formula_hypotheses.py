from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(
    "/home/colleen/PycharmProjects/Envada-CASMI-2026"
)

TEST_PATH = (
    PROJECT_ROOT
    / "data/raw/test.parquet"
)

FORMULA_INDEX_PATH = (
    PROJECT_ROOT
    / "data/processed/formula_index/training_coconut_formulas.parquet"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data/processed/test"
)

OUTPUT_PATH = (
    OUTPUT_DIR
    / "multispectrum_formula_hypotheses.parquet"
)


#
# Frozen validation settings.
#
PPM_TOLERANCE = 10.0


#
# Adduct mass shifts.
#
# neutral_mass = precursor_mz - shift
#
# For [M-H]- the shift is negative, so this correctly becomes:
#
#     neutral_mass = precursor_mz + proton_mass
#
ADDUCT_MASS_SHIFT = {
    "[M+H]+": +1.007276466621,
    "[M-H]-": -1.007276466621,
    "[M+Na]+": +22.989218,
    "[M+NH4]+": +18.033823,
    "[M+CH2O2-H]-": +44.998201,
}


def ppm_error(
    observed_mass: float,
    expected_mass: float,
) -> float:
    return (
        (observed_mass - expected_mass)
        / expected_mass
        * 1_000_000.0
    )


def mist_usable_peaks(
    mzs,
    intensities,
    precursor_mz: float,
) -> int:
    """
    Mirror the important MIST process_spec_file behavior.

    MIST keeps peaks satisfying approximately:

        mz <= precursor_mz + 1

    We additionally require finite m/z values and finite positive
    intensities.

    A spectrum is considered processable if at least one peak survives.
    """

    mzs = np.asarray(
        mzs,
        dtype=float,
    )

    intensities = np.asarray(
        intensities,
        dtype=float,
    )

    if (
        mzs.ndim != 1
        or intensities.ndim != 1
        or len(mzs) != len(intensities)
        or len(mzs) == 0
    ):
        return 0

    mask = (
        np.isfinite(mzs)
        & np.isfinite(intensities)
        & (intensities > 0)
        & (mzs <= precursor_mz + 1.0)
    )

    return int(
        mask.sum()
    )


def select_test_spectra(
    test: pd.DataFrame,
) -> pd.DataFrame:
    """
    Preserve the frozen H+ path whenever a molecule has [M+H]+ spectra.

    For molecules with no [M+H]+ spectra, fall back to all spectra whose
    adduct is explicitly supported by ADDUCT_MASS_SHIFT.
    """

    molecules_with_hplus = set(
        test.loc[
            test["adduct"] == "[M+H]+",
            "molecule_id",
        ]
    )

    def should_use(row) -> bool:
        if row.molecule_id in molecules_with_hplus:
            return row.adduct == "[M+H]+"

        return row.adduct in ADDUCT_MASS_SHIFT

    mask = [
        should_use(row)
        for row in test.itertuples(
            index=False
        )
    ]

    return test.loc[
        mask
    ].copy()


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    #
    # Load test data.
    #
    test = pd.read_parquet(
        TEST_PATH
    ).reset_index(
        drop=True
    )

    #
    # Assign stable IDs BEFORE filtering.
    #
    test[
        "source_test_row_id"
    ] = np.arange(
        len(test),
        dtype=np.int64,
    )

    print(
        "Test spectra:",
        len(test),
    )

    print(
        "Test molecules:",
        test[
            "molecule_id"
        ].nunique(),
    )

    print()
    print(
        "Adduct counts:"
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

    #
    # Select spectra.
    #
    # Molecules with H+:
    #     use H+ only
    #
    # Molecules without H+:
    #     use supported fallback adducts
    #
    selected = select_test_spectra(
        test
    )

    print()
    print(
        "Selected spectra:",
        len(selected),
    )

    print(
        "Molecules represented by selected spectra:",
        selected[
            "molecule_id"
        ].nunique(),
    )

    print()
    print(
        "Selected adduct counts:"
    )

    print(
        selected[
            "adduct"
        ]
        .value_counts(
            dropna=False
        )
        .to_string()
    )

    #
    # Determine MIST processability.
    #
    selected[
        "mist_usable_peak_count"
    ] = [
        mist_usable_peaks(
            row.ms2_mzs,
            row.ms2_normalized_intensities,
            float(
                row.precursor_mz
            ),
        )
        for row in selected.itertuples(
            index=False
        )
    ]

    selected[
        "mist_processable"
    ] = (
        selected[
            "mist_usable_peak_count"
        ]
        > 0
    )

    print()
    print(
        "MIST-processable selected spectra:",
        int(
            selected[
                "mist_processable"
            ].sum()
        ),
    )

    print(
        "Molecules with at least one processable selected spectrum:",
        selected.loc[
            selected[
                "mist_processable"
            ],
            "molecule_id",
        ].nunique(),
    )

    #
    # Calculate neutral mass using the selected spectrum's adduct.
    #
    selected[
        "observed_neutral_mass"
    ] = [
        float(
            row.precursor_mz
        )
        - ADDUCT_MASS_SHIFT[
            row.adduct
        ]
        for row in selected.itertuples(
            index=False
        )
    ]

    #
    # Load frozen training + approved COCONUT formula universe.
    #
    formula_index = pd.read_parquet(
        FORMULA_INDEX_PATH
    )[
        [
            "molecular_formula",
            "monoisotopic_mass",
        ]
    ].copy()

    formula_index = (
        formula_index
        .dropna()
        .drop_duplicates(
            subset=[
                "molecular_formula"
            ]
        )
        .sort_values(
            "monoisotopic_mass"
        )
        .reset_index(
            drop=True
        )
    )

    formula_masses = (
        formula_index[
            "monoisotopic_mass"
        ]
        .astype(float)
        .to_numpy()
    )

    formula_names = (
        formula_index[
            "molecular_formula"
        ]
        .astype(str)
        .to_numpy()
    )

    print()
    print(
        "Formula index entries:",
        len(formula_index),
    )

    #
    # Find every formula within the frozen ±10 ppm window for
    # every selected spectrum.
    #
    hypothesis_rows = []

    for spectrum in selected.itertuples(
        index=False
    ):
        observed_mass = float(
            spectrum.observed_neutral_mass
        )

        mass_delta = (
            observed_mass
            * PPM_TOLERANCE
            / 1_000_000.0
        )

        lower = (
            observed_mass
            - mass_delta
        )

        upper = (
            observed_mass
            + mass_delta
        )

        left = int(
            np.searchsorted(
                formula_masses,
                lower,
                side="left",
            )
        )

        right = int(
            np.searchsorted(
                formula_masses,
                upper,
                side="right",
            )
        )

        for idx in range(
            left,
            right,
        ):
            expected_mass = float(
                formula_masses[
                    idx
                ]
            )

            error_ppm = ppm_error(
                observed_mass,
                expected_mass,
            )

            hypothesis_rows.append(
                {
                    "molecule_id":
                        spectrum.molecule_id,

                    "candidate_formula":
                        formula_names[
                            idx
                        ],

                    "formula_mass":
                        expected_mass,

                    "source_test_row_id":
                        int(
                            spectrum.source_test_row_id
                        ),

                    "source_spectrum_id":
                        spectrum.spectrum_id,

                    "source_adduct":
                        spectrum.adduct,

                    "source_ionization_mode":
                        spectrum.ionization_mode,

                    "source_precursor_mz":
                        float(
                            spectrum.precursor_mz
                        ),

                    "source_raw_peak_count":
                        len(
                            spectrum.ms2_mzs
                        ),

                    "source_mist_usable_peak_count":
                        int(
                            spectrum.mist_usable_peak_count
                        ),

                    "source_mist_processable":
                        bool(
                            spectrum.mist_processable
                        ),

                    "mass_error_ppm":
                        float(
                            error_ppm
                        ),
                }
            )

    per_spectrum = pd.DataFrame(
        hypothesis_rows
    )

    print()
    print(
        "Raw spectrum/formula matches:",
        len(
            per_spectrum
        ),
    )

    if per_spectrum.empty:
        raise RuntimeError(
            "No formula hypotheses were generated."
        )

    #
    # Multi-spectrum union.
    #
    # Support is computed across every selected spectrum for the
    # molecule, regardless of adduct.
    #
    support = (
        per_spectrum.groupby(
            [
                "molecule_id",
                "candidate_formula",
            ]
        )
        .agg(
            formula_support_count=(
                "source_test_row_id",
                "nunique",
            ),
            best_abs_mass_error_ppm=(
                "mass_error_ppm",
                lambda values: float(
                    np.abs(
                        values
                    ).min()
                ),
            ),
        )
        .reset_index()
    )

    total_spectra = (
        selected.groupby(
            "molecule_id"
        )[
            "source_test_row_id"
        ]
        .nunique()
        .rename(
            "selected_spectrum_count"
        )
        .reset_index()
    )

    support = support.merge(
        total_spectra,
        on="molecule_id",
        how="left",
        validate="many_to_one",
    )

    support[
        "formula_support_fraction"
    ] = (
        support[
            "formula_support_count"
        ]
        / support[
            "selected_spectrum_count"
        ]
    )

    #
    # Frozen representative-spectrum rule:
    #
    # For every molecule/formula hypothesis:
    #
    #   1. consider only MIST-processable spectra
    #   2. choose smallest |ppm error|
    #   3. use source_test_row_id as deterministic tie-breaker
    #
    processable = per_spectrum[
        per_spectrum[
            "source_mist_processable"
        ]
    ].copy()

    processable[
        "abs_mass_error_ppm"
    ] = (
        processable[
            "mass_error_ppm"
        ].abs()
    )

    processable = processable.sort_values(
        [
            "molecule_id",
            "candidate_formula",
            "abs_mass_error_ppm",
            "source_test_row_id",
        ],
        ascending=[
            True,
            True,
            True,
            True,
        ],
    )

    representative = (
        processable
        .drop_duplicates(
            subset=[
                "molecule_id",
                "candidate_formula",
            ],
            keep="first",
        )
        .copy()
    )

    representative = representative.merge(
        support,
        on=[
            "molecule_id",
            "candidate_formula",
        ],
        how="left",
        validate="one_to_one",
    )

    #
    # Sort first so hypothesis IDs are deterministic.
    #
    representative = (
        representative
        .sort_values(
            [
                "molecule_id",
                "candidate_formula",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    #
    # Stable hypothesis names.
    #
    hypothesis_names = {}

    for molecule_id, group in (
        representative.groupby(
            "molecule_id",
            sort=False,
        )
    ):
        for local_index, row_index in enumerate(
            group.index
        ):
            hypothesis_names[
                row_index
            ] = (
                f"test_{molecule_id}_"
                f"formula_{local_index:03d}"
            )

    representative[
        "hypothesis_spec"
    ] = representative.index.map(
        hypothesis_names
    )

    #
    # Normalize selected-spectrum column names so downstream code
    # has a clear distinction between the hypothesis and its chosen
    # raw spectrum.
    #
    representative = representative.rename(
        columns={
            "mass_error_ppm":
                "selected_mass_error_ppm",

            "source_mist_usable_peak_count":
                "selected_mist_usable_peak_count",

            "source_raw_peak_count":
                "selected_raw_peak_count",

            "source_precursor_mz":
                "selected_precursor_mz",

            "source_spectrum_id":
                "selected_spectrum_id",

            "source_adduct":
                "selected_adduct",

            "source_ionization_mode":
                "selected_ionization_mode",
        }
    )

    output_columns = [
        "molecule_id",
        "hypothesis_spec",
        "candidate_formula",
        "formula_mass",
        "formula_support_count",
        "formula_support_fraction",
        "selected_spectrum_count",
        "best_abs_mass_error_ppm",
        "selected_mass_error_ppm",
        "source_test_row_id",
        "selected_spectrum_id",
        "selected_adduct",
        "selected_ionization_mode",
        "selected_precursor_mz",
        "selected_raw_peak_count",
        "selected_mist_usable_peak_count",
    ]

    representative = representative[
        output_columns
    ]

    #
    # Sanity checks.
    #
    if representative[
        "hypothesis_spec"
    ].duplicated().any():
        raise RuntimeError(
            "Duplicate hypothesis_spec values generated."
        )

    if (
        representative[
            "selected_mist_usable_peak_count"
        ]
        <= 0
    ).any():
        raise RuntimeError(
            "A selected representative spectrum is not MIST-processable."
        )

    if (
        representative[
            "selected_mass_error_ppm"
        ]
        .abs()
        > PPM_TOLERANCE + 1e-9
    ).any():
        raise RuntimeError(
            "Generated hypothesis outside frozen ppm tolerance."
        )

    representative.to_parquet(
        OUTPUT_PATH,
        index=False,
    )

    #
    # Summary.
    #
    print()
    print(
        "Test multi-spectrum formula hypotheses"
    )
    print(
        "--------------------------------------"
    )

    print(
        "Hypotheses:",
        len(
            representative
        ),
    )

    print(
        "Molecules with hypotheses:",
        representative[
            "molecule_id"
        ].nunique(),
    )

    print(
        "Total test molecules:",
        test[
            "molecule_id"
        ].nunique(),
    )

    formula_counts = (
        representative.groupby(
            "molecule_id"
        )
        .size()
    )

    print(
        "Formulas/molecule mean:",
        float(
            formula_counts.mean()
        ),
    )

    print(
        "Formulas/molecule median:",
        float(
            formula_counts.median()
        ),
    )

    print(
        "Formulas/molecule max:",
        int(
            formula_counts.max()
        ),
    )

    print(
        "Unique selected spectra:",
        representative[
            "source_test_row_id"
        ].nunique(),
    )

    print(
        "Minimum selected MIST-usable peaks:",
        int(
            representative[
                "selected_mist_usable_peak_count"
            ].min()
        ),
    )

    print(
        "Maximum selected |ppm error|:",
        float(
            representative[
                "selected_mass_error_ppm"
            ]
            .abs()
            .max()
        ),
    )

    print()
    print(
        "Selected representative adducts:"
    )

    print(
        representative[
            "selected_adduct"
        ]
        .value_counts(
            dropna=False
        )
        .to_string()
    )

    missing_molecules = sorted(
        set(
            test[
                "molecule_id"
            ]
        )
        - set(
            representative[
                "molecule_id"
            ]
        )
    )

    print()
    print(
        "Molecules without usable hypotheses:",
        len(
            missing_molecules
        ),
    )

    if missing_molecules:
        print(
            "Missing molecule IDs:"
        )

        for molecule_id in missing_molecules:
            print(
                " ",
                molecule_id,
            )

    print()
    print(
        f"Wrote {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()