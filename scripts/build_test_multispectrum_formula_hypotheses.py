from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(
    os.environ.get(
        "CASMI26_PROJECT_ROOT",
        Path(__file__).resolve().parent.parent,
    )
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


# ---------------------------------------------------------------------------
# Frozen inference parameters
# ---------------------------------------------------------------------------

FORMULA_TOLERANCE_PPM = 10.0

ADDUCT_SHIFTS = {
    "[M+H]+": 1.007276466621,
    "[M-H]-": -1.007276466621,
    "[M+Na]+": 22.989218,
    "[M+K]+": 38.963158,
    "[M+NH4]+": 18.033823,
    "[M+CH2O2-H]-": 44.998201,
    "[M+Cl]-": 34.969402,
}

PRIMARY_ADDUCT = "[M+H]+"


# ---------------------------------------------------------------------------
# Private-rerun diagnostics
# ---------------------------------------------------------------------------

DEBUG_STAGE = (
    os.environ.get(
        "CASMI26_DEBUG_STAGE",
        "",
    )
    .strip()
)


def stop_after(
    stage: str,
) -> bool:
    if DEBUG_STAGE == stage:
        print()
        print(
            f"DEBUG: stopping successfully after {stage}"
        )
        return True

    return False


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def neutral_mass_from_adduct(
    precursor_mz: float,
    adduct: str,
) -> float:
    shift = ADDUCT_SHIFTS.get(
        adduct
    )

    if shift is None:
        return np.nan

    if not np.isfinite(
        precursor_mz
    ):
        return np.nan

    return (
        float(precursor_mz)
        - shift
    )


def ppm_error(
    observed_mass: float,
    theoretical_mass: float,
) -> float:
    if (
        not np.isfinite(
            observed_mass
        )
        or not np.isfinite(
            theoretical_mass
        )
        or theoretical_mass <= 0
    ):
        return np.nan

    return (
        (
            observed_mass
            - theoretical_mass
        )
        / theoretical_mass
        * 1_000_000.0
    )


def is_mist_processable(
    mzs,
    intensities,
    precursor_mz: float,
) -> tuple[bool, int]:
    if (
        mzs is None
        or intensities is None
    ):
        return False, 0

    mzs = np.asarray(
        mzs,
        dtype=np.float64,
    )

    intensities = np.asarray(
        intensities,
        dtype=np.float64,
    )

    if (
        mzs.size == 0
        or intensities.size == 0
    ):
        return False, 0

    if (
        mzs.size
        != intensities.size
    ):
        return False, 0

    mask = (
        np.isfinite(
            mzs
        )
        & np.isfinite(
            intensities
        )
        & (
            intensities
            > 0
        )
    )

    if np.isfinite(
        precursor_mz
    ):
        mask &= (
            mzs
            <= (
                float(
                    precursor_mz
                )
                + 1.0
            )
        )

    usable_count = int(
        mask.sum()
    )

    return (
        usable_count > 0,
        usable_count,
    )


def validate_formula_index(
    formula_index: pd.DataFrame,
) -> pd.DataFrame:
    required = {
        "molecular_formula",
        "monoisotopic_mass",
    }

    missing = (
        required
        - set(
            formula_index.columns
        )
    )

    if missing:
        raise ValueError(
            "Formula index is missing required columns: "
            f"{sorted(missing)}"
        )

    out = formula_index[
        [
            "molecular_formula",
            "monoisotopic_mass",
        ]
    ].copy()

    out = out.dropna(
        subset=[
            "molecular_formula",
            "monoisotopic_mass",
        ]
    )

    out[
        "molecular_formula"
    ] = (
        out[
            "molecular_formula"
        ]
        .astype(str)
    )

    out[
        "monoisotopic_mass"
    ] = pd.to_numeric(
        out[
            "monoisotopic_mass"
        ],
        errors="coerce",
    )

    out = out[
        np.isfinite(
            out[
                "monoisotopic_mass"
            ]
        )
    ].copy()

    out = out.drop_duplicates(
        subset=[
            "molecular_formula",
        ]
    )

    out = out.sort_values(
        "monoisotopic_mass"
    ).reset_index(
        drop=True
    )

    return out


def select_query_spectra(
    test: pd.DataFrame,
) -> pd.DataFrame:
    """
    Frozen selection rule:

    - if a molecule has [M+H]+ spectra, use only those;
    - otherwise use spectra whose adduct has a supported
      neutral-mass transform.
    """

    selected_parts = []

    for (
        molecule_id,
        group,
    ) in test.groupby(
        "molecule_id",
        sort=False,
    ):
        protonated = group[
            group[
                "adduct"
            ]
            == PRIMARY_ADDUCT
        ]

        if not protonated.empty:
            selected = (
                protonated.copy()
            )

            selected[
                "selection_reason"
            ] = "primary_hplus"

        else:
            selected = group[
                group[
                    "adduct"
                ].isin(
                    ADDUCT_SHIFTS
                )
            ].copy()

            selected[
                "selection_reason"
            ] = (
                "fallback_supported_adduct"
            )

        if not selected.empty:
            selected_parts.append(
                selected
            )

    if not selected_parts:
        return (
            test
            .iloc[0:0]
            .copy()
        )

    return pd.concat(
        selected_parts,
        ignore_index=True,
    )


def build_mass_lookup(
    formula_index: pd.DataFrame,
):
    formulas = (
        formula_index[
            "molecular_formula"
        ]
        .to_numpy()
    )

    masses = (
        formula_index[
            "monoisotopic_mass"
        ]
        .to_numpy(
            dtype=np.float64
        )
    )

    return (
        formulas,
        masses,
    )


def find_formula_matches(
    observed_neutral_mass: float,
    formulas: np.ndarray,
    masses: np.ndarray,
) -> list[
    tuple[
        str,
        float,
        float,
    ]
]:
    if not np.isfinite(
        observed_neutral_mass
    ):
        return []

    mass_delta = (
        observed_neutral_mass
        * FORMULA_TOLERANCE_PPM
        / 1_000_000.0
    )

    low = (
        observed_neutral_mass
        - mass_delta
    )

    high = (
        observed_neutral_mass
        + mass_delta
    )

    left = np.searchsorted(
        masses,
        low,
        side="left",
    )

    right = np.searchsorted(
        masses,
        high,
        side="right",
    )

    matches = []

    for i in range(
        left,
        right,
    ):
        theoretical = float(
            masses[i]
        )

        err = ppm_error(
            observed_neutral_mass,
            theoretical,
        )

        matches.append(
            (
                str(
                    formulas[i]
                ),
                theoretical,
                err,
            )
        )

    return matches


def empty_hypothesis_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "molecule_id",
            "source_test_row_id",
            "source_spectrum_id",
            "selected_spectrum_id",
            "selected_adduct",
            "selected_ionization_mode",
            "selection_reason",
            "precursor_mz",
            "selected_precursor_mz",
            "neutral_mass",
            "molecular_formula",
            "candidate_formula",
            "formula_monoisotopic_mass",
            "selected_mass_error_ppm",
            "selected_abs_mass_error_ppm",
            "mist_processable",
            "mist_usable_peak_count",
            "spectrum_support_count",
            "selected_spectrum_count",
            "spectrum_support_fraction",
            "min_abs_mass_error_ppm",
            "hypothesis_spec",
        ]
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "Project root:",
        PROJECT_ROOT,
    )

    print(
        "Test:",
        TEST_PATH,
    )

    print(
        "Formula index:",
        FORMULA_INDEX_PATH,
    )

    print(
        "Output:",
        OUTPUT_PATH,
    )

    if DEBUG_STAGE:
        print(
            "Debug stage:",
            DEBUG_STAGE,
        )

    print()

    # ------------------------------------------------------------------
    # Stage 1: load test
    # ------------------------------------------------------------------

    test = pd.read_parquet(
        TEST_PATH
    )

    required_test_columns = {
        "molecule_id",
        "spectrum_id",
        "ms2_mzs",
        "ms2_normalized_intensities",
        "adduct",
        "ionization_mode",
        "precursor_mz",
    }

    missing_test_columns = (
        required_test_columns
        - set(
            test.columns
        )
    )

    if missing_test_columns:
        raise ValueError(
            "Test parquet is missing columns: "
            f"{sorted(missing_test_columns)}"
        )

    test = (
        test
        .reset_index(
            drop=True
        )
        .copy()
    )

    test[
        "source_test_row_id"
    ] = np.arange(
        len(test),
        dtype=np.int64,
    )

    test[
        "molecule_id"
    ] = (
        test[
            "molecule_id"
        ]
        .astype(str)
    )

    test[
        "spectrum_id"
    ] = (
        test[
            "spectrum_id"
        ]
        .astype(str)
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
        "Test adducts:"
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

    if stop_after(
        "load_test"
    ):
        return

    # ------------------------------------------------------------------
    # Stage 2: formula index
    # ------------------------------------------------------------------

    formula_index = (
        pd.read_parquet(
            FORMULA_INDEX_PATH
        )
    )

    formula_index = (
        validate_formula_index(
            formula_index
        )
    )

    print()
    print(
        "Formula index entries:",
        len(
            formula_index
        ),
    )

    (
        formulas,
        masses,
    ) = build_mass_lookup(
        formula_index
    )

    if stop_after(
        "formula_index"
    ):
        return

    # ------------------------------------------------------------------
    # Stage 3: select spectra
    # ------------------------------------------------------------------

    selected = (
        select_query_spectra(
            test
        )
    )

    print()
    print(
        "Selected spectra:",
        len(selected),
    )

    print(
        "Selected molecules:",
        selected[
            "molecule_id"
        ].nunique(),
    )

    print()
    print(
        "Selected adducts:"
    )

    if not selected.empty:
        print(
            selected[
                "adduct"
            ]
            .value_counts(
                dropna=False
            )
            .to_string()
        )
    else:
        print(
            "(none)"
        )

    all_molecule_ids = set(
        test[
            "molecule_id"
        ]
    )

    selected_molecule_ids = set(
        selected[
            "molecule_id"
        ]
    )

    unsupported_molecules = sorted(
        all_molecule_ids
        - selected_molecule_ids
    )

    if unsupported_molecules:
        print()
        print(
            "WARNING: molecules with no supported adduct:",
            len(
                unsupported_molecules
            ),
        )

        print(
            test.loc[
                test[
                    "molecule_id"
                ].isin(
                    unsupported_molecules
                ),
                [
                    "molecule_id",
                    "ionization_mode",
                    "adduct",
                ],
            ]
            .drop_duplicates()
            .sort_values(
                [
                    "molecule_id",
                    "adduct",
                ]
            )
            .to_string(
                index=False
            )
        )

    if stop_after(
        "select_spectra"
    ):
        return

    if selected.empty:
        empty = (
            empty_hypothesis_frame()
        )

        empty.to_parquet(
            OUTPUT_PATH,
            index=False,
        )

        print()
        print(
            "No supported spectra were available."
        )

        print(
            f"Wrote empty hypothesis file: {OUTPUT_PATH}"
        )

        return

    # ------------------------------------------------------------------
    # Stage 4: neutral mass + processability
    # ------------------------------------------------------------------

    selected[
        "neutral_mass"
    ] = [
        neutral_mass_from_adduct(
            precursor_mz,
            adduct,
        )
        for (
            precursor_mz,
            adduct,
        )
        in zip(
            selected[
                "precursor_mz"
            ],
            selected[
                "adduct"
            ],
        )
    ]

    processability = [
        is_mist_processable(
            mzs,
            intensities,
            precursor_mz,
        )
        for (
            mzs,
            intensities,
            precursor_mz,
        )
        in zip(
            selected[
                "ms2_mzs"
            ],
            selected[
                "ms2_normalized_intensities"
            ],
            selected[
                "precursor_mz"
            ],
        )
    ]

    selected[
        "mist_processable"
    ] = [
        value[0]
        for value
        in processability
    ]

    selected[
        "mist_usable_peak_count"
    ] = [
        value[1]
        for value
        in processability
    ]

    print()
    print(
        "Selected spectra with finite neutral mass:",
        int(
            np.isfinite(
                selected[
                    "neutral_mass"
                ]
            ).sum()
        ),
    )

    print(
        "MIST-processable selected spectra:",
        int(
            selected[
                "mist_processable"
            ].sum()
        ),
    )

    if stop_after(
        "processability"
    ):
        return

    # ------------------------------------------------------------------
    # Stage 5: formula matching
    # ------------------------------------------------------------------

    rows = []

    raw_match_count = 0

    for row in selected.itertuples(
        index=False
    ):
        matches = (
            find_formula_matches(
                float(
                    row.neutral_mass
                ),
                formulas,
                masses,
            )
        )

        raw_match_count += len(
            matches
        )

        for (
            molecular_formula,
            theoretical_mass,
            signed_ppm,
        ) in matches:
            rows.append(
                {
                    "molecule_id":
                        row.molecule_id,

                    "source_test_row_id":
                        int(
                            row.source_test_row_id
                        ),

                    "source_spectrum_id":
                        row.spectrum_id,

                    "selected_adduct":
                        row.adduct,

                    "selected_ionization_mode":
                        row.ionization_mode,

                    "selection_reason":
                        row.selection_reason,

                    "precursor_mz":
                        float(
                            row.precursor_mz
                        ),

                    "neutral_mass":
                        float(
                            row.neutral_mass
                        ),

                    "molecular_formula":
                        molecular_formula,

                    "formula_monoisotopic_mass":
                        theoretical_mass,

                    "selected_mass_error_ppm":
                        signed_ppm,

                    "selected_abs_mass_error_ppm":
                        abs(
                            signed_ppm
                        ),

                    "mist_processable":
                        bool(
                            row.mist_processable
                        ),

                    "mist_usable_peak_count":
                        int(
                            row.mist_usable_peak_count
                        ),
                }
            )

    matches_df = pd.DataFrame(
        rows
    )

    print()
    print(
        "Raw formula matches:",
        raw_match_count,
    )

    if stop_after(
        "formula_matching"
    ):
        return

    if matches_df.empty:
        empty = (
            empty_hypothesis_frame()
        )

        empty.to_parquet(
            OUTPUT_PATH,
            index=False,
        )

        print()
        print(
            "WARNING: no formula hypotheses were generated."
        )

        print(
            f"Wrote empty hypothesis file: {OUTPUT_PATH}"
        )

        return

    # ------------------------------------------------------------------
    # Stage 6: support aggregation
    # ------------------------------------------------------------------

    support = (
        matches_df
        .groupby(
            [
                "molecule_id",
                "molecular_formula",
            ],
            as_index=False,
        )
        .agg(
            spectrum_support_count=(
                "source_spectrum_id",
                "nunique",
            ),

            min_abs_mass_error_ppm=(
                "selected_abs_mass_error_ppm",
                "min",
            ),
        )
    )

    selected_spectrum_counts = (
        selected
        .groupby(
            "molecule_id"
        )[
            "spectrum_id"
        ]
        .nunique()
        .rename(
            "selected_spectrum_count"
        )
        .reset_index()
    )

    support = support.merge(
        selected_spectrum_counts,
        on="molecule_id",
        how="left",
        validate="many_to_one",
    )

    support[
        "spectrum_support_fraction"
    ] = (
        support[
            "spectrum_support_count"
        ]
        / support[
            "selected_spectrum_count"
        ]
    )

    print()
    print(
        "Supported molecule/formula pairs:",
        len(
            support
        ),
    )

    if stop_after(
        "support"
    ):
        return

    # ------------------------------------------------------------------
    # Stage 7: choose representative spectra and build hypotheses
    # ------------------------------------------------------------------

    ranked = matches_df.sort_values(
        by=[
            "molecule_id",
            "molecular_formula",
            "mist_processable",
            "selected_abs_mass_error_ppm",
            "mist_usable_peak_count",
            "source_test_row_id",
        ],
        ascending=[
            True,
            True,
            False,
            True,
            False,
            True,
        ],
    )

    representatives = (
        ranked
        .drop_duplicates(
            subset=[
                "molecule_id",
                "molecular_formula",
            ],
            keep="first",
        )
        .copy()
    )

    hypotheses = (
        representatives
        .merge(
            support,
            on=[
                "molecule_id",
                "molecular_formula",
            ],
            how="left",
            validate="one_to_one",
        )
    )

    hypotheses = (
        hypotheses
        .sort_values(
            by=[
                "molecule_id",
                "min_abs_mass_error_ppm",
                "molecular_formula",
            ],
            ascending=[
                True,
                True,
                True,
            ],
        )
        .reset_index(
            drop=True
        )
    )

    hypothesis_number = (
        hypotheses
        .groupby(
            "molecule_id"
        )
        .cumcount()
    )

    hypotheses[
        "hypothesis_spec"
    ] = [
        (
            f"{molecule_id}"
            f"__formula_{number:04d}"
        )
        for (
            molecule_id,
            number,
        )
        in zip(
            hypotheses[
                "molecule_id"
            ],
            hypothesis_number,
        )
    ]

    #
    # Compatibility aliases required by export_test_mist_inputs.py.
    #
    hypotheses[
        "selected_spectrum_id"
    ] = (
        hypotheses[
            "source_spectrum_id"
        ]
    )

    hypotheses[
        "selected_precursor_mz"
    ] = (
        hypotheses[
            "precursor_mz"
        ]
    )

    hypotheses[
        "candidate_formula"
    ] = (
        hypotheses[
            "molecular_formula"
        ]
    )

    if stop_after(
        "hypotheses"
    ):
        return

    # ------------------------------------------------------------------
    # Stage 8: diagnostics + write
    # ------------------------------------------------------------------

    hypothesis_molecule_ids = set(
        hypotheses[
            "molecule_id"
        ]
    )

    missing_hypothesis_ids = sorted(
        all_molecule_ids
        - hypothesis_molecule_ids
    )

    print()
    print(
        "Formula hypothesis summary"
    )

    print(
        "--------------------------"
    )

    print(
        "Hypotheses:",
        len(
            hypotheses
        ),
    )

    print(
        "Molecules:",
        hypotheses[
            "molecule_id"
        ].nunique(),
    )

    print(
        "Unique formulas:",
        hypotheses[
            "molecular_formula"
        ].nunique(),
    )

    formula_counts = (
        hypotheses
        .groupby(
            "molecule_id"
        )
        .size()
    )

    if not formula_counts.empty:
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
        "MIST-processable hypotheses:",
        int(
            hypotheses[
                "mist_processable"
            ].sum()
        ),
    )

    print(
        "Selected unique spectra:",
        hypotheses[
            "source_spectrum_id"
        ].nunique(),
    )

    if not hypotheses.empty:
        print(
            "Min MIST usable peaks:",
            int(
                hypotheses[
                    "mist_usable_peak_count"
                ].min()
            ),
        )

        print(
            "Max abs ppm:",
            float(
                hypotheses[
                    "selected_abs_mass_error_ppm"
                ].max()
            ),
        )

    print()
    print(
        "Test molecules:",
        len(
            all_molecule_ids
        ),
    )

    print(
        "Molecules with formula hypotheses:",
        len(
            hypothesis_molecule_ids
        ),
    )

    print(
        "Missing after formula generation:",
        len(
            missing_hypothesis_ids
        ),
    )

    if missing_hypothesis_ids:
        print()
        print(
            "WARNING: test molecules with no formula hypotheses:",
            len(
                missing_hypothesis_ids
            ),
        )

        print()
        print(
            "Missing molecule adducts:"
        )

        print(
            test.loc[
                test[
                    "molecule_id"
                ].isin(
                    missing_hypothesis_ids
                ),
                [
                    "molecule_id",
                    "ionization_mode",
                    "adduct",
                ],
            ]
            .drop_duplicates()
            .sort_values(
                [
                    "molecule_id",
                    "adduct",
                ]
            )
            .to_string(
                index=False
            )
        )

        print()
        print(
            "These molecules will be handled later by "
            "same-polarity spectral retrieval."
        )

    hypotheses.to_parquet(
        OUTPUT_PATH,
        index=False,
    )

    print()
    print(
        f"Wrote {OUTPUT_PATH}"
    )

    if stop_after(
        "write"
    ):
        return


if __name__ == "__main__":
    main()