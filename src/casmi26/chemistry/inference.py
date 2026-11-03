from __future__ import annotations

import numpy as np
import pandas as pd


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


def neutral_mass_from_adduct(
    precursor_mz: float,
    adduct: str,
) -> float:
    shift = ADDUCT_SHIFTS.get(
        str(adduct)
    )

    if shift is None:
        return np.nan

    try:
        precursor = float(
            precursor_mz
        )
    except (
        TypeError,
        ValueError,
    ):
        return np.nan

    if not np.isfinite(
        precursor
    ):
        return np.nan

    return precursor - shift


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
        or mzs.size
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


def select_query_spectra(
    test: pd.DataFrame,
) -> pd.DataFrame:
    """
    Production formula-gate spectrum selection.

    Per molecule:
    - if [M+H]+ spectra exist, use only those;
    - otherwise use spectra with a supported neutral-mass transform.
    """
    selected_parts = []

    for (
        molecule_id,
        group,
    ) in test.groupby(
        "molecule_id",
        sort=False,
    ):
        protonated = group.loc[
            group[
                "adduct"
            ].astype(str)
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
            selected = group.loc[
                group[
                    "adduct"
                ].astype(str)
                .isin(
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
            test.iloc[0:0]
            .copy()
        )

    return pd.concat(
        selected_parts,
        ignore_index=True,
    )
