from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from casmi26.chemistry.formula_enumeration import (
    neutral_mass_from_adduct,
    select_query_spectra,
)


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

SHORTLIST_PATH = (
    PROJECT_ROOT
    / "data/processed/formula_index/"
    "generated_test_top25_formulas.parquet"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data/processed/test/"
    "multispectrum_formula_hypotheses.parquet"
)

TOP_K = int(
    os.environ.get(
        "CASMI26_GENERATED_FORMULA_TOP_K",
        "3",
    )
)


def is_mist_processable(
    mzs,
    intensities,
    precursor_mz: float,
) -> tuple[bool, int]:
    if mzs is None or intensities is None:
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
        or mzs.size != intensities.size
    ):
        return False, 0

    mask = (
        np.isfinite(mzs)
        & np.isfinite(intensities)
        & (intensities > 0)
    )

    if np.isfinite(precursor_mz):
        mask &= (
            mzs
            <= (
                float(precursor_mz)
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


def main() -> None:
    if TOP_K <= 0:
        raise ValueError(
            "CASMI26_GENERATED_FORMULA_TOP_K must be positive"
        )

    for path in [
        TEST_PATH,
        SHORTLIST_PATH,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"Missing required artifact: {path}"
            )

    test = (
        pd.read_parquet(
            TEST_PATH
        )
        .reset_index(
            drop=True
        )
        .copy()
    )

    required_test = {
        "molecule_id",
        "spectrum_id",
        "ms2_mzs",
        "ms2_normalized_intensities",
        "adduct",
        "ionization_mode",
        "precursor_mz",
    }

    missing_test = (
        required_test
        - set(test.columns)
    )

    if missing_test:
        raise ValueError(
            "Test parquet missing columns: "
            f"{sorted(missing_test)}"
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

    shortlist = pd.read_parquet(
        SHORTLIST_PATH
    ).copy()

    required_shortlist = {
        "molecule_id",
        "formula_rank",
        "molecular_formula",
        "monoisotopic_mass",
        "abs_mass_error_ppm",
        "log_formula_prior",
        "formula_score",
    }

    missing_shortlist = (
        required_shortlist
        - set(shortlist.columns)
    )

    if missing_shortlist:
        raise ValueError(
            "Generated formula shortlist missing columns: "
            f"{sorted(missing_shortlist)}"
        )

    shortlist[
        "molecule_id"
    ] = (
        shortlist[
            "molecule_id"
        ]
        .astype(str)
    )

    shortlist = (
        shortlist.loc[
            shortlist[
                "formula_rank"
            ]
            <= TOP_K
        ]
        .sort_values(
            [
                "molecule_id",
                "formula_rank",
            ],
            kind="stable",
        )
        .copy()
    )

    rows = []

    for molecule_id, group in test.groupby(
        "molecule_id",
        sort=False,
    ):
        formulas = shortlist.loc[
            shortlist[
                "molecule_id"
            ]
            == str(molecule_id)
        ]

        if formulas.empty:
            continue

        selected = select_query_spectra(
            group
        )

        if selected.empty:
            continue

        spectrum_rows = []

        for spectrum in selected.itertuples(
            index=False
        ):
            neutral_mass = (
                neutral_mass_from_adduct(
                    spectrum.precursor_mz,
                    spectrum.adduct,
                )
            )

            processable, usable_count = (
                is_mist_processable(
                    spectrum.ms2_mzs,
                    spectrum.ms2_normalized_intensities,
                    spectrum.precursor_mz,
                )
            )

            spectrum_rows.append(
                (
                    spectrum,
                    neutral_mass,
                    processable,
                    usable_count,
                )
            )

        for formula in formulas.itertuples(
            index=False
        ):
            theoretical_mass = float(
                formula.monoisotopic_mass
            )

            ranked_spectra = []

            for (
                spectrum,
                neutral_mass,
                processable,
                usable_count,
            ) in spectrum_rows:
                if (
                    not np.isfinite(
                        neutral_mass
                    )
                    or theoretical_mass <= 0
                ):
                    abs_ppm = np.inf
                    signed_ppm = np.nan
                else:
                    signed_ppm = (
                        (
                            neutral_mass
                            - theoretical_mass
                        )
                        / theoretical_mass
                        * 1_000_000.0
                    )

                    abs_ppm = abs(
                        signed_ppm
                    )

                ranked_spectra.append(
                    (
                        not processable,
                        abs_ppm,
                        -usable_count,
                        int(
                            spectrum.source_test_row_id
                        ),
                        spectrum,
                        neutral_mass,
                        processable,
                        usable_count,
                        signed_ppm,
                    )
                )

            ranked_spectra.sort(
                key=lambda item: (
                    item[0],
                    item[1],
                    item[2],
                    item[3],
                )
            )

            (
                _,
                abs_ppm,
                _,
                _,
                spectrum,
                neutral_mass,
                processable,
                usable_count,
                signed_ppm,
            ) = ranked_spectra[0]

            formula_rank = int(
                formula.formula_rank
            )

            spec = (
                f"{molecule_id}"
                f"__generated_formula_"
                f"{formula_rank:04d}"
            )

            rows.append(
                {
                    "molecule_id":
                        str(molecule_id),
                    "source_test_row_id":
                        int(
                            spectrum.source_test_row_id
                        ),
                    "source_spectrum_id":
                        str(
                            spectrum.spectrum_id
                        ),
                    "selected_spectrum_id":
                        str(
                            spectrum.spectrum_id
                        ),
                    "selected_adduct":
                        str(
                            spectrum.adduct
                        ),
                    "selected_ionization_mode":
                        str(
                            spectrum.ionization_mode
                        ),
                    "selection_reason":
                        "generated_formula_prior",
                    "precursor_mz":
                        float(
                            spectrum.precursor_mz
                        ),
                    "selected_precursor_mz":
                        float(
                            spectrum.precursor_mz
                        ),
                    "neutral_mass":
                        float(
                            neutral_mass
                        ),
                    "molecular_formula":
                        str(
                            formula.molecular_formula
                        ),
                    "candidate_formula":
                        str(
                            formula.molecular_formula
                        ),
                    "formula_monoisotopic_mass":
                        theoretical_mass,
                    "selected_mass_error_ppm":
                        float(
                            signed_ppm
                        ),
                    "selected_abs_mass_error_ppm":
                        float(
                            abs_ppm
                        ),
                    "mist_processable":
                        bool(
                            processable
                        ),
                    "mist_usable_peak_count":
                        int(
                            usable_count
                        ),
                    "spectrum_support_count":
                        1,
                    "selected_spectrum_count":
                        len(
                            selected
                        ),
                    "spectrum_support_fraction":
                        (
                            1.0
                            / len(
                                selected
                            )
                        ),
                    "min_abs_mass_error_ppm":
                        float(
                            abs_ppm
                        ),
                    "hypothesis_spec":
                        spec,
                    "formula_rank":
                        formula_rank,
                    "log_formula_prior":
                        float(
                            formula.log_formula_prior
                        ),
                    "formula_score":
                        float(
                            formula.formula_score
                        ),
                }
            )

    hypotheses = pd.DataFrame(
        rows
    )

    if hypotheses.empty:
        raise RuntimeError(
            "Generated formula shortlist produced no hypotheses"
        )

    if hypotheses[
        "hypothesis_spec"
    ].duplicated().any():
        raise ValueError(
            "Duplicate generated hypothesis_spec values"
        )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    hypotheses.to_parquet(
        OUTPUT_PATH,
        index=False,
        compression="zstd",
    )

    counts = (
        hypotheses.groupby(
            "molecule_id"
        )
        .size()
    )

    print()
    print(
        "Generated formula hypotheses"
    )
    print("=" * 92)
    print(
        "Configured Top-K:",
        TOP_K,
    )
    print(
        "Hypotheses:",
        f"{len(hypotheses):,}",
    )
    print(
        "Molecules:",
        f"{hypotheses['molecule_id'].nunique():,}",
    )
    print(
        "Unique formulas:",
        f"{hypotheses['candidate_formula'].nunique():,}",
    )
    print(
        "MIST-processable hypotheses:",
        f"{int(hypotheses['mist_processable'].sum()):,}",
    )
    print(
        "Formulas/molecule mean:",
        f"{counts.mean():.2f}",
    )
    print()
    print(
        "Wrote:",
        OUTPUT_PATH,
    )


if __name__ == "__main__":
    main()
