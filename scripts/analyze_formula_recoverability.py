from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from casmi26.chemistry.formula import (
    MONOISOTOPIC_MASSES,
    monoisotopic_mass,
    parse_formula,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent

ROOT = (
    PROJECT_ROOT
    / "data/processed/blind_pseudotest"
)

MISSING_PATH = (
    ROOT
    / "missing_candidate_truths.parquet"
)

TEST_PATH = (
    ROOT
    / "test.parquet"
)

TRAINING_STRUCTURES_PATH = (
    PROJECT_ROOT
    / "data/processed/structure_index/"
    "training_structures.parquet"
)

OUTPUT_PATH = (
    ROOT
    / "formula_recoverability.parquet"
)

SUMMARY_PATH = (
    ROOT
    / "formula_recoverability_summary.csv"
)


TARGET_RECOVERY_CLASS = (
    "recoverable_by_expanding_formula_asset"
)

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

# Conventional valences used only for a broad DBE sanity check.
DBE_VALENCE = {
    "C": 4,
    "H": 1,
    "N": 3,
    "P": 3,
    "F": 1,
    "Cl": 1,
    "Br": 1,
    "I": 1,
    "B": 3,
    "Si": 4,
}

DBE_IGNORED = {
    "O",
    "S",
    "Se",
}


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


def select_query_spectra(
    group: pd.DataFrame,
) -> pd.DataFrame:
    protonated = group.loc[
        group[
            "adduct"
        ].astype(str)
        == PRIMARY_ADDUCT
    ]

    if not protonated.empty:
        return protonated.copy()

    return group.loc[
        group[
            "adduct"
        ].astype(str)
        .isin(
            ADDUCT_SHIFTS
        )
    ].copy()


def formula_dbe(
    composition: dict[str, int],
) -> float | None:
    unknown = (
        set(
            composition
        )
        - set(
            DBE_VALENCE
        )
        - DBE_IGNORED
    )

    if unknown:
        return None

    total = 2.0

    for element, count in (
        composition.items()
    ):
        if element in DBE_IGNORED:
            continue

        valence = DBE_VALENCE[
            element
        ]

        total += (
            count
            * (
                valence
                - 2
            )
        )

    return total / 2.0


def build_training_formula_rules() -> tuple[
    set[str],
    dict[str, int],
    pd.DataFrame,
]:
    training = pd.read_parquet(
        TRAINING_STRUCTURES_PATH,
        columns=[
            "molecular_formula",
        ],
    )

    formulas = (
        training[
            "molecular_formula"
        ]
        .dropna()
        .astype(str)
        .drop_duplicates()
    )

    element_max: Counter[
        str
    ] = Counter()

    element_formula_counts: Counter[
        str
    ] = Counter()

    parsed_count = 0
    invalid_count = 0

    for formula in formulas:
        try:
            composition = (
                parse_formula(
                    formula
                )
            )
        except ValueError:
            invalid_count += 1
            continue

        parsed_count += 1

        for (
            element,
            count,
        ) in composition.items():
            element_formula_counts[
                element
            ] += 1

            element_max[
                element
            ] = max(
                element_max[
                    element
                ],
                int(
                    count
                ),
            )

    allowed_elements = set(
        element_max
    )

    rules = pd.DataFrame(
        [
            {
                "element":
                    element,
                "max_training_count":
                    element_max[
                        element
                    ],
                "training_formula_count":
                    element_formula_counts[
                        element
                    ],
                "monoisotopic_mass":
                    MONOISOTOPIC_MASSES[
                        element
                    ],
            }
            for element
            in sorted(
                allowed_elements
            )
        ]
    )

    print(
        "Training-only formula rules"
    )
    print(
        "=" * 76
    )
    print(
        "Unique training formulas:",
        f"{len(formulas):,}",
    )
    print(
        "Parsed formulas:",
        f"{parsed_count:,}",
    )
    print(
        "Invalid formulas:",
        f"{invalid_count:,}",
    )
    print(
        "Allowed elements:",
        ", ".join(
            sorted(
                allowed_elements
            )
        ),
    )
    print()
    print(
        rules.to_string(
            index=False
        )
    )

    return (
        allowed_elements,
        dict(
            element_max
        ),
        rules,
    )


def main() -> None:
    for path in [
        MISSING_PATH,
        TEST_PATH,
        TRAINING_STRUCTURES_PATH,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"Missing required artifact: {path}"
            )

    missing = pd.read_parquet(
        MISSING_PATH
    ).copy()

    targets = missing.loc[
        missing[
            "recovery_class"
        ].astype(str)
        == TARGET_RECOVERY_CLASS
    ].copy()

    if targets.empty:
        raise RuntimeError(
            "No recoverable-by-expanding-formula-asset cases found"
        )

    test = pd.read_parquet(
        TEST_PATH
    ).copy()

    test[
        "molecule_id"
    ] = (
        test[
            "molecule_id"
        ]
        .astype(str)
    )

    targets[
        "molecule_id"
    ] = (
        targets[
            "molecule_id"
        ]
        .astype(str)
    )

    (
        allowed_elements,
        element_max,
        rules,
    ) = (
        build_training_formula_rules()
    )

    rules_path = (
        ROOT
        / "training_formula_rules.csv"
    )

    rules.to_csv(
        rules_path,
        index=False,
    )

    rows = []

    for target in targets.itertuples(
        index=False
    ):
        molecule_id = str(
            target.molecule_id
        )

        truth_formula = str(
            target.molecular_formula
        )

        spectra = test.loc[
            test[
                "molecule_id"
            ]
            == molecule_id
        ].copy()

        selected = (
            select_query_spectra(
                spectra
            )
        )

        try:
            composition = (
                parse_formula(
                    truth_formula
                )
            )

            truth_mass = (
                monoisotopic_mass(
                    truth_formula
                )
            )

            parse_ok = True

        except ValueError:
            composition = {}
            truth_mass = np.nan
            parse_ok = False

        formula_elements = set(
            composition
        )

        unsupported_elements = sorted(
            formula_elements
            - allowed_elements
        )

        count_violations = {
            element:
                {
                    "truth":
                        int(
                            count
                        ),
                    "training_max":
                        int(
                            element_max.get(
                                element,
                                0,
                            )
                        ),
                }
            for (
                element,
                count,
            ) in composition.items()
            if (
                element
                not in element_max
                or count
                > element_max[
                    element
                ]
            )
        }

        dbe = (
            formula_dbe(
                composition
            )
            if parse_ok
            else None
        )

        dbe_valid = (
            dbe is not None
            and np.isfinite(
                dbe
            )
            and dbe >= 0
            and np.isclose(
                dbe,
                round(
                    dbe
                ),
                atol=1e-9,
            )
        )

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

            error = (
                ppm_error(
                    neutral_mass,
                    truth_mass,
                )
                if parse_ok
                else np.nan
            )

            spectrum_rows.append(
                {
                    "adduct":
                        str(
                            spectrum.adduct
                        ),
                    "neutral_mass":
                        neutral_mass,
                    "ppm_error":
                        error,
                }
            )

        finite_errors = [
            abs(
                float(
                    item[
                        "ppm_error"
                    ]
                )
            )
            for item
            in spectrum_rows
            if np.isfinite(
                item[
                    "ppm_error"
                ]
            )
        ]

        best_abs_ppm = (
            min(
                finite_errors
            )
            if finite_errors
            else np.nan
        )

        within_5ppm = (
            np.isfinite(
                best_abs_ppm
            )
            and best_abs_ppm
            <= 5.0
        )

        within_10ppm = (
            np.isfinite(
                best_abs_ppm
            )
            and best_abs_ppm
            <= FORMULA_TOLERANCE_PPM
        )

        within_20ppm = (
            np.isfinite(
                best_abs_ppm
            )
            and best_abs_ppm
            <= 20.0
        )

        supported_elements = (
            parse_ok
            and not unsupported_elements
        )

        within_count_bounds = (
            parse_ok
            and not count_violations
        )

        recoverable_by_mass_rules = (
            parse_ok
            and supported_elements
            and within_count_bounds
            and dbe_valid
            and within_10ppm
        )

        if not parse_ok:
            failure_reason = (
                "formula_parse_failed"
            )
        elif unsupported_elements:
            failure_reason = (
                "unsupported_element"
            )
        elif count_violations:
            failure_reason = (
                "element_count_outside_training_bounds"
            )
        elif not dbe_valid:
            failure_reason = (
                "dbe_rule_failed"
            )
        elif not selected.shape[
            0
        ]:
            failure_reason = (
                "no_supported_inference_adduct"
            )
        elif not within_10ppm:
            failure_reason = (
                "truth_mass_outside_10ppm"
            )
        else:
            failure_reason = (
                "recoverable"
            )

        rows.append(
            {
                "molecule_id":
                    molecule_id,
                "inchikey14":
                    str(
                        target.inchikey14
                    ),
                "truth_formula":
                    truth_formula,
                "truth_mass":
                    truth_mass,
                "selected_spectrum_count":
                    len(
                        selected
                    ),
                "selected_adducts":
                    ";".join(
                        sorted(
                            set(
                                selected[
                                    "adduct"
                                ]
                                .dropna()
                                .astype(str)
                            )
                        )
                    ),
                "best_abs_truth_mass_error_ppm":
                    best_abs_ppm,
                "within_5ppm":
                    within_5ppm,
                "within_10ppm":
                    within_10ppm,
                "within_20ppm":
                    within_20ppm,
                "formula_parse_ok":
                    parse_ok,
                "supported_elements":
                    supported_elements,
                "within_training_element_count_bounds":
                    within_count_bounds,
                "unsupported_elements":
                    ";".join(
                        unsupported_elements
                    ),
                "count_violations":
                    ";".join(
                        (
                            f"{element}:"
                            f"{values['truth']}>"
                            f"{values['training_max']}"
                        )
                        for (
                            element,
                            values,
                        )
                        in sorted(
                            count_violations.items()
                        )
                    ),
                "dbe":
                    dbe,
                "dbe_valid":
                    dbe_valid,
                "recoverable_by_mass_rules":
                    recoverable_by_mass_rules,
                "failure_reason":
                    failure_reason,
            }
        )

    result = pd.DataFrame(
        rows
    )

    result.to_parquet(
        OUTPUT_PATH,
        index=False,
        compression="zstd",
    )

    summary = (
        result.groupby(
            "failure_reason",
            dropna=False,
        )
        .agg(
            queries=(
                "molecule_id",
                "size",
            ),
            formulas=(
                "truth_formula",
                "nunique",
            ),
            median_abs_ppm=(
                "best_abs_truth_mass_error_ppm",
                "median",
            ),
            max_abs_ppm=(
                "best_abs_truth_mass_error_ppm",
                "max",
            ),
        )
        .reset_index()
        .sort_values(
            [
                "queries",
                "failure_reason",
            ],
            ascending=[
                False,
                True,
            ],
        )
    )

    summary[
        "fraction"
    ] = (
        summary[
            "queries"
        ]
        / len(
            result
        )
    )

    summary.to_csv(
        SUMMARY_PATH,
        index=False,
    )

    print()
    print(
        "Formula recoverability audit"
    )
    print(
        "=" * 92
    )
    print(
        "Target PubChem-recoverable queries:",
        f"{len(result):,}",
    )
    print(
        "Recoverable by training-derived mass rules:",
        (
            f"{int(result['recoverable_by_mass_rules'].sum()):,} / "
            f"{len(result):,} "
            f"({result['recoverable_by_mass_rules'].mean():.2%})"
        ),
    )
    print(
        "Within 5 ppm:",
        (
            f"{int(result['within_5ppm'].sum()):,} / "
            f"{len(result):,} "
            f"({result['within_5ppm'].mean():.2%})"
        ),
    )
    print(
        "Within 10 ppm:",
        (
            f"{int(result['within_10ppm'].sum()):,} / "
            f"{len(result):,} "
            f"({result['within_10ppm'].mean():.2%})"
        ),
    )
    print(
        "Within 20 ppm:",
        (
            f"{int(result['within_20ppm'].sum()):,} / "
            f"{len(result):,} "
            f"({result['within_20ppm'].mean():.2%})"
        ),
    )
    print()
    print(
        summary[
            [
                "failure_reason",
                "queries",
                "fraction",
                "formulas",
                "median_abs_ppm",
                "max_abs_ppm",
            ]
        ]
        .to_string(
            index=False,
            formatters={
                "fraction":
                    lambda value:
                        f"{value:.2%}",
                "median_abs_ppm":
                    lambda value:
                        (
                            "n/a"
                            if pd.isna(
                                value
                            )
                            else f"{value:.3f}"
                        ),
                "max_abs_ppm":
                    lambda value:
                        (
                            "n/a"
                            if pd.isna(
                                value
                            )
                            else f"{value:.3f}"
                        ),
            },
        )
    )

    print()
    print(
        "Element-rule failures"
    )
    print(
        "---------------------"
    )

    failures = result.loc[
        ~result[
            "recoverable_by_mass_rules"
        ]
    ]

    if failures.empty:
        print(
            "(none)"
        )
    else:
        print(
            failures[
                [
                    "molecule_id",
                    "truth_formula",
                    "selected_adducts",
                    "best_abs_truth_mass_error_ppm",
                    "unsupported_elements",
                    "count_violations",
                    "dbe",
                    "failure_reason",
                ]
            ]
            .to_string(
                index=False
            )
        )

    print()
    print(
        "Wrote:",
        OUTPUT_PATH,
    )
    print(
        "Wrote:",
        SUMMARY_PATH,
    )
    print(
        "Wrote:",
        rules_path,
    )


if __name__ == "__main__":
    main()
