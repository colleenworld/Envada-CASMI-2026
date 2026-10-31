from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from casmi26.chemistry.formula import (
    MONOISOTOPIC_MASSES,
    parse_formula,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent

ROOT = (
    PROJECT_ROOT
    / "data/processed/blind_pseudotest"
)

RECOVERABILITY_PATH = (
    ROOT
    / "formula_recoverability.parquet"
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
    / "blind_formula_enumeration.parquet"
)

SUMMARY_PATH = (
    ROOT
    / "blind_formula_enumeration_summary.csv"
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

ELEMENT_ORDER = [
    "C",
    "N",
    "O",
    "S",
    "P",
    "F",
    "Cl",
    "Br",
    "I",
    "B",
    "Si",
    "Se",
]

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


@dataclass(frozen=True)
class SignatureRule:
    elements: tuple[str, ...]
    maxima: dict[str, int]
    observed_formulas: int


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


def formula_from_composition(
    composition: dict[str, int],
) -> str:
    parts = []

    for element in [
        "C",
        "H",
        "B",
        "Br",
        "Cl",
        "F",
        "I",
        "N",
        "O",
        "P",
        "S",
        "Se",
        "Si",
    ]:
        count = int(
            composition.get(
                element,
                0,
            )
        )

        if count <= 0:
            continue

        parts.append(
            element
        )

        if count != 1:
            parts.append(
                str(
                    count
                )
            )

    return "".join(
        parts
    )


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

        total += (
            int(
                count
            )
            * (
                DBE_VALENCE[
                    element
                ]
                - 2
            )
        )

    return total / 2.0


def valid_dbe(
    composition: dict[str, int],
) -> bool:
    value = formula_dbe(
        composition
    )

    if (
        value is None
        or not np.isfinite(
            value
        )
        or value < 0
    ):
        return False

    return bool(
        np.isclose(
            value,
            round(
                value
            ),
            atol=1e-9,
        )
    )


def build_signature_rules() -> list[
    SignatureRule
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

    counts_by_signature: dict[
        tuple[str, ...],
        list[
            dict[str, int]
        ],
    ] = defaultdict(
        list
    )

    for formula in formulas:
        try:
            composition = parse_formula(
                formula
            )
        except ValueError:
            continue

        if (
            "C"
            not in composition
            or "H"
            not in composition
        ):
            continue

        signature = tuple(
            element
            for element in ELEMENT_ORDER
            if composition.get(
                element,
                0,
            )
            > 0
        )

        counts_by_signature[
            signature
        ].append(
            composition
        )

    rules = []

    for (
        signature,
        rows,
    ) in counts_by_signature.items():
        maxima = {
            element:
                max(
                    int(
                        row.get(
                            element,
                            0,
                        )
                    )
                    for row
                    in rows
                )
            for element
            in (
                set(
                    signature
                )
                | {
                    "H",
                }
            )
        }

        rules.append(
            SignatureRule(
                elements=
                    signature,
                maxima=
                    maxima,
                observed_formulas=
                    len(
                        rows
                    ),
            )
        )

    rules.sort(
        key=lambda rule: (
            -rule.observed_formulas,
            rule.elements,
        )
    )

    return rules


def enumerate_for_mass(
    neutral_mass: float,
    rules: list[
        SignatureRule
    ],
) -> list[
    tuple[
        str,
        float,
        float,
        int,
    ]
]:
    tolerance_da = (
        neutral_mass
        * FORMULA_TOLERANCE_PPM
        / 1_000_000.0
    )

    lower = (
        neutral_mass
        - tolerance_da
    )

    upper = (
        neutral_mass
        + tolerance_da
    )

    hydrogen_mass = (
        MONOISOTOPIC_MASSES[
            "H"
        ]
    )

    candidates: dict[
        str,
        tuple[
            float,
            float,
            int,
        ],
    ] = {}

    for rule_index, rule in enumerate(
        rules
    ):
        elements = [
            element
            for element in rule.elements
            if element != "H"
        ]

        # Exact branch-and-bound over non-hydrogen atoms. Hydrogen
        # is solved at each leaf from the residual exact mass.
        composition: dict[
            str,
            int
        ] = {}

        suffix_min_mass = [
            0.0
        ] * (
            len(
                elements
            )
            + 1
        )

        suffix_max_mass = [
            0.0
        ] * (
            len(
                elements
            )
            + 1
        )

        for index in range(
            len(
                elements
            )
            - 1,
            -1,
            -1,
        ):
            element = elements[
                index
            ]

            mass = (
                MONOISOTOPIC_MASSES[
                    element
                ]
            )

            # Elements in the signature are present, so minimum is 1.
            suffix_min_mass[
                index
            ] = (
                suffix_min_mass[
                    index
                    + 1
                ]
                + mass
            )

            suffix_max_mass[
                index
            ] = (
                suffix_max_mass[
                    index
                    + 1
                ]
                + (
                    mass
                    * rule.maxima[
                        element
                    ]
                )
            )

        max_h = int(
            rule.maxima.get(
                "H",
                0,
            )
        )

        def visit(
            index: int,
            current_mass: float,
        ) -> None:
            if index == len(
                elements
            ):
                residual = (
                    neutral_mass
                    - current_mass
                )

                h_estimate = int(
                    round(
                        residual
                        / hydrogen_mass
                    )
                )

                for h_count in range(
                    max(
                        1,
                        h_estimate - 1,
                    ),
                    min(
                        max_h,
                        h_estimate + 1,
                    )
                    + 1,
                ):
                    total_mass = (
                        current_mass
                        + (
                            h_count
                            * hydrogen_mass
                        )
                    )

                    if (
                        total_mass
                        < lower
                        or total_mass
                        > upper
                    ):
                        continue

                    full = dict(
                        composition
                    )

                    full[
                        "H"
                    ] = h_count

                    if not valid_dbe(
                        full
                    ):
                        continue

                    formula = (
                        formula_from_composition(
                            full
                        )
                    )

                    ppm = (
                        (
                            neutral_mass
                            - total_mass
                        )
                        / total_mass
                        * 1_000_000.0
                    )

                    current = (
                        candidates.get(
                            formula
                        )
                    )

                    value = (
                        total_mass,
                        ppm,
                        rule_index,
                    )

                    if (
                        current is None
                        or abs(
                            ppm
                        )
                        < abs(
                            current[
                                1
                            ]
                        )
                    ):
                        candidates[
                            formula
                        ] = value

                return

            element = elements[
                index
            ]

            element_mass = (
                MONOISOTOPIC_MASSES[
                    element
                ]
            )

            minimum_count = 1

            maximum_count = min(
                int(
                    rule.maxima[
                        element
                    ]
                ),
                int(
                    max(
                        0.0,
                        (
                            upper
                            - current_mass
                            - suffix_min_mass[
                                index
                                + 1
                            ]
                            - hydrogen_mass
                        )
                        / element_mass
                    )
                ),
            )

            if maximum_count < minimum_count:
                return

            for count in range(
                minimum_count,
                maximum_count
                + 1,
            ):
                next_mass = (
                    current_mass
                    + (
                        count
                        * element_mass
                    )
                )

                minimum_possible = (
                    next_mass
                    + suffix_min_mass[
                        index
                        + 1
                    ]
                    + hydrogen_mass
                )

                if (
                    minimum_possible
                    > upper
                ):
                    break

                maximum_possible = (
                    next_mass
                    + suffix_max_mass[
                        index
                        + 1
                    ]
                    + (
                        max_h
                        * hydrogen_mass
                    )
                )

                if (
                    maximum_possible
                    < lower
                ):
                    continue

                composition[
                    element
                ] = count

                visit(
                    index
                    + 1,
                    next_mass,
                )

                composition.pop(
                    element,
                    None,
                )

        visit(
            0,
            0.0,
        )

    ordered = sorted(
        [
            (
                formula,
                mass,
                ppm,
                rule_index,
            )
            for (
                formula,
                (
                    mass,
                    ppm,
                    rule_index,
                ),
            )
            in candidates.items()
        ],
        key=lambda item: (
            abs(
                item[
                    2
                ]
            ),
            item[
                0
            ],
        ),
    )

    return ordered


def main() -> None:
    for path in [
        RECOVERABILITY_PATH,
        TEST_PATH,
        TRAINING_STRUCTURES_PATH,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"Missing required artifact: {path}"
            )

    recoverability = pd.read_parquet(
        RECOVERABILITY_PATH
    ).copy()

    targets = recoverability.loc[
        recoverability[
            "recoverable_by_mass_rules"
        ]
        .fillna(False)
        .astype(bool)
    ].copy()

    if targets.empty:
        raise RuntimeError(
            "No mass-recoverable target formulas found"
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

    rules = (
        build_signature_rules()
    )

    print(
        "Blind formula enumeration"
    )
    print(
        "=" * 92
    )
    print(
        "Training-derived signatures:",
        f"{len(rules):,}",
    )
    print(
        "Target queries:",
        f"{len(targets):,}",
    )

    rows = []

    for number, target in enumerate(
        targets.itertuples(
            index=False
        ),
        start=1,
    ):
        molecule_id = str(
            target.molecule_id
        )

        truth_formula = str(
            target.truth_formula
        )

        selected = (
            select_query_spectra(
                test.loc[
                    test[
                        "molecule_id"
                    ]
                    == molecule_id
                ]
            )
        )

        merged: dict[
            str,
            tuple[
                float,
                float,
                str,
                int,
            ],
        ] = {}

        for spectrum in selected.itertuples(
            index=False
        ):
            neutral_mass = (
                neutral_mass_from_adduct(
                    spectrum.precursor_mz,
                    spectrum.adduct,
                )
            )

            if not np.isfinite(
                neutral_mass
            ):
                continue

            generated = (
                enumerate_for_mass(
                    neutral_mass,
                    rules,
                )
            )

            for (
                formula,
                mass,
                ppm,
                rule_index,
            ) in generated:
                existing = (
                    merged.get(
                        formula
                    )
                )

                candidate = (
                    mass,
                    ppm,
                    str(
                        spectrum.adduct
                    ),
                    rule_index,
                )

                if (
                    existing is None
                    or abs(
                        ppm
                    )
                    < abs(
                        existing[
                            1
                        ]
                    )
                ):
                    merged[
                        formula
                    ] = candidate

        ranked = sorted(
            merged.items(),
            key=lambda item: (
                abs(
                    item[
                        1
                    ][
                        1
                    ]
                ),
                item[
                    0
                ],
            ),
        )

        truth_rank = None
        truth_ppm = np.nan

        for rank, (
            formula,
            (
                _,
                ppm,
                _,
                _,
            ),
        ) in enumerate(
            ranked,
            start=1,
        ):
            if formula == truth_formula:
                truth_rank = rank
                truth_ppm = ppm
                break

        rows.append(
            {
                "molecule_id":
                    molecule_id,
                "truth_formula":
                    truth_formula,
                "generated_formula_count":
                    len(
                        ranked
                    ),
                "truth_generated":
                    truth_rank
                    is not None,
                "truth_formula_rank_by_abs_ppm":
                    truth_rank,
                "truth_formula_ppm_error":
                    truth_ppm,
                "top10_formula":
                    (
                        truth_rank
                        is not None
                        and truth_rank
                        <= 10
                    ),
                "top25_formula":
                    (
                        truth_rank
                        is not None
                        and truth_rank
                        <= 25
                    ),
                "top100_formula":
                    (
                        truth_rank
                        is not None
                        and truth_rank
                        <= 100
                    ),
            }
        )

        print(
            f"  {number:>2}/{len(targets)} "
            f"{molecule_id}: "
            f"{len(ranked):,} formulas | "
            f"truth rank={truth_rank}"
        )

    result = pd.DataFrame(
        rows
    )

    result.to_parquet(
        OUTPUT_PATH,
        index=False,
        compression="zstd",
    )

    summary = pd.DataFrame(
        [
            {
                "queries":
                    len(
                        result
                    ),
                "truth_generated":
                    int(
                        result[
                            "truth_generated"
                        ].sum()
                    ),
                "truth_generated_fraction":
                    result[
                        "truth_generated"
                    ].mean(),
                "top10":
                    int(
                        result[
                            "top10_formula"
                        ].sum()
                    ),
                "top10_fraction":
                    result[
                        "top10_formula"
                    ].mean(),
                "top25":
                    int(
                        result[
                            "top25_formula"
                        ].sum()
                    ),
                "top25_fraction":
                    result[
                        "top25_formula"
                    ].mean(),
                "top100":
                    int(
                        result[
                            "top100_formula"
                        ].sum()
                    ),
                "top100_fraction":
                    result[
                        "top100_formula"
                    ].mean(),
                "median_generated_formulas":
                    result[
                        "generated_formula_count"
                    ].median(),
                "p90_generated_formulas":
                    result[
                        "generated_formula_count"
                    ].quantile(
                        0.90
                    ),
                "max_generated_formulas":
                    result[
                        "generated_formula_count"
                    ].max(),
            }
        ]
    )

    summary.to_csv(
        SUMMARY_PATH,
        index=False,
    )

    row = summary.iloc[
        0
    ]

    print()
    print(
        "Blind formula enumeration summary"
    )
    print(
        "=" * 92
    )
    print(
        "Queries:",
        int(
            row[
                "queries"
            ]
        ),
    )
    print(
        "Truth generated:",
        (
            f"{int(row['truth_generated'])} / "
            f"{int(row['queries'])} "
            f"({row['truth_generated_fraction']:.2%})"
        ),
    )
    print(
        "Truth formula Top-10 by |ppm|:",
        (
            f"{int(row['top10'])} / "
            f"{int(row['queries'])} "
            f"({row['top10_fraction']:.2%})"
        ),
    )
    print(
        "Truth formula Top-25 by |ppm|:",
        (
            f"{int(row['top25'])} / "
            f"{int(row['queries'])} "
            f"({row['top25_fraction']:.2%})"
        ),
    )
    print(
        "Truth formula Top-100 by |ppm|:",
        (
            f"{int(row['top100'])} / "
            f"{int(row['queries'])} "
            f"({row['top100_fraction']:.2%})"
        ),
    )
    print(
        "Generated formulas/query median:",
        f"{row['median_generated_formulas']:.1f}",
    )
    print(
        "Generated formulas/query p90:",
        f"{row['p90_generated_formulas']:.1f}",
    )
    print(
        "Generated formulas/query max:",
        int(
            row[
                "max_generated_formulas"
            ]
        ),
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


if __name__ == "__main__":
    main()
