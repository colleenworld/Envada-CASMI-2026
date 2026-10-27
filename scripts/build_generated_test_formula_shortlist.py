from __future__ import annotations

from collections import Counter, defaultdict
from math import log
from pathlib import Path

import numpy as np
import pandas as pd

from casmi26.chemistry.formula import parse_formula
from casmi26.chemistry.formula_enumeration import (
    build_signature_rules,
    enumerate_for_mass,
    neutral_mass_from_adduct,
    select_query_spectra,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent

TEST_PATH = (
    PROJECT_ROOT
    / "data/raw/test.parquet"
)

TRAINING_STRUCTURES_PATH = (
    PROJECT_ROOT
    / "data/processed/structure_index/training_structures.parquet"
)

CURRENT_FORMULA_INDEX_PATH = (
    PROJECT_ROOT
    / "data/processed/formula_index/training_coconut_formulas.parquet"
)

PUBCHEM_FORMULAS_PATH = (
    PROJECT_ROOT
    / "data/processed/pubchem_formula_index/formula_buckets.parquet"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data/processed/formula_index"
)

OUTPUT_PATH = (
    OUTPUT_DIR
    / "generated_test_top25_formulas.parquet"
)

MISSING_OUTPUT_PATH = (
    OUTPUT_DIR
    / "generated_test_top25_missing_pubchem_formulas.parquet"
)

SUMMARY_PATH = (
    OUTPUT_DIR
    / "generated_test_top25_formula_summary.csv"
)

FIXED_PRIOR_WEIGHT = 0.50
TOP_K = 25

ELEMENTS = [
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
]


def signature_for(
    composition: dict[str, int],
) -> tuple[str, ...]:
    return tuple(
        element
        for element in ELEMENTS
        if composition.get(
            element,
            0,
        )
        > 0
    )


def build_training_prior():
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

    signature_counts: Counter[
        tuple[str, ...]
    ] = Counter()

    count_histograms = defaultdict(
        lambda:
            defaultdict(
                Counter
            )
    )

    signature_totals: Counter[
        tuple[str, ...]
    ] = Counter()

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

        signature = signature_for(
            composition
        )

        signature_counts[
            signature
        ] += 1

        signature_totals[
            signature
        ] += 1

        for element in signature:
            count_histograms[
                signature
            ][
                element
            ][
                int(
                    composition[
                        element
                    ]
                )
            ] += 1

    return (
        signature_counts,
        count_histograms,
        signature_totals,
        sum(
            signature_counts.values()
        ),
    )


def log_formula_prior(
    formula: str,
    signature_counts,
    count_histograms,
    signature_totals,
    total_formulas: int,
) -> float:
    try:
        composition = parse_formula(
            formula
        )
    except ValueError:
        return -1e9

    signature = signature_for(
        composition
    )

    signature_log_prior = log(
        (
            signature_counts.get(
                signature,
                0,
            )
            + 1.0
        )
        / (
            total_formulas
            + len(
                signature_counts
            )
        )
    )

    total_for_signature = (
        signature_totals.get(
            signature,
            0,
        )
    )

    if total_for_signature <= 0:
        return signature_log_prior

    count_log_prior = 0.0

    for element in signature:
        histogram = (
            count_histograms[
                signature
            ][
                element
            ]
        )

        observed_values = max(
            1,
            len(
                histogram
            ),
        )

        count = int(
            composition[
                element
            ]
        )

        count_log_prior += log(
            (
                histogram.get(
                    count,
                    0,
                )
                + 1.0
            )
            / (
                total_for_signature
                + observed_values
            )
        )

    if signature:
        count_log_prior /= len(
            signature
        )

    return (
        signature_log_prior
        + count_log_prior
    )


def main() -> None:
    for path in [
        TEST_PATH,
        TRAINING_STRUCTURES_PATH,
        CURRENT_FORMULA_INDEX_PATH,
        PUBCHEM_FORMULAS_PATH,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"Missing required artifact: {path}"
            )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    test = pd.read_parquet(
        TEST_PATH
    ).copy()

    required = {
        "molecule_id",
        "precursor_mz",
        "adduct",
    }

    missing = (
        required
        - set(
            test.columns
        )
    )

    if missing:
        raise ValueError(
            "Test parquet missing columns: "
            f"{sorted(missing)}"
        )

    test[
        "molecule_id"
    ] = (
        test[
            "molecule_id"
        ]
        .astype(str)
    )

    existing_formula_index = set(
        pd.read_parquet(
            CURRENT_FORMULA_INDEX_PATH,
            columns=[
                "molecular_formula",
            ],
        )[
            "molecular_formula"
        ]
        .dropna()
        .astype(str)
    )

    pubchem_formulas = set(
        pd.read_parquet(
            PUBCHEM_FORMULAS_PATH,
            columns=[
                "molecular_formula",
            ],
        )[
            "molecular_formula"
        ]
        .dropna()
        .astype(str)
    )

    rules = build_signature_rules(
        TRAINING_STRUCTURES_PATH
    )

    (
        signature_counts,
        count_histograms,
        signature_totals,
        total_formulas,
    ) = build_training_prior()

    rows = []

    molecule_count = (
        test[
            "molecule_id"
        ]
        .nunique()
    )

    for number, (
        molecule_id,
        group,
    ) in enumerate(
        test.groupby(
            "molecule_id",
            sort=False,
        ),
        start=1,
    ):
        selected = (
            select_query_spectra(
                group
            )
        )

        merged: dict[
            str,
            tuple[
                float,
                float,
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

            for (
                formula,
                mass,
                ppm,
                _,
            ) in enumerate_for_mass(
                neutral_mass,
                rules,
            ):
                abs_ppm = abs(
                    float(
                        ppm
                    )
                )

                existing = (
                    merged.get(
                        formula
                    )
                )

                if (
                    existing is None
                    or abs_ppm
                    < existing[
                        1
                    ]
                ):
                    merged[
                        formula
                    ] = (
                        float(
                            mass
                        ),
                        abs_ppm,
                    )

        ranked = []

        for (
            formula,
            (
                mass,
                abs_ppm,
            ),
        ) in merged.items():
            prior = log_formula_prior(
                formula,
                signature_counts,
                count_histograms,
                signature_totals,
                total_formulas,
            )

            score = (
                -abs_ppm
                + (
                    FIXED_PRIOR_WEIGHT
                    * prior
                )
            )

            ranked.append(
                (
                    formula,
                    mass,
                    abs_ppm,
                    prior,
                    score,
                )
            )

        ranked.sort(
            key=lambda item: (
                -item[
                    4
                ],
                item[
                    0
                ],
            )
        )

        for rank, (
            formula,
            mass,
            abs_ppm,
            prior,
            score,
        ) in enumerate(
            ranked[
                :TOP_K
            ],
            start=1,
        ):
            rows.append(
                {
                    "molecule_id":
                        str(
                            molecule_id
                        ),
                    "formula_rank":
                        rank,
                    "molecular_formula":
                        formula,
                    "monoisotopic_mass":
                        mass,
                    "abs_mass_error_ppm":
                        abs_ppm,
                    "log_formula_prior":
                        prior,
                    "formula_score":
                        score,
                    "already_in_formula_index":
                        formula
                        in existing_formula_index,
                    "already_in_pubchem_asset":
                        formula
                        in pubchem_formulas,
                }
            )

        if (
            number % 50 == 0
            or number == molecule_count
        ):
            print(
                f"  processed "
                f"{number:,}/{molecule_count:,}"
            )

    shortlist = pd.DataFrame(
        rows
    )

    shortlist.to_parquet(
        OUTPUT_PATH,
        index=False,
        compression="zstd",
    )

    unique = (
        shortlist[
            [
                "molecular_formula",
                "monoisotopic_mass",
                "already_in_formula_index",
                "already_in_pubchem_asset",
            ]
        ]
        .drop_duplicates(
            subset=[
                "molecular_formula",
            ]
        )
        .sort_values(
            [
                "monoisotopic_mass",
                "molecular_formula",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    missing_pubchem = (
        unique.loc[
            ~unique[
                "already_in_pubchem_asset"
            ]
        ]
        .copy()
    )

    missing_pubchem.to_parquet(
        MISSING_OUTPUT_PATH,
        index=False,
        compression="zstd",
    )

    summary = pd.DataFrame(
        [
            {
                "test_molecules":
                    molecule_count,
                "shortlist_rows":
                    len(
                        shortlist
                    ),
                "unique_top25_formulas":
                    len(
                        unique
                    ),
                "already_in_formula_index":
                    int(
                        unique[
                            "already_in_formula_index"
                        ].sum()
                    ),
                "missing_from_formula_index":
                    int(
                        (
                            ~unique[
                                "already_in_formula_index"
                            ]
                        ).sum()
                    ),
                "already_in_pubchem_asset":
                    int(
                        unique[
                            "already_in_pubchem_asset"
                        ].sum()
                    ),
                "missing_from_pubchem_asset":
                    int(
                        (
                            ~unique[
                                "already_in_pubchem_asset"
                            ]
                        ).sum()
                    ),
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
        "Production test formula shortlist"
    )
    print(
        "=" * 92
    )
    print(
        "Test molecules:",
        f"{int(row['test_molecules']):,}",
    )
    print(
        "Top-25 shortlist rows:",
        f"{int(row['shortlist_rows']):,}",
    )
    print(
        "Unique Top-25 formulas:",
        f"{int(row['unique_top25_formulas']):,}",
    )
    print(
        "Already in current formula index:",
        f"{int(row['already_in_formula_index']):,}",
    )
    print(
        "Missing from current formula index:",
        f"{int(row['missing_from_formula_index']):,}",
    )
    print(
        "Already in current PubChem asset:",
        f"{int(row['already_in_pubchem_asset']):,}",
    )
    print(
        "Missing from current PubChem asset:",
        f"{int(row['missing_from_pubchem_asset']):,}",
    )
    print()
    print(
        "Wrote:",
        OUTPUT_PATH,
    )
    print(
        "Wrote:",
        MISSING_OUTPUT_PATH,
    )
    print(
        "Wrote:",
        SUMMARY_PATH,
    )


if __name__ == "__main__":
    main()
