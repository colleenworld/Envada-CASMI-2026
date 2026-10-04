from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from casmi26.chemistry.formula import (
    monoisotopic_mass,
)


TRAINING_PATH = Path(
    "data/processed/structure_index/training_structures.parquet"
)

COCONUT_PATH = Path(
    "data/external/coconut/coconut_approved.parquet"
)

OUTPUT_PATH = Path(
    "data/processed/formula_index/"
    "training_coconut_formulas.parquet"
)


def collect_formulas(
    frame: pd.DataFrame,
    column: str = "molecular_formula",
) -> set[str]:
    return {
        str(value).strip()
        for value in frame[column].dropna()
        if str(value).strip()
    }


def main() -> None:
    training = pd.read_parquet(
        TRAINING_PATH,
        columns=["molecular_formula"],
    )

    coconut = pd.read_parquet(
        COCONUT_PATH,
        columns=["molecular_formula"],
    )

    training_formulas = collect_formulas(training)
    coconut_formulas = collect_formulas(coconut)

    combined_formulas = (
        training_formulas | coconut_formulas
    )

    print("Formula sources")
    print("=" * 70)

    print(
        f"Training formulas:        "
        f"{len(training_formulas):,}"
    )

    print(
        f"COCONUT formulas:         "
        f"{len(coconut_formulas):,}"
    )

    print(
        f"Shared formulas:          "
        f"{len(training_formulas & coconut_formulas):,}"
    )

    print(
        f"COCONUT-only formulas:    "
        f"{len(coconut_formulas - training_formulas):,}"
    )

    print(
        f"Combined formulas:        "
        f"{len(combined_formulas):,}"
    )

    rows: list[dict[str, object]] = []
    invalid = 0

    for formula in combined_formulas:
        try:
            mass = monoisotopic_mass(formula)
        except (ValueError, KeyError):
            invalid += 1
            continue

        if not np.isfinite(mass) or mass <= 0:
            invalid += 1
            continue

        in_training = formula in training_formulas
        in_coconut = formula in coconut_formulas

        if in_training and in_coconut:
            source = "training+coconut"
        elif in_training:
            source = "training"
        else:
            source = "coconut"

        rows.append(
            {
                "molecular_formula": formula,
                "monoisotopic_mass": mass,
                "source": source,
                "in_training": in_training,
                "in_coconut": in_coconut,
            }
        )

    result = pd.DataFrame(rows)

    result = (
        result.sort_values(
            [
                "monoisotopic_mass",
                "molecular_formula",
            ]
        )
        .reset_index(drop=True)
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_parquet(
        OUTPUT_PATH,
        index=False,
    )

    print()
    print("Built formula index")
    print("=" * 70)

    print(
        f"Valid formulas:           "
        f"{len(result):,}"
    )

    print(
        f"Invalid formulas:         "
        f"{invalid:,}"
    )

    print()

    print("By source:")
    print(
        result["source"]
        .value_counts()
        .to_string()
    )

    print()

    print(
        f"Min mass:                 "
        f"{result['monoisotopic_mass'].min():.6f}"
    )

    print(
        f"Median mass:              "
        f"{result['monoisotopic_mass'].median():.6f}"
    )

    print(
        f"Max mass:                 "
        f"{result['monoisotopic_mass'].max():.6f}"
    )

    print()

    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()