from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parent.parent

TRAIN_PATH = PROJECT_ROOT / "data/raw/train.parquet"
TEST_PATH = PROJECT_ROOT / "data/raw/test.parquet"

FORMULA_INDEX_PATH = (
    PROJECT_ROOT
    / "data/processed/formula_index/training_coconut_formulas.parquet"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data/processed/formula_index/"
    "pubchem_relevant_formulas.parquet"
)

ADDUCT_SHIFTS = {
    "[M+H]+": 1.007276466621,
    "[M-H]-": -1.007276466621,
    "[M+Na]+": 22.989218,
    "[M+K]+": 38.963158,
    "[M+NH4]+": 18.033823,
    "[M+CH2O2-H]-": 44.998201,
    "[M+Cl]-": 34.969402,
}

#
# This is a coverage guard, not a fitted scoring parameter.
# It expands the observed train+visible-test neutral-mass envelope.
#
MASS_MARGIN_FRACTION = 0.05
ABSOLUTE_MARGIN_DA = 25.0


def neutral_masses(path: Path) -> np.ndarray:
    parquet = pq.ParquetFile(path)

    values = []

    for batch in parquet.iter_batches(
        columns=[
            "precursor_mz",
            "adduct",
        ],
        batch_size=100_000,
    ):
        frame = batch.to_pandas()

        for adduct, shift in ADDUCT_SHIFTS.items():
            subset = frame.loc[
                frame["adduct"].astype(str) == adduct,
                "precursor_mz",
            ]

            precursor = pd.to_numeric(
                subset,
                errors="coerce",
            ).to_numpy(
                dtype=np.float64
            )

            precursor = precursor[
                np.isfinite(precursor)
            ]

            if precursor.size:
                values.append(
                    precursor
                    - shift
                )

    if not values:
        return np.array(
            [],
            dtype=np.float64,
        )

    return np.concatenate(values)


def main() -> None:
    if not TRAIN_PATH.exists():
        raise FileNotFoundError(
            f"Missing train parquet: {TRAIN_PATH}"
        )

    if not TEST_PATH.exists():
        raise FileNotFoundError(
            f"Missing test parquet: {TEST_PATH}"
        )

    formula_index = pd.read_parquet(
        FORMULA_INDEX_PATH
    )

    required = {
        "molecular_formula",
        "monoisotopic_mass",
    }

    missing = required - set(formula_index.columns)

    if missing:
        raise ValueError(
            "Formula index is missing columns: "
            f"{sorted(missing)}"
        )

    print("Scanning supported neutral masses...")

    train_masses = neutral_masses(
        TRAIN_PATH
    )

    test_masses = neutral_masses(
        TEST_PATH
    )

    combined = np.concatenate(
        [
            train_masses,
            test_masses,
        ]
    )

    if not combined.size:
        raise RuntimeError(
            "No supported precursor/adduct neutral masses found"
        )

    observed_min = float(
        np.min(combined)
    )

    observed_max = float(
        np.max(combined)
    )

    span = (
        observed_max
        - observed_min
    )

    margin = max(
        ABSOLUTE_MARGIN_DA,
        span
        * MASS_MARGIN_FRACTION,
    )

    allowed_min = max(
        0.0,
        observed_min
        - margin,
    )

    allowed_max = (
        observed_max
        + margin
    )

    masses = pd.to_numeric(
        formula_index[
            "monoisotopic_mass"
        ],
        errors="coerce",
    )

    relevant = formula_index.loc[
        masses.between(
            allowed_min,
            allowed_max,
            inclusive="both",
        )
    ].copy()

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    relevant.to_parquet(
        OUTPUT_PATH,
        index=False,
    )

    print()
    print("PubChem inference formula envelope")
    print("=================================")
    print(
        "Training supported neutral masses:",
        f"{len(train_masses):,}",
    )
    print(
        "Visible-test supported neutral masses:",
        f"{len(test_masses):,}",
    )
    print(
        "Observed neutral-mass minimum:",
        f"{observed_min:.4f}",
    )
    print(
        "Observed neutral-mass maximum:",
        f"{observed_max:.4f}",
    )
    print(
        "Safety margin:",
        f"{margin:.4f} Da",
    )
    print(
        "Allowed formula mass minimum:",
        f"{allowed_min:.4f}",
    )
    print(
        "Allowed formula mass maximum:",
        f"{allowed_max:.4f}",
    )

    total = len(
        formula_index
    )

    kept = len(
        relevant
    )

    print()
    print(
        "All inference formulas:",
        f"{total:,}",
    )
    print(
        "Mass-relevant formulas:",
        f"{kept:,}",
    )
    print(
        "Retained:",
        f"{kept / total:.2%}"
        if total
        else "n/a",
    )
    print(
        "Eliminated:",
        f"{total - kept:,}",
    )

    print()
    print(
        f"Wrote {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()
