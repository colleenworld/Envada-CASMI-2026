from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from casmi26.chemistry.adducts import neutral_mass
from casmi26.chemistry.formula import (
    mass_error_ppm,
    monoisotopic_mass,
)


TRAIN_PATH = Path("data/raw/train.parquet")

COLUMNS = [
    "ingest_lib",
    "normalized_smiles",
    "inchikey14",
    "molecular_formula",
    "ionization_mode",
    "adduct",
    "adduct_orig",
    "precursor_mz",
    "precursor_error_ppm",
]

SUPPORTED_ADDUCTS_TO_INSPECT = {
    "[M+H]+",
    "[M+Na]+",
    "[M-H]-",
    "[M-H2O+H]+",
    "[M+K]+",
}

MULTIMER_ADDUCTS = {
    "[2M+H]+",
    "[2M+Na]+",
    "[2M-H]-",
}

EXAMPLE_COUNT = 10


def calculate_error(
    formula: str | None,
    precursor_mz: float | None,
    adduct: str | None,
) -> tuple[float | None, float | None]:
    if (
        not formula
        or precursor_mz is None
        or not np.isfinite(precursor_mz)
        or not adduct
    ):
        return None, None

    try:
        theoretical = monoisotopic_mass(formula)
    except ValueError:
        return None, None

    inferred = neutral_mass(
        float(precursor_mz),
        str(adduct),
    )

    if inferred is None:
        return None, None

    ppm = mass_error_ppm(
        inferred,
        theoretical,
    )

    return inferred, ppm


def print_rows(
    title: str,
    frame: pd.DataFrame,
    n: int = EXAMPLE_COUNT,
) -> None:
    print()
    print("=" * 100)
    print(title)
    print("=" * 100)

    if frame.empty:
        print("No rows.")
        return

    columns = [
        "ingest_lib",
        "inchikey14",
        "molecular_formula",
        "ionization_mode",
        "adduct",
        "adduct_orig",
        "precursor_mz",
        "precursor_error_ppm",
    ]

    extra = [
        column
        for column in [
            "calculated_ppm",
            "ppm_difference",
        ]
        if column in frame.columns
    ]

    print(
        frame[
            columns + extra
        ]
        .head(n)
        .to_string(index=False)
    )


def main() -> None:
    parquet = pq.ParquetFile(TRAIN_PATH)

    charged_formula_rows = []
    multimer_rows = []
    supported_rows = []

    for batch in parquet.iter_batches(
        columns=COLUMNS,
        batch_size=100_000,
    ):
        frame = batch.to_pandas()

        # Charged molecular formulas.
        formula_text = (
            frame["molecular_formula"]
            .fillna("")
            .astype(str)
        )

        charged_mask = formula_text.str.endswith(
            ("+", "-")
        )

        if charged_mask.any():
            charged_formula_rows.append(
                frame.loc[charged_mask].copy()
            )

        # Multimer adducts that our current neutral_mass()
        # implementation cannot yet handle.
        multimer_mask = frame["adduct"].isin(
            MULTIMER_ADDUCTS
        )

        if multimer_mask.any():
            multimer_rows.append(
                frame.loc[multimer_mask].copy()
            )

        # Supported adducts: calculate our own error so that
        # we can inspect the tails.
        supported_mask = frame["adduct"].isin(
            SUPPORTED_ADDUCTS_TO_INSPECT
        )

        supported = frame.loc[
            supported_mask
        ].copy()

        if not supported.empty:
            calculated_ppm = []

            for row in supported.itertuples(
                index=False
            ):
                _, ppm = calculate_error(
                    row.molecular_formula,
                    row.precursor_mz,
                    row.adduct,
                )

                calculated_ppm.append(ppm)

            supported["calculated_ppm"] = (
                calculated_ppm
            )

            supplied = pd.to_numeric(
                supported["precursor_error_ppm"],
                errors="coerce",
            )

            calculated = pd.to_numeric(
                supported["calculated_ppm"],
                errors="coerce",
            )

            supported["ppm_difference"] = (
                calculated.abs()
                - supplied.abs()
            ).abs()

            supported_rows.append(supported)

    charged = (
        pd.concat(
            charged_formula_rows,
            ignore_index=True,
        )
        if charged_formula_rows
        else pd.DataFrame()
    )

    multimers = (
        pd.concat(
            multimer_rows,
            ignore_index=True,
        )
        if multimer_rows
        else pd.DataFrame()
    )

    supported = (
        pd.concat(
            supported_rows,
            ignore_index=True,
        )
        if supported_rows
        else pd.DataFrame()
    )

    #
    # 1. Charged formulas
    #

    print()
    print("Charged molecular formulas")
    print("--------------------------")
    print(f"Rows: {len(charged):,}")

    if not charged.empty:
        print()
        print("By library:")
        print(
            charged["ingest_lib"]
            .value_counts()
            .to_string()
        )

        print()
        print("By adduct:")
        print(
            charged["adduct"]
            .value_counts()
            .head(20)
            .to_string()
        )

        print_rows(
            "Examples of charged molecular formulas",
            charged,
        )

    #
    # 2. Unsupported multimer adducts
    #

    print()
    print("Unsupported multimer adducts")
    print("----------------------------")
    print(f"Rows: {len(multimers):,}")

    if not multimers.empty:
        print()
        print("By adduct and library:")

        counts = (
            multimers.groupby(
                ["adduct", "ingest_lib"]
            )
            .size()
            .sort_values(
                ascending=False
            )
        )

        print(counts.head(30).to_string())

        print_rows(
            "Examples of unsupported multimers",
            multimers,
        )

    #
    # 3. Supported-adduct error tails
    #

    valid = supported[
        supported["calculated_ppm"].notna()
    ].copy()

    valid["abs_calculated_ppm"] = (
        valid["calculated_ppm"].abs()
    )

    for adduct in sorted(
        SUPPORTED_ADDUCTS_TO_INSPECT
    ):
        subset = valid[
            valid["adduct"] == adduct
        ].copy()

        if subset.empty:
            continue

        print()
        print("=" * 100)
        print(f"Error analysis: {adduct}")
        print("=" * 100)
        print(f"Rows: {len(subset):,}")

        for threshold in [
            10,
            20,
            50,
            100,
            1_000,
        ]:
            count = int(
                (
                    subset["abs_calculated_ppm"]
                    > threshold
                ).sum()
            )

            print(
                f"> {threshold:>5,} ppm: "
                f"{count:>8,} "
                f"({count / len(subset):7.3%})"
            )

        print()
        print("Worst libraries by >100 ppm count:")

        extreme = subset[
            subset["abs_calculated_ppm"] > 100
        ]

        if extreme.empty:
            print("None")
        else:
            summary = (
                extreme.groupby("ingest_lib")
                .size()
                .sort_values(
                    ascending=False
                )
            )

            print(
                summary.head(20).to_string()
            )

        worst = subset.sort_values(
            "abs_calculated_ppm",
            ascending=False,
        )

        print_rows(
            f"Worst {adduct} examples",
            worst,
        )

    #
    # 4. Disagreement with supplied precursor_error_ppm
    #

    comparable = valid[
        valid["precursor_error_ppm"].notna()
        & valid["ppm_difference"].notna()
    ].copy()

    print()
    print("=" * 100)
    print("Calculated vs supplied precursor error")
    print("=" * 100)
    print(f"Comparable rows: {len(comparable):,}")

    for threshold in [
        1,
        2,
        5,
        10,
        100,
        1_000,
    ]:
        count = int(
            (
                comparable["ppm_difference"]
                > threshold
            ).sum()
        )

        print(
            f"Difference > {threshold:>5,} ppm: "
            f"{count:>9,} "
            f"({count / len(comparable):7.3%})"
        )

    print()
    print("Largest disagreement by library:")

    disagreement = (
        comparable.groupby("ingest_lib")[
            "ppm_difference"
        ]
        .agg(
            [
                "count",
                "median",
                lambda x: x.quantile(0.95),
                "max",
            ]
        )
    )

    disagreement.columns = [
        "count",
        "median",
        "p95",
        "max",
    ]

    print(
        disagreement.sort_values(
            "p95",
            ascending=False,
        ).to_string()
    )

    worst_disagreement = (
        comparable.sort_values(
            "ppm_difference",
            ascending=False,
        )
    )

    print_rows(
        "Worst calculated/supplied disagreements",
        worst_disagreement,
    )


if __name__ == "__main__":
    main()