from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from casmi26.chemistry.formula import monoisotopic_mass
from casmi26.chemistry.formula_index import build_formula_index


TRAIN_PATH = Path("data/raw/train.parquet")

OUTPUT_DIR = Path(
    "data/processed/formula_index"
)

OUTPUT_PATH = OUTPUT_DIR / "formulas.parquet"


def main() -> None:
    parquet = pq.ParquetFile(TRAIN_PATH)

    formulas: set[str] = set()

    total_rows = 0
    invalid_rows = 0
    invalid_formulas: Counter[str] = Counter()

    print("Collecting molecular formulas...")

    for batch_number, batch in enumerate(
        parquet.iter_batches(
            columns=["molecular_formula"],
            batch_size=100_000,
        ),
        start=1,
    ):
        values = batch.column(0).to_pylist()

        total_rows += len(values)

        for value in values:
            if not value:
                continue

            formula = str(value)

            try:
                monoisotopic_mass(formula)
            except ValueError:
                invalid_rows += 1
                invalid_formulas[formula] += 1
                continue

            formulas.add(formula)

        print(
            f"\rRows {total_rows:,} | "
            f"unique valid formulas {len(formulas):,}",
            end="",
            flush=True,
        )

    print()
    print()

    print("Calculating theoretical masses...")

    ordered_formulas = sorted(formulas)

    masses = [
        monoisotopic_mass(formula)
        for formula in ordered_formulas
    ]

    index = build_formula_index(
        ordered_formulas,
        masses,
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    table = pa.table(
        {
            "formula": pa.array(
                index.formulas.tolist(),
                type=pa.string(),
            ),
            "monoisotopic_mass": pa.array(
                index.masses,
                type=pa.float64(),
            ),
        }
    )

    pq.write_table(
        table,
        OUTPUT_PATH,
        compression="zstd",
    )

    print("Formula index")
    print("=" * 50)
    print(f"Training rows:          {total_rows:,}")
    print(f"Valid unique formulas:  {len(index.formulas):,}")
    print(f"Invalid formula rows:   {invalid_rows:,}")

    if len(index.masses):
        print(
            f"Minimum mass:           "
            f"{index.masses.min():.6f}"
        )
        print(
            f"Median mass:            "
            f"{np.median(index.masses):.6f}"
        )
        print(
            f"Maximum mass:           "
            f"{index.masses.max():.6f}"
        )

    if invalid_formulas:
        print()
        print("Most common invalid formulas:")

        for formula, count in (
            invalid_formulas.most_common(20)
        ):
            print(
                f"  {formula:30s} "
                f"{count:>8,}"
            )

    print()
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()