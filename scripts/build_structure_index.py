from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from casmi26.chemistry.formula import monoisotopic_mass
from casmi26.chemistry.structure_index import (
    build_structure_index,
)


TRAIN_PATH = Path("data/raw/train.parquet")

OUTPUT_DIR = Path(
    "data/processed/structure_index"
)

OUTPUT_PATH = OUTPUT_DIR / "structures.parquet"


def main() -> None:
    parquet = pq.ParquetFile(TRAIN_PATH)

    #
    # structure -> Counter[(smiles, formula)]
    #
    # In principle each InChIKey14 should have consistent
    # metadata. In practice, multiple source libraries may
    # disagree, so collect the observations first and report
    # how often this happens.
    #

    structures: dict[
        str,
        Counter[tuple[str, str]],
    ] = {}

    total_rows = 0
    missing_structure = 0
    missing_smiles = 0
    missing_formula = 0

    print("Collecting structures...")

    for batch in parquet.iter_batches(
        columns=[
            "inchikey14",
            "normalized_smiles",
            "molecular_formula",
        ],
        batch_size=100_000,
    ):
        frame = batch.to_pandas()

        total_rows += len(frame)

        for row in frame.itertuples(
            index=False
        ):
            if pd.isna(row.inchikey14):
                missing_structure += 1
                continue

            if pd.isna(row.normalized_smiles):
                missing_smiles += 1
                continue

            if pd.isna(row.molecular_formula):
                missing_formula += 1
                continue

            key = str(row.inchikey14)
            smiles = str(row.normalized_smiles)
            formula = str(row.molecular_formula)

            counter = structures.setdefault(
                key,
                Counter(),
            )

            counter[
                (smiles, formula)
            ] += 1

        print(
            f"\rRows {total_rows:,} | "
            f"structures {len(structures):,}",
            end="",
            flush=True,
        )

    print()

    #
    # Resolve metadata.
    #
    # If libraries disagree about metadata for the same
    # InChIKey14, use the most frequently observed pair.
    # Keep track of disagreements rather than silently
    # pretending they don't exist.
    #

    keys: list[str] = []
    smiles_values: list[str] = []
    formula_values: list[str] = []
    masses: list[float] = []

    conflicting_structures = 0
    invalid_formula_structures = 0

    conflict_examples: list[
        tuple[
            str,
            list[tuple[tuple[str, str], int]],
        ]
    ] = []

    for key, observations in structures.items():
        if len(observations) > 1:
            conflicting_structures += 1

            if len(conflict_examples) < 20:
                conflict_examples.append(
                    (
                        key,
                        observations.most_common(5),
                    )
                )

        (
            smiles,
            formula,
        ), _ = observations.most_common(1)[0]

        try:
            mass = monoisotopic_mass(
                formula
            )
        except ValueError:
            invalid_formula_structures += 1
            continue

        if not np.isfinite(mass):
            invalid_formula_structures += 1
            continue

        keys.append(key)
        smiles_values.append(smiles)
        formula_values.append(formula)
        masses.append(mass)

    #
    # Use the tested abstraction to construct and
    # deterministically order the final index.
    #

    index = build_structure_index(
        inchikey14=keys,
        smiles=smiles_values,
        formulas=formula_values,
        masses=masses,
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    table = pa.table(
        {
            "inchikey14": pa.array(
                index.inchikey14.tolist(),
                type=pa.string(),
            ),
            "normalized_smiles": pa.array(
                index.smiles.tolist(),
                type=pa.string(),
            ),
            "molecular_formula": pa.array(
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

    #
    # Summary.
    #

    print()
    print("Structure index")
    print("=" * 60)

    print(
        f"Training rows:                "
        f"{total_rows:,}"
    )

    print(
        f"Observed structures:          "
        f"{len(structures):,}"
    )

    print(
        f"Indexed structures:           "
        f"{len(index):,}"
    )

    print(
        f"Missing structure rows:       "
        f"{missing_structure:,}"
    )

    print(
        f"Missing SMILES rows:          "
        f"{missing_smiles:,}"
    )

    print(
        f"Missing formula rows:         "
        f"{missing_formula:,}"
    )

    print(
        f"Conflicting structures:       "
        f"{conflicting_structures:,}"
    )

    print(
        f"Invalid-formula structures:   "
        f"{invalid_formula_structures:,}"
    )

    if len(index):
        print()
        print(
            f"Minimum mass:                 "
            f"{index.masses.min():.6f}"
        )

        print(
            f"Median mass:                  "
            f"{np.median(index.masses):.6f}"
        )

        print(
            f"Maximum mass:                 "
            f"{index.masses.max():.6f}"
        )

    if conflict_examples:
        print()
        print("Example metadata conflicts:")

        for key, observations in (
            conflict_examples
        ):
            print()
            print(f"  {key}")

            for (
                smiles,
                formula,
            ), count in observations:
                print(
                    f"    {count:>6,}  "
                    f"{formula:20s} "
                    f"{smiles}"
                )

    print()
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()