from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


INPUT_PATH = Path(
    "data/external/coconut/coconut_csv-10-2026.csv"
)

OUTPUT_PATH = Path(
    "data/external/coconut/coconut_full.parquet"
)

CHUNK_SIZE = 100_000

INPUT_COLUMNS = [
    "identifier",
    "canonical_smiles",
    "standard_inchi_key",
    "molecular_formula",
    "exact_molecular_weight",
    "collections",
    "np_likeness",
    "chemical_class",
    "chemical_sub_class",
    "chemical_super_class",
    "np_classifier_pathway",
    "np_classifier_superclass",
    "np_classifier_class",
]


def valid_inchikey(value: object) -> bool:
    if not isinstance(
        value,
        str,
    ):
        return False

    value = value.strip()

    return (
        len(value) == 27
        and value[14] == "-"
        and value[25] == "-"
    )


def normalize_text(
    value: object,
) -> str | None:
    if pd.isna(
        value
    ):
        return None

    value = str(
        value
    ).strip()

    return value or None


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Missing COCONUT CSV: {INPUT_PATH}"
        )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    frames: list[pd.DataFrame] = []

    rows_read = 0
    valid_rows = 0
    invalid_inchikey = 0
    invalid_smiles = 0
    invalid_formula = 0
    invalid_mass = 0

    for chunk in pd.read_csv(
        INPUT_PATH,
        usecols=INPUT_COLUMNS,
        chunksize=CHUNK_SIZE,
    ):
        rows_read += len(
            chunk
        )

        output_rows: list[
            dict[str, object]
        ] = []

        for row in chunk.itertuples(
            index=False
        ):
            inchikey = normalize_text(
                row.standard_inchi_key
            )

            if not valid_inchikey(
                inchikey
            ):
                invalid_inchikey += 1
                continue

            smiles = normalize_text(
                row.canonical_smiles
            )

            if smiles is None:
                invalid_smiles += 1
                continue

            formula = normalize_text(
                row.molecular_formula
            )

            if formula is None:
                invalid_formula += 1
                continue

            try:
                mass = float(
                    row.exact_molecular_weight
                )
            except (
                TypeError,
                ValueError,
            ):
                invalid_mass += 1
                continue

            if (
                not np.isfinite(
                    mass
                )
                or mass <= 0
            ):
                invalid_mass += 1
                continue

            output_rows.append(
                {
                    "coconut_id":
                        normalize_text(
                            row.identifier
                        ),
                    "inchikey":
                        inchikey,
                    "inchikey14":
                        inchikey[:14],
                    "smiles":
                        smiles,
                    "molecular_formula":
                        formula,
                    "mass":
                        mass,
                    "collections":
                        normalize_text(
                            row.collections
                        ),
                    "np_likeness":
                        row.np_likeness,
                    "chemical_class":
                        row.chemical_class,
                    "chemical_sub_class":
                        row.chemical_sub_class,
                    "chemical_super_class":
                        row.chemical_super_class,
                    "np_classifier_pathway":
                        row.np_classifier_pathway,
                    "np_classifier_superclass":
                        row.np_classifier_superclass,
                    "np_classifier_class":
                        row.np_classifier_class,
                }
            )

        if output_rows:
            frame = pd.DataFrame(
                output_rows
            )

            frames.append(
                frame
            )

            valid_rows += len(
                frame
            )

        print(
            f"\rRows read: {rows_read:,}  "
            f"valid: {valid_rows:,}",
            end="",
            flush=True,
        )

    print()

    if not frames:
        raise RuntimeError(
            "No valid COCONUT records were produced"
        )

    catalog = pd.concat(
        frames,
        ignore_index=True,
    )

    before_dedup = len(
        catalog
    )

    #
    # Competition identity is InChIKey14. Prefer the first stable
    # COCONUT identifier when duplicate stereochemical/full-key rows
    # collapse to the same connectivity identity.
    #
    catalog[
        "_sort_id"
    ] = (
        catalog[
            "coconut_id"
        ]
        .fillna("")
        .astype(str)
    )

    catalog = (
        catalog
        .sort_values(
            [
                "inchikey14",
                "_sort_id",
            ],
            kind="stable",
        )
        .drop_duplicates(
            subset=[
                "inchikey14"
            ],
            keep="first",
        )
        .drop(
            columns=[
                "_sort_id"
            ]
        )
        .reset_index(
            drop=True
        )
    )

    duplicates_removed = (
        before_dedup
        - len(
            catalog
        )
    )

    catalog.to_parquet(
        OUTPUT_PATH,
        index=False,
        compression="zstd",
    )

    print()
    print(
        "COCONUT full competition catalog"
    )
    print(
        "=" * 70
    )
    print(
        f"Input rows:              {rows_read:,}"
    )
    print(
        f"Invalid InChIKey:        {invalid_inchikey:,}"
    )
    print(
        f"Missing SMILES:          {invalid_smiles:,}"
    )
    print(
        f"Missing formula:         {invalid_formula:,}"
    )
    print(
        f"Invalid mass:            {invalid_mass:,}"
    )
    print(
        f"Valid rows pre-dedup:    {before_dedup:,}"
    )
    print(
        f"InChIKey14 duplicates:   {duplicates_removed:,}"
    )
    print(
        f"Final structures:        {len(catalog):,}"
    )
    print(
        f"Unique formulas:         "
        f"{catalog['molecular_formula'].nunique():,}"
    )
    print()
    print(
        f"Saved: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()
