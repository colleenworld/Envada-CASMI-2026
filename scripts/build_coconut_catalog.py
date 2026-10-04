from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from casmi26.chemistry.coconut_policy import (
    CoconutDisposition,
    approved_collections,
    classify_collections,
    parse_collections,
)


INPUT_PATH = Path(
    "data/external/coconut/coconut_csv-10-2026.csv"
)

OUTPUT_PATH = Path(
    "data/external/coconut/coconut_approved.parquet"
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
    if not isinstance(value, str):
        return False

    value = value.strip()

    # Standard InChIKey:
    # XXXXXXXXXXXXXX-XXXXXXXXXX-X
    return (
        len(value) == 27
        and value[14] == "-"
        and value[25] == "-"
    )


def normalize_text(value: object) -> str | None:
    if pd.isna(value):
        return None

    value = str(value).strip()

    return value or None


def main() -> None:
    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    disposition_counts: Counter[str] = Counter()

    rows_read = 0
    rows_approved = 0
    invalid_inchikey = 0
    invalid_smiles = 0
    invalid_formula = 0
    invalid_mass = 0

    frames: list[pd.DataFrame] = []

    for chunk in pd.read_csv(
        INPUT_PATH,
        usecols=INPUT_COLUMNS,
        chunksize=CHUNK_SIZE,
    ):
        rows_read += len(chunk)

        output_rows: list[dict[str, object]] = []

        for row in chunk.itertuples(index=False):
            collections = parse_collections(
                row.collections
            )

            disposition = classify_collections(
                collections
            )

            disposition_counts[
                disposition.value
            ] += 1

            if disposition != CoconutDisposition.APPROVED:
                continue

            inchikey = normalize_text(
                row.standard_inchi_key
            )

            if not valid_inchikey(inchikey):
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
            except (TypeError, ValueError):
                invalid_mass += 1
                continue

            if not np.isfinite(mass) or mass <= 0:
                invalid_mass += 1
                continue

            approved_sources = sorted(
                approved_collections(collections)
            )

            output_rows.append(
                {
                    "coconut_id": row.identifier,
                    "inchikey": inchikey,
                    "inchikey14": inchikey[:14],
                    "smiles": smiles,
                    "molecular_formula": formula,
                    "mass": mass,
                    "collections": "|".join(
                        sorted(collections)
                    ),
                    "approved_collections": "|".join(
                        approved_sources
                    ),
                    "np_likeness": row.np_likeness,
                    "chemical_class": row.chemical_class,
                    "chemical_sub_class": (
                        row.chemical_sub_class
                    ),
                    "chemical_super_class": (
                        row.chemical_super_class
                    ),
                    "np_classifier_pathway": (
                        row.np_classifier_pathway
                    ),
                    "np_classifier_superclass": (
                        row.np_classifier_superclass
                    ),
                    "np_classifier_class": (
                        row.np_classifier_class
                    ),
                }
            )

        if output_rows:
            frame = pd.DataFrame(output_rows)
            frames.append(frame)
            rows_approved += len(frame)

        print(
            f"\rRows read: {rows_read:,}  "
            f"approved/valid: {rows_approved:,}",
            end="",
            flush=True,
        )

    print()

    if not frames:
        raise RuntimeError(
            "No approved COCONUT records were produced"
        )

    catalog = pd.concat(
        frames,
        ignore_index=True,
    )

    before_dedup = len(catalog)

    # COCONUT should largely already be structure-normalized,
    # but our competition identity is InChIKey14. Keep one
    # representative row per identity.
    catalog = (
        catalog.sort_values(
            [
                "inchikey14",
                "coconut_id",
            ]
        )
        .drop_duplicates(
            subset=["inchikey14"],
            keep="first",
        )
        .reset_index(drop=True)
    )

    duplicates_removed = (
        before_dedup - len(catalog)
    )

    catalog.to_parquet(
        OUTPUT_PATH,
        index=False,
    )

    print()
    print("COCONUT approved catalog")
    print("=" * 70)

    print(
        f"Input rows:              "
        f"{rows_read:,}"
    )

    for disposition in CoconutDisposition:
        count = disposition_counts[
            disposition.value
        ]

        print(
            f"{disposition.value:24s}"
            f"{count:>10,}  "
            f"{count / rows_read:>8.3%}"
        )

    print()
    print(
        f"Approved before validation: "
        f"{disposition_counts['approved']:,}"
    )

    print(
        f"Invalid InChIKey:           "
        f"{invalid_inchikey:,}"
    )

    print(
        f"Missing SMILES:              "
        f"{invalid_smiles:,}"
    )

    print(
        f"Missing formula:             "
        f"{invalid_formula:,}"
    )

    print(
        f"Invalid mass:                "
        f"{invalid_mass:,}"
    )

    print(
        f"Valid approved rows:         "
        f"{before_dedup:,}"
    )

    print(
        f"InChIKey14 duplicates:       "
        f"{duplicates_removed:,}"
    )

    print(
        f"Final structures:            "
        f"{len(catalog):,}"
    )

    print()
    print(
        f"Saved: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()