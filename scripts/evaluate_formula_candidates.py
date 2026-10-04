from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from casmi26.chemistry.adducts import neutral_mass
from casmi26.chemistry.formula_index import (
    FormulaIndex,
    search_formula_index,
)


TRAIN_PATH = Path("data/raw/train.parquet")
FORMULA_INDEX_PATH = Path(
    "data/processed/formula_index/formulas.parquet"
)

# This is the structure-disjoint validation split created earlier.
SPLIT_DIR = Path("data/processed/splits")

TOLERANCES_PPM = [5.0, 10.0, 20.0]


def find_validation_path() -> Path:
    candidates = [
        SPLIT_DIR / "validation.parquet",
        SPLIT_DIR / "structure_validation.parquet",
        SPLIT_DIR / "validation_structures.parquet",
    ]

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        "Could not locate the structure-disjoint validation "
        "manifest. Expected one of:\n"
        + "\n".join(
            f"  {path}"
            for path in candidates
        )
    )


def load_formula_index() -> FormulaIndex:
    frame = pq.read_table(
        FORMULA_INDEX_PATH
    ).to_pandas()

    # build_formula_index() saved this sorted by mass.
    masses = frame["monoisotopic_mass"].to_numpy(
        dtype=np.float64
    )

    if np.any(masses[1:] < masses[:-1]):
        raise ValueError(
            "Formula index is not sorted by mass"
        )

    return FormulaIndex(
        formulas=frame["formula"].to_numpy(
            dtype=object
        ),
        masses=masses,
    )


def summarize_counts(
    values: list[int],
) -> str:
    if not values:
        return "n=0"

    arr = np.asarray(values)

    return (
        f"median={np.median(arr):.0f} "
        f"p90={np.percentile(arr, 90):.0f} "
        f"p95={np.percentile(arr, 95):.0f} "
        f"p99={np.percentile(arr, 99):.0f} "
        f"max={arr.max():,}"
    )


def main() -> None:
    validation_path = find_validation_path()

    print(f"Validation manifest: {validation_path}")

    manifest = pq.read_table(
        validation_path
    ).to_pandas()

    print(
        f"Validation manifest rows: "
        f"{len(manifest):,}"
    )

    #
    # Determine which column contains the structure key.
    #

    if "inchikey14" in manifest.columns:
        validation_keys = set(
            manifest["inchikey14"]
            .dropna()
            .astype(str)
        )
    elif "structure_id" in manifest.columns:
        validation_keys = set(
            manifest["structure_id"]
            .dropna()
            .astype(str)
        )
    else:
        raise ValueError(
            "Validation manifest has neither "
            "'inchikey14' nor 'structure_id'. "
            f"Columns: {manifest.columns.tolist()}"
        )

    print(
        f"Validation structures: "
        f"{len(validation_keys):,}"
    )

    index = load_formula_index()

    print(
        f"Formula index: "
        f"{len(index.formulas):,} formulas"
    )

    #
    # Collect spectra belonging to validation structures.
    #

    parquet = pq.ParquetFile(TRAIN_PATH)

    query_rows: list[pd.DataFrame] = []

    columns = [
        "inchikey14",
        "molecular_formula",
        "ionization_mode",
        "adduct",
        "precursor_mz",
        "ingest_lib",
    ]

    total_rows = 0

    print("Collecting validation spectra...")

    for batch in parquet.iter_batches(
        columns=columns,
        batch_size=100_000,
    ):
        frame = batch.to_pandas()

        total_rows += len(frame)

        mask = frame["inchikey14"].isin(
            validation_keys
        )

        if mask.any():
            query_rows.append(
                frame.loc[mask].copy()
            )

        print(
            f"\rRows scanned: {total_rows:,}",
            end="",
            flush=True,
        )

    print()

    queries = pd.concat(
        query_rows,
        ignore_index=True,
    )

    print(
        f"Validation spectra: {len(queries):,}"
    )

    #
    # Evaluate each spectrum independently first.
    #
    # Later we can aggregate multiple spectra/adducts belonging
    # to the same structure.
    #

    usable = 0
    invalid_formula = 0
    unsupported_adduct = 0

    recalls: dict[float, int] = {
        tolerance: 0
        for tolerance in TOLERANCES_PPM
    }

    candidate_counts: dict[
        float,
        list[int],
    ] = {
        tolerance: []
        for tolerance in TOLERANCES_PPM
    }

    by_adduct = defaultdict(
        lambda: {
            tolerance: {
                "n": 0,
                "hit": 0,
                "counts": [],
            }
            for tolerance in TOLERANCES_PPM
        }
    )

    examples_missed: dict[
        float,
        list[dict],
    ] = {
        tolerance: []
        for tolerance in TOLERANCES_PPM
    }

    formula_lookup = set(
        index.formulas.tolist()
    )

    for row in queries.itertuples(
        index=False
    ):
        formula = str(row.molecular_formula)

        # Invalid/charged formulas were deliberately not put
        # into the formula index.
        if formula not in formula_lookup:
            invalid_formula += 1
            continue

        mass = neutral_mass(
            float(row.precursor_mz),
            str(row.adduct),
        )

        if mass is None:
            unsupported_adduct += 1
            continue

        usable += 1

        for tolerance in TOLERANCES_PPM:
            candidate_indices = (
                search_formula_index(
                    index,
                    neutral_mass=mass,
                    tolerance_ppm=tolerance,
                )
            )

            candidates = index.formulas[
                candidate_indices
            ]

            count = len(candidates)
            hit = bool(
                np.any(candidates == formula)
            )

            candidate_counts[
                tolerance
            ].append(count)

            recalls[tolerance] += int(hit)

            stats = by_adduct[
                str(row.adduct)
            ][tolerance]

            stats["n"] += 1
            stats["hit"] += int(hit)
            stats["counts"].append(count)

            if (
                not hit
                and len(
                    examples_missed[tolerance]
                ) < 20
            ):
                examples_missed[
                    tolerance
                ].append(
                    {
                        "inchikey14": (
                            row.inchikey14
                        ),
                        "formula": formula,
                        "adduct": row.adduct,
                        "precursor_mz": (
                            row.precursor_mz
                        ),
                        "neutral_mass": mass,
                        "candidate_count": count,
                        "library": (
                            row.ingest_lib
                        ),
                    }
                )

    print()
    print("Formula candidate evaluation")
    print("=" * 60)
    print(
        f"Validation spectra:     "
        f"{len(queries):,}"
    )
    print(
        f"Usable spectra:         "
        f"{usable:,}"
    )
    print(
        f"Formula not indexed:    "
        f"{invalid_formula:,}"
    )
    print(
        f"Unsupported adduct:     "
        f"{unsupported_adduct:,}"
    )

    print()
    print("Overall:")

    for tolerance in TOLERANCES_PPM:
        hit = recalls[tolerance]

        recall = (
            hit / usable
            if usable
            else 0.0
        )

        print()
        print(
            f"{tolerance:g} ppm"
        )
        print(
            f"  Formula recall: "
            f"{hit:,}/{usable:,} "
            f"({recall:.3%})"
        )
        print(
            "  Candidate count: "
            + summarize_counts(
                candidate_counts[
                    tolerance
                ]
            )
        )

    #
    # Adduct-specific results at 10 ppm.
    #

    print()
    print("By adduct at 10 ppm:")
    print()

    rows = []

    for adduct, tolerance_stats in (
        by_adduct.items()
    ):
        stats = tolerance_stats[10.0]

        if not stats["n"]:
            continue

        counts = np.asarray(
            stats["counts"]
        )

        rows.append(
            (
                adduct,
                stats["n"],
                stats["hit"] / stats["n"],
                np.median(counts),
                np.percentile(counts, 95),
            )
        )

    rows.sort(
        key=lambda item: -item[1]
    )

    for (
        adduct,
        n,
        recall,
        median_count,
        p95_count,
    ) in rows:
        print(
            f"{adduct:20s} "
            f"n={n:>8,} "
            f"recall={recall:8.3%} "
            f"median={median_count:5.0f} "
            f"p95={p95_count:6.0f}"
        )

    #
    # Miss examples.
    #

    for tolerance in TOLERANCES_PPM:
        misses = examples_missed[
            tolerance
        ]

        print()
        print(
            f"Example misses at "
            f"{tolerance:g} ppm:"
        )

        if not misses:
            print("  None")
            continue

        frame = pd.DataFrame(misses)

        print(
            frame.to_string(
                index=False
            )
        )


if __name__ == "__main__":
    main()