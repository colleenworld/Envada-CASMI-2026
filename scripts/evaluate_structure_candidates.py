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

VALIDATION_PATH = Path(
    "data/processed/splits/"
    "validation_structures.parquet"
)

FORMULA_INDEX_PATH = Path(
    "data/processed/formula_index/"
    "formulas.parquet"
)

STRUCTURE_INDEX_PATH = Path(
    "data/processed/structure_index/"
    "structures.parquet"
)

TOLERANCE_PPM = 10.0


def load_formula_index() -> FormulaIndex:
    frame = pq.read_table(
        FORMULA_INDEX_PATH
    ).to_pandas()

    masses = frame[
        "monoisotopic_mass"
    ].to_numpy(
        dtype=np.float64
    )

    if np.any(masses[1:] < masses[:-1]):
        raise ValueError(
            "Formula index is not sorted by mass"
        )

    return FormulaIndex(
        formulas=frame[
            "formula"
        ].to_numpy(
            dtype=object
        ),
        masses=masses,
    )


def main() -> None:
    #
    # Validation population.
    #

    manifest = pq.read_table(
        VALIDATION_PATH
    ).to_pandas()

    validation_keys = set(
        manifest["inchikey14"]
        .dropna()
        .astype(str)
    )

    print(
        f"Validation structures: "
        f"{len(validation_keys):,}"
    )

    #
    # Formula index.
    #

    formula_index = load_formula_index()

    print(
        f"Formula index: "
        f"{len(formula_index.formulas):,} formulas"
    )

    #
    # Structure index.
    #

    structures = pq.read_table(
        STRUCTURE_INDEX_PATH
    ).to_pandas()

    print(
        f"Structure index: "
        f"{len(structures):,} structures"
    )

    #
    # Build formula -> structure IDs lookup.
    #

    formula_to_structures: dict[
        str,
        set[str],
    ] = defaultdict(set)

    for row in structures.itertuples(
        index=False
    ):
        formula_to_structures[
            str(row.molecular_formula)
        ].add(
            str(row.inchikey14)
        )

    formula_structure_counts = np.asarray(
        [
            len(values)
            for values in formula_to_structures.values()
        ],
        dtype=np.int64,
    )

    print()
    print("Structures per formula")
    print("=" * 60)

    print(
        f"Formulas represented: "
        f"{len(formula_to_structures):,}"
    )

    print(
        f"Median: "
        f"{np.median(formula_structure_counts):.0f}"
    )

    print(
        f"P90:    "
        f"{np.percentile(formula_structure_counts, 90):.0f}"
    )

    print(
        f"P95:    "
        f"{np.percentile(formula_structure_counts, 95):.0f}"
    )

    print(
        f"P99:    "
        f"{np.percentile(formula_structure_counts, 99):.0f}"
    )

    print(
        f"Max:    "
        f"{formula_structure_counts.max():,}"
    )

    #
    # Collect validation spectra.
    #

    parquet = pq.ParquetFile(
        TRAIN_PATH
    )

    query_frames: list[pd.DataFrame] = []

    columns = [
        "inchikey14",
        "molecular_formula",
        "adduct",
        "precursor_mz",
    ]

    print()
    print("Collecting validation spectra...")

    scanned = 0

    for batch in parquet.iter_batches(
        columns=columns,
        batch_size=100_000,
    ):
        frame = batch.to_pandas()

        scanned += len(frame)

        mask = frame[
            "inchikey14"
        ].isin(
            validation_keys
        )

        if mask.any():
            query_frames.append(
                frame.loc[mask].copy()
            )

        print(
            f"\rRows scanned: {scanned:,}",
            end="",
            flush=True,
        )

    print()

    queries = pd.concat(
        query_frames,
        ignore_index=True,
    )

    print(
        f"Validation spectra: "
        f"{len(queries):,}"
    )

    #
    # Structure-level candidate generation.
    #

    evaluated = 0

    formula_hit_count = 0
    structure_hit_count = 0

    formula_candidate_counts: list[int] = []
    structure_candidate_counts: list[int] = []

    truth_formula_structure_counts: list[int] = []

    failures: list[
        dict[str, object]
    ] = []

    grouped = queries.groupby(
        "inchikey14",
        sort=False,
    )

    for structure_id, group in grouped:
        structure_id = str(
            structure_id
        )

        #
        # Truth formula.
        #

        formulas = (
            group["molecular_formula"]
            .dropna()
            .astype(str)
            .value_counts()
        )

        if formulas.empty:
            continue

        truth_formula = str(
            formulas.index[0]
        )

        #
        # Truth structure must exist in the structure
        # index for this internal-catalog experiment.
        #

        truth_structures = (
            formula_to_structures.get(
                truth_formula
            )
        )

        if not truth_structures:
            continue

        if (
            structure_id
            not in truth_structures
        ):
            continue

        #
        # Generate neutral-mass observations.
        #

        observed_masses: list[float] = []

        for row in group.itertuples(
            index=False
        ):
            try:
                precursor = float(
                    row.precursor_mz
                )
            except (TypeError, ValueError):
                continue

            if not np.isfinite(
                precursor
            ):
                continue

            mass = neutral_mass(
                precursor,
                str(row.adduct),
            )

            if mass is None:
                continue

            if not np.isfinite(mass):
                continue

            observed_masses.append(
                mass
            )

        if not observed_masses:
            continue

        evaluated += 1

        #
        # Union of formula candidates from all spectra.
        #

        formula_candidates: set[str] = set()

        for observed_mass in observed_masses:
            indices = (
                search_formula_index(
                    formula_index,
                    neutral_mass=(
                        observed_mass
                    ),
                    tolerance_ppm=(
                        TOLERANCE_PPM
                    ),
                )
            )

            formula_candidates.update(
                str(
                    formula_index.formulas[i]
                )
                for i in indices
            )

        formula_candidate_counts.append(
            len(formula_candidates)
        )

        formula_hit = (
            truth_formula
            in formula_candidates
        )

        if formula_hit:
            formula_hit_count += 1

        #
        # Expand candidate formulas into every structure
        # having one of those formulas.
        #

        structure_candidates: set[str] = set()

        for formula in formula_candidates:
            structure_candidates.update(
                formula_to_structures.get(
                    formula,
                    ()
                )
            )

        structure_candidate_counts.append(
            len(structure_candidates)
        )

        structure_hit = (
            structure_id
            in structure_candidates
        )

        if structure_hit:
            structure_hit_count += 1

        truth_formula_structure_counts.append(
            len(truth_structures)
        )

        if (
            not structure_hit
            and len(failures) < 30
        ):
            failures.append(
                {
                    "inchikey14": (
                        structure_id
                    ),
                    "truth_formula": (
                        truth_formula
                    ),
                    "usable_spectra": (
                        len(observed_masses)
                    ),
                    "formula_candidates": (
                        len(formula_candidates)
                    ),
                    "structure_candidates": (
                        len(structure_candidates)
                    ),
                }
            )

    #
    # Results.
    #

    print()
    print(
        "Formula -> structure candidate expansion"
    )
    print("=" * 60)

    print(
        f"Structures evaluated:      "
        f"{evaluated:,}"
    )

    if not evaluated:
        return

    print(
        f"Formula candidate recall:  "
        f"{formula_hit_count:,}/{evaluated:,} "
        f"({formula_hit_count / evaluated:.3%})"
    )

    print(
        f"Structure candidate recall:"
        f" {structure_hit_count:,}/{evaluated:,} "
        f"({structure_hit_count / evaluated:.3%})"
    )

    formulas = np.asarray(
        formula_candidate_counts,
        dtype=np.int64,
    )

    structures_per_query = np.asarray(
        structure_candidate_counts,
        dtype=np.int64,
    )

    truth_isomers = np.asarray(
        truth_formula_structure_counts,
        dtype=np.int64,
    )

    print()
    print("Formula candidates per query:")
    print(
        f"  median: "
        f"{np.median(formulas):.0f}"
    )
    print(
        f"  p90:    "
        f"{np.percentile(formulas, 90):.0f}"
    )
    print(
        f"  p95:    "
        f"{np.percentile(formulas, 95):.0f}"
    )
    print(
        f"  p99:    "
        f"{np.percentile(formulas, 99):.0f}"
    )
    print(
        f"  max:    "
        f"{formulas.max():,}"
    )

    print()
    print("Structure candidates per query:")
    print(
        f"  median: "
        f"{np.median(structures_per_query):.0f}"
    )
    print(
        f"  p90:    "
        f"{np.percentile(structures_per_query, 90):.0f}"
    )
    print(
        f"  p95:    "
        f"{np.percentile(structures_per_query, 95):.0f}"
    )
    print(
        f"  p99:    "
        f"{np.percentile(structures_per_query, 99):.0f}"
    )
    print(
        f"  max:    "
        f"{structures_per_query.max():,}"
    )

    print()
    print(
        "Structures sharing the truth formula:"
    )
    print(
        f"  median: "
        f"{np.median(truth_isomers):.0f}"
    )
    print(
        f"  p90:    "
        f"{np.percentile(truth_isomers, 90):.0f}"
    )
    print(
        f"  p95:    "
        f"{np.percentile(truth_isomers, 95):.0f}"
    )
    print(
        f"  p99:    "
        f"{np.percentile(truth_isomers, 99):.0f}"
    )
    print(
        f"  max:    "
        f"{truth_isomers.max():,}"
    )

    #
    # Useful threshold counts for designing the
    # eventual ranker.
    #

    print()
    print("Candidate-set thresholds:")

    for threshold in [
        25,
        50,
        100,
        250,
        500,
        1000,
    ]:
        count = int(
            np.sum(
                structures_per_query
                <= threshold
            )
        )

        print(
            f"  <= {threshold:4d}: "
            f"{count:>6,}/{evaluated:,} "
            f"({count / evaluated:.3%})"
        )

    if failures:
        print()
        print(
            "Example candidate-generation failures:"
        )

        print(
            pd.DataFrame(
                failures
            ).to_string(
                index=False
            )
        )


if __name__ == "__main__":
    main()