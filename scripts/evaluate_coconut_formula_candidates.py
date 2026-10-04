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
    "training_coconut_formulas.parquet"
)

TRAINING_STRUCTURES_PATH = Path(
    "data/processed/structure_index/"
    "training_structures.parquet"
)

COCONUT_PATH = Path(
    "data/external/coconut/"
    "coconut_approved.parquet"
)

TOLERANCE_PPM = 10.0


def load_formula_index() -> FormulaIndex:
    frame = pd.read_parquet(
        FORMULA_INDEX_PATH
    )

    masses = frame[
        "monoisotopic_mass"
    ].to_numpy(dtype=np.float64)

    if np.any(masses[1:] < masses[:-1]):
        raise ValueError(
            "Formula index is not sorted by mass"
        )

    return FormulaIndex(
        formulas=frame[
            "molecular_formula"
        ].to_numpy(dtype=object),
        masses=masses,
    )


def describe_counts(
    values: list[int],
) -> str:
    if not values:
        return "no values"

    counts = np.asarray(
        values,
        dtype=np.int64,
    )

    return (
        f"median={np.median(counts):.0f} "
        f"p90={np.percentile(counts, 90):.0f} "
        f"p95={np.percentile(counts, 95):.0f} "
        f"p99={np.percentile(counts, 99):.0f} "
        f"max={counts.max():,}"
    )


def main() -> None:
    #
    # Validation identities.
    #

    validation = pd.read_parquet(
        VALIDATION_PATH
    )

    validation_keys = set(
        validation["inchikey14"]
        .dropna()
        .astype(str)
    )

    #
    # Leakage-safe formula universe.
    #

    index = load_formula_index()

    formula_lookup = set(
        str(formula)
        for formula in index.formulas
    )

    #
    # Training-only structure catalog.
    #

    training = pd.read_parquet(
        TRAINING_STRUCTURES_PATH,
        columns=[
            "inchikey14",
            "molecular_formula",
        ],
    )

    training_keys = set(
        training["inchikey14"]
        .dropna()
        .astype(str)
    )

    validation_overlap = (
        training_keys & validation_keys
    )

    if validation_overlap:
        raise RuntimeError(
            "Training structure catalog contains "
            "validation structures"
        )

    #
    # Approved COCONUT catalog.
    #

    coconut = pd.read_parquet(
        COCONUT_PATH,
        columns=[
            "inchikey14",
            "molecular_formula",
        ],
    )

    coconut["inchikey14"] = (
        coconut["inchikey14"].astype(str)
    )

    coconut["molecular_formula"] = (
        coconut["molecular_formula"].astype(str)
    )

    coconut_keys = set(
        coconut["inchikey14"]
    )

    coconut_truth_keys = (
        validation_keys & coconut_keys
    )

    #
    # Formula -> structures.
    #

    training_by_formula = (
        training.groupby(
            "molecular_formula"
        )["inchikey14"]
        .agg(
            lambda values: set(
                values.astype(str)
            )
        )
        .to_dict()
    )

    coconut_by_formula = (
        coconut.groupby(
            "molecular_formula"
        )["inchikey14"]
        .agg(
            lambda values: set(
                values.astype(str)
            )
        )
        .to_dict()
    )

    training_formulas = set(
        training_by_formula
    )

    coconut_formulas = set(
        coconut_by_formula
    )

    #
    # Collect validation spectra from train.
    #

    parquet = pq.ParquetFile(
        TRAIN_PATH
    )

    columns = [
        "inchikey14",
        "molecular_formula",
        "adduct",
        "precursor_mz",
        "ingest_lib",
    ]

    query_frames: list[pd.DataFrame] = []

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

    if not query_frames:
        raise RuntimeError(
            "No validation spectra found"
        )

    queries = pd.concat(
        query_frames,
        ignore_index=True,
    )

    print(
        f"Validation spectra: "
        f"{len(queries):,}"
    )

    #
    # Cohort counters.
    #
    # Each cohort records structures rather than
    # individual spectra.
    #

    cohort_total: defaultdict[str, int] = (
        defaultdict(int)
    )

    cohort_usable: defaultdict[str, int] = (
        defaultdict(int)
    )

    cohort_formula_hit: defaultdict[str, int] = (
        defaultdict(int)
    )

    cohort_structure_hit: defaultdict[str, int] = (
        defaultdict(int)
    )

    cohort_formula_counts: defaultdict[
        str,
        list[int],
    ] = defaultdict(list)

    cohort_structure_counts: defaultdict[
        str,
        list[int],
    ] = defaultdict(list)

    #
    # Extra diagnostics for the most important
    # cohort: validation formulas supplied by
    # COCONUT but absent from training.
    #

    coconut_only_total = 0
    coconut_only_usable = 0
    coconut_only_formula_hit = 0
    coconut_only_structure_hit = 0

    #
    # Keep stage totals for all validation.
    #

    structures_seen = 0
    usable_structures = 0

    grouped = queries.groupby(
        "inchikey14",
        sort=False,
    )

    for structure_id, group in grouped:
        structure_id = str(
            structure_id
        )

        structures_seen += 1

        #
        # Truth formula is used ONLY for evaluation.
        # It is never inserted into the candidate set.
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
        # Determine evaluation cohorts.
        #

        in_coconut = (
            structure_id
            in coconut_truth_keys
        )

        formula_in_training = (
            truth_formula
            in training_formulas
        )

        formula_in_coconut = (
            truth_formula
            in coconut_formulas
        )

        cohorts = ["all"]

        if in_coconut:
            cohorts.append(
                "coconut_truth_available"
            )
        else:
            cohorts.append(
                "coconut_truth_unavailable"
            )

        if (
            in_coconut
            and formula_in_training
        ):
            cohorts.append(
                "coconut_truth_formula_in_training"
            )

        if (
            in_coconut
            and not formula_in_training
            and formula_in_coconut
        ):
            cohorts.append(
                "coconut_truth_formula_only_coconut"
            )

        for cohort in cohorts:
            cohort_total[cohort] += 1

        is_coconut_only_formula = (
            in_coconut
            and not formula_in_training
            and formula_in_coconut
        )

        if is_coconut_only_formula:
            coconut_only_total += 1

        #
        # Convert all usable spectra to neutral
        # masses, exactly as in the previous
        # consensus experiment.
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

            if not np.isfinite(precursor):
                continue

            adduct = str(
                row.adduct
            )

            observed = neutral_mass(
                precursor,
                adduct,
            )

            if observed is None:
                continue

            if not np.isfinite(observed):
                continue

            observed_masses.append(
                observed
            )

        if not observed_masses:
            continue

        usable_structures += 1

        for cohort in cohorts:
            cohort_usable[cohort] += 1

        if is_coconut_only_formula:
            coconut_only_usable += 1

        #
        # Same consensus behavior as the previous
        # experiment:
        #
        # UNION formula candidates produced by all
        # spectra for this structure.
        #

        formula_candidates: set[str] = set()

        for observed_mass in observed_masses:
            indices = search_formula_index(
                index,
                neutral_mass=observed_mass,
                tolerance_ppm=TOLERANCE_PPM,
            )

            formula_candidates.update(
                str(index.formulas[i])
                for i in indices
            )

        formula_hit = (
            truth_formula
            in formula_candidates
        )

        #
        # Expand every inferred formula into the
        # leakage-safe combined structure catalog.
        #

        structure_candidates: set[str] = set()

        for formula in formula_candidates:
            structure_candidates.update(
                training_by_formula.get(
                    formula,
                    set(),
                )
            )

            structure_candidates.update(
                coconut_by_formula.get(
                    formula,
                    set(),
                )
            )

        structure_hit = (
            structure_id
            in structure_candidates
        )

        for cohort in cohorts:
            cohort_formula_counts[
                cohort
            ].append(
                len(formula_candidates)
            )

            cohort_structure_counts[
                cohort
            ].append(
                len(structure_candidates)
            )

            if formula_hit:
                cohort_formula_hit[
                    cohort
                ] += 1

            if structure_hit:
                cohort_structure_hit[
                    cohort
                ] += 1

        if is_coconut_only_formula:
            if formula_hit:
                coconut_only_formula_hit += 1

            if structure_hit:
                coconut_only_structure_hit += 1

    #
    # Reporting.
    #

    print()
    print(
        "Leakage-safe COCONUT candidate evaluation"
    )
    print("=" * 70)

    print(
        f"Formula tolerance:          "
        f"{TOLERANCE_PPM:g} ppm"
    )

    print(
        f"Formula index:              "
        f"{len(index.formulas):,}"
    )

    print(
        f"Training structures:        "
        f"{len(training_keys):,}"
    )

    print(
        f"COCONUT structures:         "
        f"{len(coconut_keys):,}"
    )

    print(
        f"Validation structures:      "
        f"{len(validation_keys):,}"
    )

    print(
        f"Validation structures seen: "
        f"{structures_seen:,}"
    )

    print(
        f"Usable structures:          "
        f"{usable_structures:,}"
    )

    print()

    cohort_order = [
        "all",
        "coconut_truth_available",
        "coconut_truth_unavailable",
        "coconut_truth_formula_in_training",
        "coconut_truth_formula_only_coconut",
    ]

    names = {
        "all":
            "All validation",
        "coconut_truth_available":
            "COCONUT truth available",
        "coconut_truth_unavailable":
            "COCONUT truth unavailable",
        "coconut_truth_formula_in_training":
            "COCONUT truth + formula in training",
        "coconut_truth_formula_only_coconut":
            "COCONUT truth + formula only in COCONUT",
    }

    for cohort in cohort_order:
        total = cohort_total[cohort]
        usable = cohort_usable[cohort]

        print(names[cohort])
        print("-" * 70)

        print(
            f"Structures:                "
            f"{total:,}"
        )

        print(
            f"Usable:                    "
            f"{usable:,}"
        )

        if not usable:
            print()
            continue

        formula_hits = (
            cohort_formula_hit[cohort]
        )

        structure_hits = (
            cohort_structure_hit[cohort]
        )

        print(
            f"Formula candidate recall:  "
            f"{formula_hits:,}/{usable:,} "
            f"({formula_hits / usable:.3%})"
        )

        print(
            f"Structure candidate recall:"
            f" {structure_hits:,}/{usable:,} "
            f"({structure_hits / usable:.3%})"
        )

        print(
            "Formula candidates:        "
            + describe_counts(
                cohort_formula_counts[
                    cohort
                ]
            )
        )

        print(
            "Structure candidates:      "
            + describe_counts(
                cohort_structure_counts[
                    cohort
                ]
            )
        )

        print()

    #
    # Sanity check: for a truth that exists in
    # COCONUT, successful formula recovery should
    # normally imply successful structure recovery.
    #

    available_formula_hits = (
        cohort_formula_hit[
            "coconut_truth_available"
        ]
    )

    available_structure_hits = (
        cohort_structure_hit[
            "coconut_truth_available"
        ]
    )

    print("Consistency checks")
    print("-" * 70)

    print(
        "COCONUT-available formula hits:   "
        f"{available_formula_hits:,}"
    )

    print(
        "COCONUT-available structure hits: "
        f"{available_structure_hits:,}"
    )

    if (
        available_formula_hits
        != available_structure_hits
    ):
        print(
            "WARNING: formula and structure "
            "recall differ for COCONUT truths."
        )

    print()

    print("Most important novel-formula cohort")
    print("-" * 70)

    print(
        f"Structures:                "
        f"{coconut_only_total:,}"
    )

    print(
        f"Usable:                    "
        f"{coconut_only_usable:,}"
    )

    if coconut_only_usable:
        print(
            f"Formula recovered:         "
            f"{coconut_only_formula_hit:,}/"
            f"{coconut_only_usable:,} "
            f"("
            f"{coconut_only_formula_hit / coconut_only_usable:.3%}"
            f")"
        )

        print(
            f"Structure recovered:       "
            f"{coconut_only_structure_hit:,}/"
            f"{coconut_only_usable:,} "
            f"("
            f"{coconut_only_structure_hit / coconut_only_usable:.3%}"
            f")"
        )


if __name__ == "__main__":
    main()