from __future__ import annotations

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

TOLERANCES_PPM = [5.0, 10.0, 20.0]


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
    # Load validation structure IDs.
    #

    manifest = pq.read_table(
        VALIDATION_PATH
    ).to_pandas()

    if "inchikey14" not in manifest.columns:
        raise ValueError(
            "Validation manifest does not contain "
            "'inchikey14'. "
            f"Columns: {manifest.columns.tolist()}"
        )

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
    # Load theoretical formula index.
    #

    index = load_formula_index()

    formula_lookup = set(
        index.formulas.tolist()
    )

    print(
        f"Formula index: "
        f"{len(index.formulas):,} formulas"
    )

    #
    # Collect all spectra belonging to the
    # structure-disjoint validation population.
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
        "ingest_lib",
    ]

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
            "No validation spectra were found"
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
    # Evaluation counters.
    #
    # A structure must have:
    #
    # 1. a truth formula that exists in our
    #    theoretical formula index;
    #
    # 2. at least one spectrum whose adduct can
    #    be converted to a neutral mass.
    #
    # We keep these conditions separate so that
    # adduct/formula coverage problems don't get
    # counted as mass-window failures.
    #

    truth_indexable = 0
    usable_structures = 0

    no_usable_spectra: list[str] = []

    usable_spectrum_counts: list[int] = []

    candidate_recall = {
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

    misses_10ppm: set[str] = set()
    hits_20ppm: set[str] = set()

    #
    # Keep some metadata for the structures that
    # remain missing even at 20 ppm. This will make
    # the next diagnostic easier if we need it.
    #

    structure_details: dict[
        str,
        dict[str, object],
    ] = {}

    grouped = queries.groupby(
        "inchikey14",
        sort=False,
    )

    for structure_id, group in grouped:
        #
        # A structure should normally have one
        # molecular formula. Use the most common
        # annotation to avoid allowing an isolated
        # source record to define the truth.
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
        # Charged/otherwise unsupported formula
        # strings were deliberately excluded when
        # we built the formula index.
        #

        if truth_formula not in formula_lookup:
            continue

        truth_indexable += 1

        #
        # Convert every usable spectrum for this
        # structure to a neutral-mass observation.
        #

        observed_masses: list[float] = []

        adducts: set[str] = set()
        libraries: set[str] = set()

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

            observed_neutral_mass = (
                neutral_mass(
                    precursor,
                    adduct,
                )
            )

            if observed_neutral_mass is None:
                continue

            if not np.isfinite(
                observed_neutral_mass
            ):
                continue

            observed_masses.append(
                observed_neutral_mass
            )

            adducts.add(adduct)

            if pd.notna(row.ingest_lib):
                libraries.add(
                    str(row.ingest_lib)
                )

        usable_spectrum_counts.append(
            len(observed_masses)
        )

        structure_details[
            str(structure_id)
        ] = {
            "truth_formula": truth_formula,
            "usable_spectra": (
                len(observed_masses)
            ),
            "adducts": sorted(adducts),
            "libraries": sorted(libraries),
        }

        if not observed_masses:
            no_usable_spectra.append(
                str(structure_id)
            )
            continue

        usable_structures += 1

        #
        # Build the UNION of formula candidates
        # produced by all spectra belonging to this
        # structure.
        #
        # We do this independently at each tolerance.
        #

        for tolerance in TOLERANCES_PPM:
            candidates: set[str] = set()

            for observed_mass in (
                observed_masses
            ):
                indices = (
                    search_formula_index(
                        index,
                        neutral_mass=(
                            observed_mass
                        ),
                        tolerance_ppm=(
                            tolerance
                        ),
                    )
                )

                candidates.update(
                    str(index.formulas[i])
                    for i in indices
                )

            candidate_counts[
                tolerance
            ].append(
                len(candidates)
            )

            hit = (
                truth_formula
                in candidates
            )

            if hit:
                candidate_recall[
                    tolerance
                ] += 1

            if (
                tolerance == 10.0
                and not hit
            ):
                misses_10ppm.add(
                    str(structure_id)
                )

            if (
                tolerance == 20.0
                and hit
            ):
                hits_20ppm.add(
                    str(structure_id)
                )

    #
    # Compare 10 ppm with 20 ppm.
    #

    recovered_at_20 = (
        misses_10ppm
        & hits_20ppm
    )

    still_missing = (
        misses_10ppm
        - hits_20ppm
    )

    #
    # Results.
    #

    print()
    print(
        "Structure-level formula candidate coverage"
    )
    print("=" * 60)

    print(
        f"Validation structures:        "
        f"{len(validation_keys):,}"
    )

    print(
        f"Truth formula indexable:      "
        f"{truth_indexable:,}"
    )

    print(
        f"At least one usable spectrum: "
        f"{usable_structures:,}"
    )

    print(
        f"No usable spectra:            "
        f"{len(no_usable_spectra):,}"
    )

    print()

    for tolerance in TOLERANCES_PPM:
        hits = candidate_recall[
            tolerance
        ]

        recall = (
            hits / usable_structures
            if usable_structures
            else 0.0
        )

        counts = np.asarray(
            candidate_counts[
                tolerance
            ],
            dtype=np.int64,
        )

        print(
            f"{tolerance:g} ppm"
        )

        print(
            f"  Candidate recall: "
            f"{hits:,}/{usable_structures:,} "
            f"({recall:.3%})"
        )

        if len(counts):
            print(
                f"  Candidate count: "
                f"median="
                f"{np.median(counts):.0f} "
                f"p90="
                f"{np.percentile(counts, 90):.0f} "
                f"p95="
                f"{np.percentile(counts, 95):.0f} "
                f"p99="
                f"{np.percentile(counts, 99):.0f} "
                f"max="
                f"{counts.max():,}"
            )

        print()

    #
    # Specifically quantify what widening from
    # 10 ppm to 20 ppm buys us.
    #

    print("10 ppm misses")
    print("-" * 60)

    print(
        f"Missed at 10 ppm:            "
        f"{len(misses_10ppm):,}"
    )

    print(
        f"Recovered at 20 ppm:         "
        f"{len(recovered_at_20):,}"
    )

    print(
        f"Still missing at 20 ppm:     "
        f"{len(still_missing):,}"
    )

    if misses_10ppm:
        recovery_rate = (
            len(recovered_at_20)
            / len(misses_10ppm)
        )

        print(
            f"20 ppm recovery rate:        "
            f"{recovery_rate:.3%}"
        )

    #
    # Distribution of usable spectra.
    #

    spectra = np.asarray(
        usable_spectrum_counts,
        dtype=np.int64,
    )

    print()
    print(
        "Usable spectra per indexable structure:"
    )

    if len(spectra):
        print(
            f"  median: "
            f"{np.median(spectra):.0f}"
        )

        print(
            f"  p90:    "
            f"{np.percentile(spectra, 90):.0f}"
        )

        print(
            f"  p95:    "
            f"{np.percentile(spectra, 95):.0f}"
        )

        print(
            f"  max:    "
            f"{spectra.max():,}"
        )

    #
    # Show some remaining 20-ppm failures.
    #

    if still_missing:
        print()
        print(
            "Example structures still missing "
            "at 20 ppm:"
        )

        print()

        for structure_id in sorted(
            still_missing
        )[:30]:
            details = structure_details[
                structure_id
            ]

            print(
                f"  {structure_id} "
                f"formula="
                f"{details['truth_formula']} "
                f"spectra="
                f"{details['usable_spectra']} "
                f"adducts="
                f"{details['adducts']} "
                f"libraries="
                f"{details['libraries']}"
            )

    #
    # Separately show structures where the formula
    # is valid/indexable but none of their spectra
    # use an adduct we currently understand.
    #

    if no_usable_spectra:
        print()
        print(
            "Example structures with no usable spectra:"
        )

        print()

        for structure_id in sorted(
            no_usable_spectra
        )[:20]:
            details = structure_details[
                structure_id
            ]

            print(
                f"  {structure_id} "
                f"formula="
                f"{details['truth_formula']} "
                f"libraries="
                f"{details['libraries']}"
            )


if __name__ == "__main__":
    main()