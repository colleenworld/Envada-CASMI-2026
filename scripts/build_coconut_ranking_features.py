from __future__ import annotations

from collections import Counter
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

OUTPUT_PATH = Path(
    "data/processed/results/"
    "coconut_ranking_features.parquet"
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


def ppm_error(
    observed: float,
    theoretical: float,
) -> float:
    return (
        (observed - theoretical)
        / theoretical
        * 1_000_000.0
    )


def clean_optional(
    value: object,
) -> object:
    if pd.isna(value):
        return None

    return value


def main() -> None:
    #
    # Validation population.
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
    # Formula index.
    #

    formula_frame = pd.read_parquet(
        FORMULA_INDEX_PATH
    )

    index = load_formula_index()

    formula_mass = dict(
        zip(
            formula_frame[
                "molecular_formula"
            ].astype(str),
            formula_frame[
                "monoisotopic_mass"
            ].astype(float),
            strict=True,
        )
    )

    #
    # Training structure catalog.
    #

    training = pd.read_parquet(
        TRAINING_STRUCTURES_PATH
    ).copy()

    training["inchikey14"] = (
        training["inchikey14"].astype(str)
    )

    training["molecular_formula"] = (
        training["molecular_formula"].astype(str)
    )

    training_keys = set(
        training["inchikey14"]
    )

    overlap = (
        training_keys & validation_keys
    )

    if overlap:
        raise RuntimeError(
            "Training structure catalog contains "
            "validation structures"
        )

    #
    # COCONUT structure catalog.
    #

    coconut = pd.read_parquet(
        COCONUT_PATH
    ).copy()

    coconut["inchikey14"] = (
        coconut["inchikey14"].astype(str)
    )

    coconut["molecular_formula"] = (
        coconut["molecular_formula"].astype(str)
    )

    coconut_keys = set(
        coconut["inchikey14"]
    )

    #
    # We are building the ranking benchmark only
    # for validation truths independently present
    # in approved COCONUT.
    #

    evaluation_keys = (
        validation_keys & coconut_keys
    )

    print(
        f"Validation structures: "
        f"{len(validation_keys):,}"
    )

    print(
        f"COCONUT-evaluable structures: "
        f"{len(evaluation_keys):,}"
    )

    #
    # Build one metadata record per candidate.
    #
    # A structure can occur in both training and
    # COCONUT. Merge the source flags rather than
    # duplicating the candidate.
    #

    candidate_metadata: dict[
        str,
        dict[str, object],
    ] = {}

    for row in training.itertuples(
        index=False
    ):
        key = str(row.inchikey14)

        candidate_metadata[key] = {
            "candidate_inchikey14": key,
            "candidate_formula": str(
                row.molecular_formula
            ),
            "in_training": True,
            "in_coconut": False,
            "np_likeness": None,
            "approved_collections": None,
            "chemical_class": None,
            "chemical_sub_class": None,
            "chemical_super_class": None,
            "np_classifier_pathway": None,
            "np_classifier_superclass": None,
            "np_classifier_class": None,
        }

    for row in coconut.itertuples(
        index=False
    ):
        key = str(row.inchikey14)

        if key in candidate_metadata:
            metadata = candidate_metadata[key]

            metadata["in_coconut"] = True

            #
            # Prefer COCONUT metadata for the
            # external NP-specific fields.
            #

            metadata["np_likeness"] = (
                clean_optional(
                    row.np_likeness
                )
            )

            metadata[
                "approved_collections"
            ] = clean_optional(
                row.approved_collections
            )

            metadata[
                "chemical_class"
            ] = clean_optional(
                row.chemical_class
            )

            metadata[
                "chemical_sub_class"
            ] = clean_optional(
                row.chemical_sub_class
            )

            metadata[
                "chemical_super_class"
            ] = clean_optional(
                row.chemical_super_class
            )

            metadata[
                "np_classifier_pathway"
            ] = clean_optional(
                row.np_classifier_pathway
            )

            metadata[
                "np_classifier_superclass"
            ] = clean_optional(
                row.np_classifier_superclass
            )

            metadata[
                "np_classifier_class"
            ] = clean_optional(
                row.np_classifier_class
            )

        else:
            candidate_metadata[key] = {
                "candidate_inchikey14": key,
                "candidate_formula": str(
                    row.molecular_formula
                ),
                "in_training": False,
                "in_coconut": True,
                "np_likeness": clean_optional(
                    row.np_likeness
                ),
                "approved_collections": (
                    clean_optional(
                        row.approved_collections
                    )
                ),
                "chemical_class": (
                    clean_optional(
                        row.chemical_class
                    )
                ),
                "chemical_sub_class": (
                    clean_optional(
                        row.chemical_sub_class
                    )
                ),
                "chemical_super_class": (
                    clean_optional(
                        row.chemical_super_class
                    )
                ),
                "np_classifier_pathway": (
                    clean_optional(
                        row.np_classifier_pathway
                    )
                ),
                "np_classifier_superclass": (
                    clean_optional(
                        row.np_classifier_superclass
                    )
                ),
                "np_classifier_class": (
                    clean_optional(
                        row.np_classifier_class
                    )
                ),
            }

    #
    # Formula -> candidate identities.
    #

    candidates_by_formula: dict[
        str,
        set[str],
    ] = {}

    for key, metadata in (
        candidate_metadata.items()
    ):
        formula = str(
            metadata["candidate_formula"]
        )

        candidates_by_formula.setdefault(
            formula,
            set(),
        ).add(key)

    print(
        f"Combined candidate structures: "
        f"{len(candidate_metadata):,}"
    )

    #
    # Collect spectra only for the COCONUT-
    # evaluable validation population.
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

    print(
        "Collecting evaluation spectra..."
    )

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
            evaluation_keys
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
            "No evaluation spectra found"
        )

    queries = pd.concat(
        query_frames,
        ignore_index=True,
    )

    print(
        f"Evaluation spectra: "
        f"{len(queries):,}"
    )

    #
    # Build ranking rows.
    #

    output_rows: list[
        dict[str, object]
    ] = []

    queries_with_candidates = 0
    queries_with_truth = 0
    queries_without_usable_spectra = 0

    candidate_counts: list[int] = []

    grouped = queries.groupby(
        "inchikey14",
        sort=False,
    )

    for query_key, group in grouped:
        query_key = str(query_key)

        #
        # Truth formula is evaluation metadata only.
        #

        truth_formulas = (
            group["molecular_formula"]
            .dropna()
            .astype(str)
            .value_counts()
        )

        if truth_formulas.empty:
            continue

        truth_formula = str(
            truth_formulas.index[0]
        )

        #
        # Gather usable neutral-mass observations.
        #

        observed_masses: list[float] = []
        observed_adducts: list[str] = []

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

            observed_adducts.append(
                adduct
            )

        if not observed_masses:
            queries_without_usable_spectra += 1
            continue

        #
        # For every inferred formula, retain the
        # individual mass-error observations.
        #
        # This lets us derive support count,
        # support fraction, best error and median
        # error without using the truth formula.
        #

        formula_errors: dict[
            str,
            list[float],
        ] = {}

        formula_spectrum_support: dict[
            str,
            set[int],
        ] = {}

        for spectrum_index, observed in enumerate(
            observed_masses
        ):
            indices = search_formula_index(
                index,
                neutral_mass=observed,
                tolerance_ppm=TOLERANCE_PPM,
            )

            for i in indices:
                formula = str(
                    index.formulas[i]
                )

                theoretical = formula_mass[
                    formula
                ]

                error = ppm_error(
                    observed,
                    theoretical,
                )

                formula_errors.setdefault(
                    formula,
                    [],
                ).append(error)

                formula_spectrum_support.setdefault(
                    formula,
                    set(),
                ).add(spectrum_index)

        inferred_formulas = set(
            formula_errors
        )

        #
        # Expand formulas into structure candidates.
        #

        query_candidates: set[str] = set()

        for formula in inferred_formulas:
            query_candidates.update(
                candidates_by_formula.get(
                    formula,
                    set(),
                )
            )

        if not query_candidates:
            continue

        queries_with_candidates += 1

        candidate_counts.append(
            len(query_candidates)
        )

        if query_key in query_candidates:
            queries_with_truth += 1

        #
        # Query-level metadata.
        #

        query_libraries = sorted(
            {
                str(value)
                for value in group[
                    "ingest_lib"
                ].dropna()
            }
        )

        adduct_counts = Counter(
            observed_adducts
        )

        #
        # Emit one row per query/candidate pair.
        #

        for candidate_key in sorted(
            query_candidates
        ):
            metadata = (
                candidate_metadata[
                    candidate_key
                ]
            )

            formula = str(
                metadata[
                    "candidate_formula"
                ]
            )

            errors = np.asarray(
                formula_errors[formula],
                dtype=np.float64,
            )

            support_count = len(
                formula_spectrum_support[
                    formula
                ]
            )

            support_fraction = (
                support_count
                / len(observed_masses)
            )

            output_rows.append(
                {
                    "query_inchikey14":
                        query_key,

                    "candidate_inchikey14":
                        candidate_key,

                    "is_truth":
                        candidate_key
                        == query_key,

                    "truth_formula":
                        truth_formula,

                    "candidate_formula":
                        formula,

                    #
                    # Query evidence.
                    #
                    "query_spectrum_count":
                        len(group),

                    "usable_spectrum_count":
                        len(observed_masses),

                    "query_library_count":
                        len(query_libraries),

                    "query_libraries":
                        "|".join(
                            query_libraries
                        ),

                    "query_adduct_count":
                        len(adduct_counts),

                    "query_adducts":
                        "|".join(
                            sorted(
                                adduct_counts
                            )
                        ),

                    #
                    # Formula/mass evidence.
                    #
                    "formula_support_count":
                        support_count,

                    "formula_support_fraction":
                        support_fraction,

                    "mass_error_best_abs_ppm":
                        float(
                            np.min(
                                np.abs(errors)
                            )
                        ),

                    "mass_error_median_abs_ppm":
                        float(
                            np.median(
                                np.abs(errors)
                            )
                        ),

                    "mass_error_mean_abs_ppm":
                        float(
                            np.mean(
                                np.abs(errors)
                            )
                        ),

                    "mass_error_signed_median_ppm":
                        float(
                            np.median(errors)
                        ),

                    #
                    # Candidate provenance.
                    #
                    "in_training":
                        metadata[
                            "in_training"
                        ],

                    "in_coconut":
                        metadata[
                            "in_coconut"
                        ],

                    "np_likeness":
                        metadata[
                            "np_likeness"
                        ],

                    "approved_collections":
                        metadata[
                            "approved_collections"
                        ],

                    "chemical_class":
                        metadata[
                            "chemical_class"
                        ],

                    "chemical_sub_class":
                        metadata[
                            "chemical_sub_class"
                        ],

                    "chemical_super_class":
                        metadata[
                            "chemical_super_class"
                        ],

                    "np_classifier_pathway":
                        metadata[
                            "np_classifier_pathway"
                        ],

                    "np_classifier_superclass":
                        metadata[
                            "np_classifier_superclass"
                        ],

                    "np_classifier_class":
                        metadata[
                            "np_classifier_class"
                        ],
                }
            )

    #
    # Save.
    #

    if not output_rows:
        raise RuntimeError(
            "No ranking rows were generated"
        )

    result = pd.DataFrame(
        output_rows
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_parquet(
        OUTPUT_PATH,
        index=False,
    )

    #
    # Validate the result.
    #

    duplicate_pairs = int(
        result.duplicated(
            subset=[
                "query_inchikey14",
                "candidate_inchikey14",
            ]
        ).sum()
    )

    if duplicate_pairs:
        raise RuntimeError(
            f"Generated {duplicate_pairs:,} "
            "duplicate query/candidate pairs"
        )

    #
    # Reporting.
    #

    print()
    print("COCONUT ranking feature table")
    print("=" * 70)

    print(
        f"Evaluation structures:        "
        f"{len(evaluation_keys):,}"
    )

    print(
        f"No usable spectra:            "
        f"{queries_without_usable_spectra:,}"
    )

    print(
        f"Queries with candidates:      "
        f"{queries_with_candidates:,}"
    )

    print(
        f"Queries containing truth:     "
        f"{queries_with_truth:,}"
    )

    print(
        f"Ranking rows:                 "
        f"{len(result):,}"
    )

    print(
        f"Duplicate query/candidate:    "
        f"{duplicate_pairs:,}"
    )

    counts = np.asarray(
        candidate_counts,
        dtype=np.int64,
    )

    print()
    print("Candidates per query")
    print("-" * 70)

    print(
        f"median={np.median(counts):.0f} "
        f"p90={np.percentile(counts, 90):.0f} "
        f"p95={np.percentile(counts, 95):.0f} "
        f"p99={np.percentile(counts, 99):.0f} "
        f"max={counts.max():,}"
    )

    print()
    print("Truth rows")
    print("-" * 70)

    truth_rows = result[
        result["is_truth"]
    ]

    print(
        f"Truth rows:                   "
        f"{len(truth_rows):,}"
    )

    print(
        f"Truth in training:            "
        f"{int(truth_rows['in_training'].sum()):,}"
    )

    print(
        f"Truth in COCONUT:             "
        f"{int(truth_rows['in_coconut'].sum()):,}"
    )

    print()

    print(
        f"Saved: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()