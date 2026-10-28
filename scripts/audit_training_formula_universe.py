from __future__ import annotations

from collections import Counter, defaultdict
from math import log
from pathlib import Path

import numpy as np
import pandas as pd

from casmi26.chemistry.formula import parse_formula
from casmi26.chemistry.formula_enumeration import (
    build_signature_rules,
    enumerate_for_mass,
    neutral_mass_from_adduct,
    select_query_spectra,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent

UNIVERSE_PATH = (
    PROJECT_ROOT
    / "data/processed/formula_index/training_prior_formula_universe_top25.parquet"
)

BLIND_TEST_PATH = (
    PROJECT_ROOT
    / "data/processed/blind_pseudotest/test.parquet"
)

BLIND_TRUTH_PATH = (
    PROJECT_ROOT
    / "data/processed/blind_pseudotest/truth.parquet"
)

TRAINING_STRUCTURES_PATH = (
    PROJECT_ROOT
    / "data/processed/structure_index/training_structures.parquet"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data/processed/blind_pseudotest/"
      "training_formula_universe_coverage.csv"
)

FIXED_PRIOR_WEIGHT = 0.50
TOP_K = 25

CHECKPOINTS = [
    1_000,
    2_500,
    5_000,
    10_000,
    20_000,
]

ELEMENTS = [
    "C", "H", "B", "Br", "Cl", "F", "I",
    "N", "O", "P", "S", "Se", "Si",
]


def signature_for(
    composition: dict[str, int],
) -> tuple[str, ...]:
    return tuple(
        element
        for element in ELEMENTS
        if composition.get(element, 0) > 0
    )


def build_training_prior():
    training = pd.read_parquet(
        TRAINING_STRUCTURES_PATH,
        columns=[
            "molecular_formula",
        ],
    )

    formulas = (
        training[
            "molecular_formula"
        ]
        .dropna()
        .astype(str)
        .drop_duplicates()
    )

    signature_counts: Counter[
        tuple[str, ...]
    ] = Counter()

    count_histograms = defaultdict(
        lambda: defaultdict(Counter)
    )

    signature_totals: Counter[
        tuple[str, ...]
    ] = Counter()

    parsed_count = 0

    for formula in formulas:
        try:
            composition = parse_formula(
                formula
            )
        except ValueError:
            continue

        if (
            "C" not in composition
            or "H" not in composition
        ):
            continue

        parsed_count += 1

        signature = signature_for(
            composition
        )

        signature_counts[
            signature
        ] += 1

        signature_totals[
            signature
        ] += 1

        for element in signature:
            count_histograms[
                signature
            ][
                element
            ][
                int(
                    composition[
                        element
                    ]
                )
            ] += 1

    return (
        signature_counts,
        count_histograms,
        signature_totals,
        parsed_count,
    )


def log_formula_prior(
    formula: str,
    signature_counts,
    count_histograms,
    signature_totals,
    total_formulas: int,
) -> float:
    try:
        composition = parse_formula(
            formula
        )
    except ValueError:
        return -1e9

    signature = signature_for(
        composition
    )

    signature_log_prior = log(
        (
            signature_counts.get(
                signature,
                0,
            )
            + 1.0
        )
        / (
            total_formulas
            + len(
                signature_counts
            )
        )
    )

    total_for_signature = (
        signature_totals.get(
            signature,
            0,
        )
    )

    if total_for_signature <= 0:
        return signature_log_prior

    count_log_prior = 0.0

    for element in signature:
        histogram = (
            count_histograms[
                signature
            ][
                element
            ]
        )

        observed_values = max(
            1,
            len(histogram),
        )

        count = int(
            composition[element]
        )

        count_log_prior += log(
            (
                histogram.get(
                    count,
                    0,
                )
                + 1.0
            )
            / (
                total_for_signature
                + observed_values
            )
        )

    if signature:
        count_log_prior /= len(
            signature
        )

    return (
        signature_log_prior
        + count_log_prior
    )


def build_blind_shortlists(
    test: pd.DataFrame,
) -> dict[str, list[str]]:
    rules = build_signature_rules(
        TRAINING_STRUCTURES_PATH
    )

    (
        signature_counts,
        count_histograms,
        signature_totals,
        total_formulas,
    ) = build_training_prior()

    shortlists = {}

    molecule_count = (
        test[
            "molecule_id"
        ]
        .nunique()
    )

    for number, (
        molecule_id,
        group,
    ) in enumerate(
        test.groupby(
            "molecule_id",
            sort=False,
        ),
        start=1,
    ):
        selected = select_query_spectra(
            group
        )

        merged: dict[
            str,
            float,
        ] = {}

        for spectrum in selected.itertuples(
            index=False
        ):
            neutral_mass = neutral_mass_from_adduct(
                spectrum.precursor_mz,
                spectrum.adduct,
            )

            if not np.isfinite(
                neutral_mass
            ):
                continue

            for (
                formula,
                _,
                ppm,
                _,
            ) in enumerate_for_mass(
                neutral_mass,
                rules,
            ):
                abs_ppm = abs(
                    float(ppm)
                )

                existing = merged.get(
                    formula
                )

                if (
                    existing is None
                    or abs_ppm < existing
                ):
                    merged[
                        formula
                    ] = abs_ppm

        ranked = []

        for (
            formula,
            abs_ppm,
        ) in merged.items():
            prior = log_formula_prior(
                formula,
                signature_counts,
                count_histograms,
                signature_totals,
                total_formulas,
            )

            score = (
                -abs_ppm
                + FIXED_PRIOR_WEIGHT
                * prior
            )

            ranked.append(
                (
                    formula,
                    score,
                )
            )

        ranked.sort(
            key=lambda item: (
                -item[1],
                item[0],
            )
        )

        shortlists[
            str(molecule_id)
        ] = [
            formula
            for formula, _
            in ranked[
                :TOP_K
            ]
        ]

        if (
            number % 100 == 0
            or number == molecule_count
        ):
            print(
                f"  blind shortlist "
                f"{number:,}/{molecule_count:,}"
            )

    return shortlists


def main() -> None:
    for path in [
        UNIVERSE_PATH,
        BLIND_TEST_PATH,
        BLIND_TRUTH_PATH,
        TRAINING_STRUCTURES_PATH,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"Missing required artifact: {path}"
            )

    universe = pd.read_parquet(
        UNIVERSE_PATH
    ).copy()

    required_universe = {
        "molecular_formula",
        "first_anchor_index",
    }

    missing = (
        required_universe
        - set(universe.columns)
    )

    if missing:
        raise ValueError(
            "Formula universe missing columns: "
            f"{sorted(missing)}"
        )

    test = pd.read_parquet(
        BLIND_TEST_PATH
    ).copy()

    test[
        "molecule_id"
    ] = (
        test[
            "molecule_id"
        ]
        .astype(str)
    )

    truth = pd.read_parquet(
        BLIND_TRUTH_PATH,
        columns=[
            "molecule_id",
            "molecular_formula",
        ],
    ).copy()

    truth[
        "molecule_id"
    ] = (
        truth[
            "molecule_id"
        ]
        .astype(str)
    )

    truth[
        "molecular_formula"
    ] = (
        truth[
            "molecular_formula"
        ]
        .astype(str)
    )

    print(
        "Recomputing frozen blind Top-25 formula shortlists..."
    )

    shortlists = build_blind_shortlists(
        test
    )

    truth_lookup = dict(
        zip(
            truth[
                "molecule_id"
            ],
            truth[
                "molecular_formula"
            ],
            strict=True,
        )
    )

    max_anchor = int(
        universe[
            "first_anchor_index"
        ].max()
    )

    checkpoints = sorted(
        {
            checkpoint
            for checkpoint in CHECKPOINTS
            if checkpoint <= max_anchor
        }
        | {
            max_anchor,
        }
    )

    rows = []

    total_shortlist_slots = sum(
        len(formulas)
        for formulas
        in shortlists.values()
    )

    for checkpoint in checkpoints:
        formulas = set(
            universe.loc[
                universe[
                    "first_anchor_index"
                ]
                <= checkpoint,
                "molecular_formula",
            ]
            .astype(str)
        )

        truth_hits = 0
        generated_truths = 0
        generated_truth_hits = 0
        covered_slots = 0
        fully_covered_queries = 0
        top3_fully_covered_queries = 0
        top10_fully_covered_queries = 0

        for (
            molecule_id,
            shortlist,
        ) in shortlists.items():
            shortlist_set = set(
                shortlist
            )

            covered_slots += sum(
                1
                for formula
                in shortlist
                if formula in formulas
            )

            if (
                shortlist
                and shortlist_set <= formulas
            ):
                fully_covered_queries += 1

            top3 = set(
                shortlist[:3]
            )

            if (
                top3
                and top3 <= formulas
            ):
                top3_fully_covered_queries += 1

            top10 = set(
                shortlist[:10]
            )

            if (
                top10
                and top10 <= formulas
            ):
                top10_fully_covered_queries += 1

            truth_formula = truth_lookup.get(
                molecule_id
            )

            if truth_formula in formulas:
                truth_hits += 1

            if truth_formula in shortlist_set:
                generated_truths += 1

                if truth_formula in formulas:
                    generated_truth_hits += 1

        query_count = len(
            shortlists
        )

        rows.append(
            {
                "anchor_count":
                    checkpoint,
                "universe_formula_count":
                    len(formulas),
                "blind_top25_slot_coverage":
                    (
                        covered_slots
                        / total_shortlist_slots
                        if total_shortlist_slots
                        else 0.0
                    ),
                "blind_queries_full_top3_covered":
                    top3_fully_covered_queries,
                "blind_queries_full_top3_coverage":
                    (
                        top3_fully_covered_queries
                        / query_count
                        if query_count
                        else 0.0
                    ),
                "blind_queries_full_top10_covered":
                    top10_fully_covered_queries,
                "blind_queries_full_top10_coverage":
                    (
                        top10_fully_covered_queries
                        / query_count
                        if query_count
                        else 0.0
                    ),
                "blind_queries_full_top25_covered":
                    fully_covered_queries,
                "blind_queries_full_top25_coverage":
                    (
                        fully_covered_queries
                        / query_count
                        if query_count
                        else 0.0
                    ),
                "blind_truth_formulas_covered":
                    truth_hits,
                "blind_truth_formula_coverage":
                    (
                        truth_hits
                        / query_count
                        if query_count
                        else 0.0
                    ),
                "blind_truths_in_generated_top25":
                    generated_truths,
                "generated_top25_truths_covered":
                    generated_truth_hits,
                "generated_top25_truth_coverage":
                    (
                        generated_truth_hits
                        / generated_truths
                        if generated_truths
                        else 0.0
                    ),
            }
        )

    summary = pd.DataFrame(
        rows
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print()
    print(
        "Hidden-independent training formula universe coverage"
    )
    print("=" * 150)

    percentage_columns = [
        "blind_top25_slot_coverage",
        "blind_queries_full_top3_coverage",
        "blind_queries_full_top10_coverage",
        "blind_queries_full_top25_coverage",
        "blind_truth_formula_coverage",
        "generated_top25_truth_coverage",
    ]

    print(
        summary.to_string(
            index=False,
            formatters={
                column:
                    lambda value:
                        f"{value:.2%}"
                for column
                in percentage_columns
            },
        )
    )

    print()
    print(
        "Wrote:",
        OUTPUT_PATH,
    )


if __name__ == "__main__":
    main()
