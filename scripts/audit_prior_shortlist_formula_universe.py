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
ROOT = PROJECT_ROOT / "data/processed/blind_pseudotest"

TEST_PATH = ROOT / "test.parquet"
TRUTH_PATH = ROOT / "truth.parquet"

TRAINING_STRUCTURES_PATH = (
    PROJECT_ROOT
    / "data/processed/structure_index/training_structures.parquet"
)

PUBCHEM_FORMULAS_PATH = (
    PROJECT_ROOT
    / "data/processed/pubchem_formula_index/formula_buckets.parquet"
)

OUTPUT_PATH = (
    ROOT
    / "prior_shortlist_formula_universe.csv"
)

FIXED_PRIOR_WEIGHT = 0.50
CUTOFFS = [3, 10, 25, 50, 100]

ELEMENTS = [
    "C", "H", "B", "Br", "Cl", "F", "I",
    "N", "O", "P", "S", "Se", "Si",
]


def signature_for(composition: dict[str, int]) -> tuple[str, ...]:
    return tuple(
        element
        for element in ELEMENTS
        if composition.get(element, 0) > 0
    )


def build_training_prior():
    training = pd.read_parquet(
        TRAINING_STRUCTURES_PATH,
        columns=["molecular_formula"],
    )

    formulas = (
        training["molecular_formula"]
        .dropna()
        .astype(str)
        .drop_duplicates()
    )

    signature_counts: Counter[tuple[str, ...]] = Counter()
    count_histograms = defaultdict(lambda: defaultdict(Counter))
    signature_totals: Counter[tuple[str, ...]] = Counter()

    for formula in formulas:
        try:
            composition = parse_formula(formula)
        except ValueError:
            continue

        if "C" not in composition or "H" not in composition:
            continue

        signature = signature_for(composition)

        signature_counts[signature] += 1
        signature_totals[signature] += 1

        for element in signature:
            count_histograms[signature][element][
                int(composition[element])
            ] += 1

    return (
        signature_counts,
        count_histograms,
        signature_totals,
        sum(signature_counts.values()),
    )


def log_formula_prior(
    formula: str,
    signature_counts,
    count_histograms,
    signature_totals,
    total_formulas: int,
) -> float:
    try:
        composition = parse_formula(formula)
    except ValueError:
        return -1e9

    signature = signature_for(composition)

    signature_log_prior = log(
        (
            signature_counts.get(signature, 0)
            + 1.0
        )
        / (
            total_formulas
            + len(signature_counts)
        )
    )

    total_for_signature = signature_totals.get(signature, 0)

    if total_for_signature <= 0:
        return signature_log_prior

    count_log_prior = 0.0

    for element in signature:
        histogram = count_histograms[signature][element]
        observed_values = max(1, len(histogram))
        count = int(composition[element])

        count_log_prior += log(
            (
                histogram.get(count, 0)
                + 1.0
            )
            / (
                total_for_signature
                + observed_values
            )
        )

    if signature:
        count_log_prior /= len(signature)

    return signature_log_prior + count_log_prior


def main() -> None:
    for path in [
        TEST_PATH,
        TRUTH_PATH,
        TRAINING_STRUCTURES_PATH,
        PUBCHEM_FORMULAS_PATH,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"Missing required artifact: {path}"
            )

    test = pd.read_parquet(TEST_PATH).copy()
    truth = pd.read_parquet(TRUTH_PATH).copy()

    test["molecule_id"] = test["molecule_id"].astype(str)
    truth["molecule_id"] = truth["molecule_id"].astype(str)
    truth["molecular_formula"] = truth["molecular_formula"].astype(str)

    truth_lookup = dict(
        zip(
            truth["molecule_id"],
            truth["molecular_formula"],
            strict=True,
        )
    )

    current_formulas = set(
        pd.read_parquet(
            PUBCHEM_FORMULAS_PATH,
            columns=["molecular_formula"],
        )["molecular_formula"]
        .dropna()
        .astype(str)
    )

    rules = build_signature_rules(
        TRAINING_STRUCTURES_PATH
    )

    (
        signature_counts,
        count_histograms,
        signature_totals,
        total_formulas,
    ) = build_training_prior()

    union_by_cutoff = {
        cutoff: set()
        for cutoff in CUTOFFS
    }

    truth_retained = {
        cutoff: 0
        for cutoff in CUTOFFS
    }

    per_query_counts = {
        cutoff: []
        for cutoff in CUTOFFS
    }

    total_queries = 0

    for number, (molecule_id, group) in enumerate(
        test.groupby(
            "molecule_id",
            sort=False,
        ),
        start=1,
    ):
        total_queries += 1

        selected = select_query_spectra(group)

        merged: dict[str, float] = {}

        for spectrum in selected.itertuples(index=False):
            neutral_mass = neutral_mass_from_adduct(
                spectrum.precursor_mz,
                spectrum.adduct,
            )

            if not np.isfinite(neutral_mass):
                continue

            for formula, _, ppm, _ in enumerate_for_mass(
                neutral_mass,
                rules,
            ):
                abs_ppm = abs(float(ppm))

                current = merged.get(formula)

                if current is None or abs_ppm < current:
                    merged[formula] = abs_ppm

        ranked = sorted(
            (
                (
                    formula,
                    (
                        -abs_ppm
                        + FIXED_PRIOR_WEIGHT
                        * log_formula_prior(
                            formula,
                            signature_counts,
                            count_histograms,
                            signature_totals,
                            total_formulas,
                        )
                    ),
                )
                for formula, abs_ppm in merged.items()
            ),
            key=lambda item: (
                -item[1],
                item[0],
            ),
        )

        truth_formula = truth_lookup.get(
            str(molecule_id)
        )

        for cutoff in CUTOFFS:
            shortlist = [
                formula
                for formula, _
                in ranked[:cutoff]
            ]

            shortlist_set = set(shortlist)

            union_by_cutoff[cutoff].update(
                shortlist_set
            )

            per_query_counts[cutoff].append(
                len(shortlist)
            )

            if truth_formula in shortlist_set:
                truth_retained[cutoff] += 1

        if number % 50 == 0:
            print(
                f"  processed {number:,}/{test['molecule_id'].nunique():,}"
            )

    rows = []

    for cutoff in CUTOFFS:
        universe = union_by_cutoff[cutoff]

        existing = universe & current_formulas
        missing = universe - current_formulas

        rows.append(
            {
                "cutoff": cutoff,
                "queries": total_queries,
                "truth_retained": truth_retained[cutoff],
                "truth_retained_fraction": (
                    truth_retained[cutoff]
                    / total_queries
                    if total_queries
                    else 0.0
                ),
                "unique_shortlist_formulas": len(universe),
                "already_in_pubchem_asset": len(existing),
                "missing_from_pubchem_asset": len(missing),
                "formula_asset_coverage_fraction": (
                    len(existing) / len(universe)
                    if universe
                    else np.nan
                ),
                "mean_formulas_per_query": float(
                    np.mean(per_query_counts[cutoff])
                ),
            }
        )

    summary = pd.DataFrame(rows)

    summary.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print()
    print(
        "Prior-shortlisted formula universe"
    )
    print("=" * 110)

    print(
        summary.to_string(
            index=False,
            formatters={
                "truth_retained_fraction":
                    lambda value:
                        f"{value:.2%}",
                "formula_asset_coverage_fraction":
                    lambda value:
                        f"{value:.2%}",
                "mean_formulas_per_query":
                    lambda value:
                        f"{value:.1f}",
            },
        )
    )

    print()
    print("Wrote:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
