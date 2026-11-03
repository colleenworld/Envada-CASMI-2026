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

TRUTH_PATH = ROOT / "truth.parquet"
TEST_PATH = ROOT / "test.parquet"
TUNING_CASES_PATH = ROOT / "formula_recoverability.parquet"

TRAINING_STRUCTURES_PATH = (
    PROJECT_ROOT
    / "data/processed/structure_index/training_structures.parquet"
)

OUTPUT_PATH = ROOT / "formula_prior_validation.parquet"
SUMMARY_PATH = ROOT / "formula_prior_validation_summary.csv"

ELEMENTS = [
    "C", "H", "B", "Br", "Cl", "F", "I",
    "N", "O", "P", "S", "Se", "Si",
]

FIXED_PRIOR_WEIGHT = 0.50
CUT_OFFS = [3, 5, 10, 25, 50, 100]


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


def summarize(
    frame: pd.DataFrame,
    *,
    label: str,
) -> dict:
    row = {
        "population": label,
        "queries": len(frame),
        "truth_generated": int(frame["truth_generated"].sum()),
        "mass_median_rank": frame["mass_rank"].median(),
        "prior_median_rank": frame["prior_rank"].median(),
    }

    for cutoff in CUT_OFFS:
        mass_ok = (
            frame["mass_rank"].notna()
            & (frame["mass_rank"] <= cutoff)
        )
        prior_ok = (
            frame["prior_rank"].notna()
            & (frame["prior_rank"] <= cutoff)
        )

        row[f"mass_top{cutoff}"] = int(mass_ok.sum())
        row[f"mass_top{cutoff}_fraction"] = mass_ok.mean()

        row[f"prior_top{cutoff}"] = int(prior_ok.sum())
        row[f"prior_top{cutoff}_fraction"] = prior_ok.mean()

        row[f"delta_top{cutoff}_points"] = (
            prior_ok.mean()
            - mass_ok.mean()
        ) * 100.0

    return row


def main() -> None:
    for path in [
        TRUTH_PATH,
        TEST_PATH,
        TRAINING_STRUCTURES_PATH,
        TUNING_CASES_PATH,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"Missing required artifact: {path}"
            )

    truth = pd.read_parquet(TRUTH_PATH).copy()
    test = pd.read_parquet(TEST_PATH).copy()
    tuning_cases = pd.read_parquet(TUNING_CASES_PATH).copy()

    truth["molecule_id"] = truth["molecule_id"].astype(str)
    truth["molecular_formula"] = truth["molecular_formula"].astype(str)
    test["molecule_id"] = test["molecule_id"].astype(str)

    tuning_ids = set(
        tuning_cases.loc[
            tuning_cases[
                "recoverable_by_mass_rules"
            ]
            .fillna(False)
            .astype(bool),
            "molecule_id",
        ].astype(str)
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

    rows = []

    print(
        "Fixed-weight blind formula-prior validation"
    )
    print("=" * 96)
    print("Prior weight:", FIXED_PRIOR_WEIGHT)
    print("Blind queries:", f"{len(truth):,}")
    print("Weight-selection queries:", f"{len(tuning_ids):,}")
    print(
        "Independent queries:",
        f"{len(truth) - len(tuning_ids):,}",
    )

    for number, target in enumerate(
        truth.itertuples(index=False),
        start=1,
    ):
        molecule_id = str(target.molecule_id)
        truth_formula = str(target.molecular_formula)

        selected = select_query_spectra(
            test.loc[
                test["molecule_id"]
                == molecule_id
            ]
        )

        merged: dict[str, float] = {}

        for spectrum in selected.itertuples(
            index=False
        ):
            neutral_mass = neutral_mass_from_adduct(
                spectrum.precursor_mz,
                spectrum.adduct,
            )

            if not np.isfinite(neutral_mass):
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
                abs_ppm = abs(float(ppm))

                current = merged.get(formula)

                if current is None or abs_ppm < current:
                    merged[formula] = abs_ppm

        if merged:
            candidate_rows = pd.DataFrame(
                [
                    {
                        "formula": formula,
                        "abs_ppm": abs_ppm,
                        "log_formula_prior":
                            log_formula_prior(
                                formula,
                                signature_counts,
                                count_histograms,
                                signature_totals,
                                total_formulas,
                            ),
                    }
                    for (
                        formula,
                        abs_ppm,
                    ) in merged.items()
                ]
            )

            mass_ranked = (
                candidate_rows
                .sort_values(
                    [
                        "abs_ppm",
                        "formula",
                    ],
                    ascending=[
                        True,
                        True,
                    ],
                    kind="stable",
                )
                .reset_index(
                    drop=True
                )
            )

            mass_ranked["mass_rank"] = (
                np.arange(
                    len(mass_ranked)
                )
                + 1
            )

            candidate_rows[
                "prior_score"
            ] = (
                -candidate_rows["abs_ppm"]
                + (
                    FIXED_PRIOR_WEIGHT
                    * candidate_rows[
                        "log_formula_prior"
                    ]
                )
            )

            prior_ranked = (
                candidate_rows
                .sort_values(
                    [
                        "prior_score",
                        "formula",
                    ],
                    ascending=[
                        False,
                        True,
                    ],
                    kind="stable",
                )
                .reset_index(
                    drop=True
                )
            )

            prior_ranked["prior_rank"] = (
                np.arange(
                    len(prior_ranked)
                )
                + 1
            )

            mass_truth = (
                mass_ranked.loc[
                    mass_ranked["formula"]
                    == truth_formula,
                    "mass_rank",
                ]
            )

            prior_truth = (
                prior_ranked.loc[
                    prior_ranked["formula"]
                    == truth_formula,
                    "prior_rank",
                ]
            )

            mass_rank = (
                int(mass_truth.iloc[0])
                if not mass_truth.empty
                else None
            )

            prior_rank = (
                int(prior_truth.iloc[0])
                if not prior_truth.empty
                else None
            )

            truth_generated = (
                mass_rank is not None
            )

        else:
            mass_rank = None
            prior_rank = None
            truth_generated = False

        rows.append(
            {
                "molecule_id":
                    molecule_id,
                "truth_formula":
                    truth_formula,
                "used_for_weight_selection":
                    molecule_id
                    in tuning_ids,
                "generated_formula_count":
                    len(merged),
                "truth_generated":
                    truth_generated,
                "mass_rank":
                    mass_rank,
                "prior_rank":
                    prior_rank,
            }
        )

        if (
            number % 50 == 0
            or number == len(truth)
        ):
            print(
                f"  processed "
                f"{number:,}/{len(truth):,}"
            )

    result = pd.DataFrame(rows)

    result.to_parquet(
        OUTPUT_PATH,
        index=False,
        compression="zstd",
    )

    independent = result.loc[
        ~result[
            "used_for_weight_selection"
        ]
    ]

    selected = result.loc[
        result[
            "used_for_weight_selection"
        ]
    ]

    summaries = pd.DataFrame(
        [
            summarize(
                result,
                label="all_blind",
            ),
            summarize(
                independent,
                label="independent_943",
            ),
            summarize(
                selected,
                label="weight_selection_57",
            ),
        ]
    )

    summaries.to_csv(
        SUMMARY_PATH,
        index=False,
    )

    columns = [
        "population",
        "queries",
        "truth_generated",
        "mass_top3",
        "mass_top3_fraction",
        "prior_top3",
        "prior_top3_fraction",
        "delta_top3_points",
        "mass_top10",
        "mass_top10_fraction",
        "prior_top10",
        "prior_top10_fraction",
        "delta_top10_points",
        "mass_top25",
        "mass_top25_fraction",
        "prior_top25",
        "prior_top25_fraction",
        "delta_top25_points",
        "mass_median_rank",
        "prior_median_rank",
    ]

    print()
    print(
        "Fixed-weight validation summary"
    )
    print("=" * 120)

    print(
        summaries[
            columns
        ].to_string(
            index=False,
            formatters={
                "mass_top3_fraction":
                    lambda value:
                        f"{value:.2%}",
                "prior_top3_fraction":
                    lambda value:
                        f"{value:.2%}",
                "delta_top3_points":
                    lambda value:
                        f"{value:+.2f}",
                "mass_top10_fraction":
                    lambda value:
                        f"{value:.2%}",
                "prior_top10_fraction":
                    lambda value:
                        f"{value:.2%}",
                "delta_top10_points":
                    lambda value:
                        f"{value:+.2f}",
                "mass_top25_fraction":
                    lambda value:
                        f"{value:.2%}",
                "prior_top25_fraction":
                    lambda value:
                        f"{value:.2%}",
                "delta_top25_points":
                    lambda value:
                        f"{value:+.2f}",
                "mass_median_rank":
                    lambda value:
                        (
                            "n/a"
                            if pd.isna(value)
                            else f"{value:.1f}"
                        ),
                "prior_median_rank":
                    lambda value:
                        (
                            "n/a"
                            if pd.isna(value)
                            else f"{value:.1f}"
                        ),
            },
        )
    )

    print()
    print("Wrote:", OUTPUT_PATH)
    print("Wrote:", SUMMARY_PATH)


if __name__ == "__main__":
    main()
