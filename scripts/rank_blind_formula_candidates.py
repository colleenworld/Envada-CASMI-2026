from __future__ import annotations

from collections import Counter, defaultdict
from math import log
from pathlib import Path

import numpy as np
import pandas as pd

from casmi26.chemistry.formula import parse_formula
from scripts.enumerate_blind_formula_candidates import (
    build_signature_rules,
    enumerate_for_mass,
    neutral_mass_from_adduct,
    select_query_spectra,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent

ROOT = (
    PROJECT_ROOT
    / "data/processed/blind_pseudotest"
)

RECOVERABILITY_PATH = (
    ROOT
    / "formula_recoverability.parquet"
)

TEST_PATH = (
    ROOT
    / "test.parquet"
)

TRAINING_STRUCTURES_PATH = (
    PROJECT_ROOT
    / "data/processed/structure_index/"
    "training_structures.parquet"
)

OUTPUT_PATH = (
    ROOT
    / "formula_prior_ranking.parquet"
)

SUMMARY_PATH = (
    ROOT
    / "formula_prior_ranking_summary.csv"
)

ELEMENTS = [
    "C",
    "H",
    "B",
    "Br",
    "Cl",
    "F",
    "I",
    "N",
    "O",
    "P",
    "S",
    "Se",
    "Si",
]

CUT_OFFS = [
    3,
    5,
    10,
    25,
    50,
    100,
]


def signature_for(
    composition: dict[str, int],
) -> tuple[str, ...]:
    return tuple(
        element
        for element in ELEMENTS
        if composition.get(
            element,
            0,
        )
        > 0
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

    count_histograms: dict[
        tuple[str, ...],
        dict[
            str,
            Counter[int],
        ],
    ] = defaultdict(
        lambda:
            defaultdict(
                Counter
            )
    )

    signature_totals: Counter[
        tuple[str, ...]
    ] = Counter()

    for formula in formulas:
        try:
            composition = parse_formula(
                formula
            )
        except ValueError:
            continue

        if (
            "C"
            not in composition
            or "H"
            not in composition
        ):
            continue

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

    total_formulas = sum(
        signature_counts.values()
    )

    return (
        signature_counts,
        count_histograms,
        signature_totals,
        total_formulas,
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

    signature_count = (
        signature_counts.get(
            signature,
            0
        )
    )

    # Add-one smoothing across observed signatures.
    signature_log_prior = log(
        (
            signature_count
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
            0
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
            len(
                histogram
            )
        )

        count = int(
            composition[
                element
            ]
        )

        count_log_prior += log(
            (
                histogram.get(
                    count,
                    0
                )
                + 1.0
            )
            / (
                total_for_signature
                + observed_values
            )
        )

    # Normalize the count contribution by signature size so larger
    # signatures are not automatically penalized simply for having
    # more elements.
    if signature:
        count_log_prior /= len(
            signature
        )

    return (
        signature_log_prior
        + count_log_prior
    )


def ranks_for_weight(
    frame: pd.DataFrame,
    weight: float,
) -> pd.Series:
    work = frame.copy()

    # Scale mass error onto roughly the same numeric range as the
    # log prior. Lower |ppm| is better; larger prior is better.
    work[
        "combined_score"
    ] = (
        -work[
            "abs_ppm"
        ]
        + (
            weight
            * work[
                "log_formula_prior"
            ]
        )
    )

    work = (
        work.sort_values(
            [
                "molecule_id",
                "combined_score",
                "formula",
            ],
            ascending=[
                True,
                False,
                True,
            ],
            kind="stable",
        )
        .copy()
    )

    work[
        "rank"
    ] = (
        work.groupby(
            "molecule_id"
        )
        .cumcount()
        + 1
    )

    truth = (
        work.loc[
            work[
                "is_truth"
            ]
        ]
        .set_index(
            "molecule_id"
        )[
            "rank"
        ]
    )

    return truth


def summarize_ranks(
    truth_ranks: pd.Series,
    molecule_ids: list[str],
    weight: float,
) -> dict:
    aligned = (
        truth_ranks.reindex(
            molecule_ids
        )
    )

    row = {
        "prior_weight":
            weight,
        "truth_generated":
            int(
                aligned.notna().sum()
            ),
    }

    for cutoff in CUT_OFFS:
        retained = (
            aligned.notna()
            & (
                aligned
                <= cutoff
            )
        )

        row[
            f"top{cutoff}"
        ] = int(
            retained.sum()
        )

        row[
            f"top{cutoff}_fraction"
        ] = retained.mean()

    finite = aligned.dropna()

    row[
        "median_truth_rank"
    ] = (
        finite.median()
        if not finite.empty
        else np.nan
    )

    return row


def main() -> None:
    for path in [
        RECOVERABILITY_PATH,
        TEST_PATH,
        TRAINING_STRUCTURES_PATH,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"Missing required artifact: {path}"
            )

    recoverability = pd.read_parquet(
        RECOVERABILITY_PATH
    ).copy()

    targets = recoverability.loc[
        recoverability[
            "recoverable_by_mass_rules"
        ]
        .fillna(False)
        .astype(bool)
    ].copy()

    test = pd.read_parquet(
        TEST_PATH
    ).copy()

    test[
        "molecule_id"
    ] = (
        test[
            "molecule_id"
        ]
        .astype(str)
    )

    rules = build_signature_rules()

    (
        signature_counts,
        count_histograms,
        signature_totals,
        total_formulas,
    ) = build_training_prior()

    rows = []

    for number, target in enumerate(
        targets.itertuples(
            index=False
        ),
        start=1,
    ):
        molecule_id = str(
            target.molecule_id
        )

        truth_formula = str(
            target.truth_formula
        )

        selected = (
            select_query_spectra(
                test.loc[
                    test[
                        "molecule_id"
                    ]
                    == molecule_id
                ]
            )
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
                    float(
                        ppm
                    )
                )

                current = merged.get(
                    formula
                )

                if (
                    current is None
                    or abs_ppm
                    < current
                ):
                    merged[
                        formula
                    ] = abs_ppm

        for (
            formula,
            abs_ppm,
        ) in merged.items():
            rows.append(
                {
                    "molecule_id":
                        molecule_id,
                    "formula":
                        formula,
                    "abs_ppm":
                        abs_ppm,
                    "log_formula_prior":
                        log_formula_prior(
                            formula,
                            signature_counts,
                            count_histograms,
                            signature_totals,
                            total_formulas,
                        ),
                    "is_truth":
                        formula
                        == truth_formula,
                }
            )

        print(
            f"  {number:>2}/{len(targets)} "
            f"{molecule_id}: "
            f"{len(merged):,} formulas"
        )

    candidates = pd.DataFrame(
        rows
    )

    if candidates.empty:
        raise RuntimeError(
            "Formula enumeration produced no candidates"
        )

    molecule_ids = (
        targets[
            "molecule_id"
        ]
        .astype(str)
        .tolist()
    )

    # Include the mass-only baseline (weight 0) and a broad log-prior
    # sweep. This is diagnostic only; we do not pick a production weight
    # from hidden/Kaggle labels.
    weights = [
        0.0,
        0.05,
        0.10,
        0.20,
        0.35,
        0.50,
        0.75,
        1.0,
        1.5,
        2.0,
    ]

    summaries = []

    rank_columns = {}

    for weight in weights:
        truth_ranks = ranks_for_weight(
            candidates,
            weight,
        )

        summaries.append(
            summarize_ranks(
                truth_ranks,
                molecule_ids,
                weight,
            )
        )

        rank_columns[
            weight
        ] = truth_ranks

    summary = pd.DataFrame(
        summaries
    )

    summary.to_csv(
        SUMMARY_PATH,
        index=False,
    )

    # Choose the diagnostic winner by Top-3, then Top-10, then median
    # truth rank. This is only for deciding whether the prior is useful.
    best = (
        summary.sort_values(
            [
                "top3",
                "top10",
                "median_truth_rank",
                "prior_weight",
            ],
            ascending=[
                False,
                False,
                True,
                True,
            ],
            kind="stable",
        )
        .iloc[
            0
        ]
    )

    best_weight = float(
        best[
            "prior_weight"
        ]
    )

    candidates[
        "best_combined_score"
    ] = (
        -candidates[
            "abs_ppm"
        ]
        + (
            best_weight
            * candidates[
                "log_formula_prior"
            ]
        )
    )

    candidates = (
        candidates.sort_values(
            [
                "molecule_id",
                "best_combined_score",
                "formula",
            ],
            ascending=[
                True,
                False,
                True,
            ],
            kind="stable",
        )
        .copy()
    )

    candidates[
        "best_rank"
    ] = (
        candidates.groupby(
            "molecule_id"
        )
        .cumcount()
        + 1
    )

    candidates[
        "best_prior_weight"
    ] = best_weight

    candidates.to_parquet(
        OUTPUT_PATH,
        index=False,
        compression="zstd",
    )

    print()
    print(
        "Training-derived formula-prior ranking"
    )
    print(
        "=" * 110
    )
    print(
        summary[
            [
                "prior_weight",
                "truth_generated",
                "top3",
                "top3_fraction",
                "top5",
                "top5_fraction",
                "top10",
                "top10_fraction",
                "top25",
                "top25_fraction",
                "top50",
                "top50_fraction",
                "top100",
                "top100_fraction",
                "median_truth_rank",
            ]
        ]
        .to_string(
            index=False,
            formatters={
                "prior_weight":
                    lambda value:
                        f"{value:.2f}",
                "top3_fraction":
                    lambda value:
                        f"{value:.2%}",
                "top5_fraction":
                    lambda value:
                        f"{value:.2%}",
                "top10_fraction":
                    lambda value:
                        f"{value:.2%}",
                "top25_fraction":
                    lambda value:
                        f"{value:.2%}",
                "top50_fraction":
                    lambda value:
                        f"{value:.2%}",
                "top100_fraction":
                    lambda value:
                        f"{value:.2%}",
                "median_truth_rank":
                    lambda value:
                        (
                            "n/a"
                            if pd.isna(
                                value
                            )
                            else f"{value:.1f}"
                        ),
            },
        )
    )

    print()
    print(
        "Best diagnostic prior weight:",
        f"{best_weight:.2f}",
    )
    print(
        "Top-3:",
        (
            f"{int(best['top3'])} / {len(targets)} "
            f"({best['top3_fraction']:.2%})"
        ),
    )
    print(
        "Top-10:",
        (
            f"{int(best['top10'])} / {len(targets)} "
            f"({best['top10_fraction']:.2%})"
        ),
    )
    print(
        "Top-25:",
        (
            f"{int(best['top25'])} / {len(targets)} "
            f"({best['top25_fraction']:.2%})"
        ),
    )
    print()
    print(
        "Wrote:",
        OUTPUT_PATH,
    )
    print(
        "Wrote:",
        SUMMARY_PATH,
    )


if __name__ == "__main__":
    main()
