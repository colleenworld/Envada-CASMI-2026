from __future__ import annotations

import argparse
import hashlib
from collections import Counter, defaultdict, deque
from math import log
from pathlib import Path

import pandas as pd

from casmi26.chemistry.formula import (
    monoisotopic_mass,
    parse_formula,
)
from casmi26.chemistry.formula_enumeration import (
    build_signature_rules,
    enumerate_for_mass,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent

TRAINING_STRUCTURES_PATH = (
    PROJECT_ROOT
    / "data/processed/structure_index/training_structures.parquet"
)

PUBCHEM_FORMULAS_PATH = (
    PROJECT_ROOT
    / "data/processed/pubchem_formula_index/formula_buckets.parquet"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data/processed/formula_index"
)

DEFAULT_OUTPUT_PATH = (
    OUTPUT_DIR
    / "training_prior_formula_universe_top25.parquet"
)

DEFAULT_SUMMARY_PATH = (
    OUTPUT_DIR
    / "training_prior_formula_universe_top25_summary.csv"
)

FIXED_PRIOR_WEIGHT = 0.50
DEFAULT_TOP_K = 25
DEFAULT_MAX_ANCHORS = 20_000
DEFAULT_CHECKPOINTS = [
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a hidden-test-independent formula universe from "
            "training-formula neutral-mass anchors."
        )
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
    )

    parser.add_argument(
        "--max-anchors",
        type=int,
        default=DEFAULT_MAX_ANCHORS,
    )

    parser.add_argument(
        "--mass-bins",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
    )

    parser.add_argument(
        "--summary-output",
        type=Path,
        default=DEFAULT_SUMMARY_PATH,
    )

    args = parser.parse_args()

    if args.top_k <= 0:
        parser.error("--top-k must be positive")

    if args.max_anchors <= 0:
        parser.error("--max-anchors must be positive")

    if args.mass_bins <= 0:
        parser.error("--mass-bins must be positive")

    return args


def signature_for(
    composition: dict[str, int],
) -> tuple[str, ...]:
    return tuple(
        element
        for element in ELEMENTS
        if composition.get(element, 0) > 0
    )


def build_training_prior(
    formulas: list[str],
):
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


def stable_hash(
    value: str,
) -> str:
    return hashlib.sha256(
        value.encode("utf-8")
    ).hexdigest()


def build_anchor_order(
    frame: pd.DataFrame,
    mass_bins: int,
) -> list[str]:
    ordered = (
        frame
        .sort_values(
            [
                "neutral_mass",
                "molecular_formula",
            ],
            kind="stable",
        )
        .reset_index(drop=True)
        .copy()
    )

    bin_count = min(
        mass_bins,
        len(ordered),
    )

    ordered[
        "mass_bin"
    ] = pd.qcut(
        ordered[
            "neutral_mass"
        ].rank(
            method="first"
        ),
        q=bin_count,
        labels=False,
        duplicates="drop",
    )

    buckets: list[
        deque[str]
    ] = []

    for _, group in ordered.groupby(
        "mass_bin",
        sort=True,
    ):
        formulas = sorted(
            group[
                "molecular_formula"
            ].astype(str),
            key=lambda formula: (
                stable_hash(formula),
                formula,
            ),
        )

        buckets.append(
            deque(formulas)
        )

    result = []

    while buckets:
        nonempty = []

        for bucket in buckets:
            if bucket:
                result.append(
                    bucket.popleft()
                )

            if bucket:
                nonempty.append(
                    bucket
                )

        buckets = nonempty

    return result


def main() -> None:
    args = parse_args()

    if not TRAINING_STRUCTURES_PATH.exists():
        raise FileNotFoundError(
            f"Missing training structures: {TRAINING_STRUCTURES_PATH}"
        )

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
        .tolist()
    )

    anchor_rows = []

    invalid_formulas = 0

    for formula in formulas:
        try:
            composition = parse_formula(
                formula
            )

            if (
                "C" not in composition
                or "H" not in composition
            ):
                continue

            mass = monoisotopic_mass(
                formula
            )

        except ValueError:
            invalid_formulas += 1
            continue

        anchor_rows.append(
            {
                "molecular_formula":
                    formula,
                "neutral_mass":
                    float(mass),
            }
        )

    anchors = pd.DataFrame(
        anchor_rows
    )

    if anchors.empty:
        raise RuntimeError(
            "No usable training formula anchors"
        )

    anchor_order = build_anchor_order(
        anchors,
        args.mass_bins,
    )

    max_anchors = min(
        args.max_anchors,
        len(anchor_order),
    )

    anchor_order = anchor_order[
        :max_anchors
    ]

    mass_lookup = dict(
        zip(
            anchors[
                "molecular_formula"
            ].astype(str),
            anchors[
                "neutral_mass"
            ].astype(float),
            strict=True,
        )
    )

    rules = build_signature_rules(
        TRAINING_STRUCTURES_PATH
    )

    (
        signature_counts,
        count_histograms,
        signature_totals,
        total_formulas,
    ) = build_training_prior(
        formulas
    )

    current_pubchem = set()

    if PUBCHEM_FORMULAS_PATH.exists():
        current_pubchem = set(
            pd.read_parquet(
                PUBCHEM_FORMULAS_PATH,
                columns=[
                    "molecular_formula",
                ],
            )[
                "molecular_formula"
            ]
            .dropna()
            .astype(str)
        )

    checkpoint_targets = sorted(
        {
            checkpoint
            for checkpoint in DEFAULT_CHECKPOINTS
            if checkpoint <= max_anchors
        }
        | {
            max_anchors,
        }
    )

    universe: dict[
        str,
        dict,
    ] = {}

    summary_rows = []

    print(
        "Training-derived formula-universe build"
    )
    print("=" * 92)
    print(
        "Unique training formulas:",
        f"{len(formulas):,}",
    )
    print(
        "Usable neutral-mass anchors:",
        f"{len(anchors):,}",
    )
    print(
        "Invalid formulas:",
        f"{invalid_formulas:,}",
    )
    print(
        "Anchors to process:",
        f"{max_anchors:,}",
    )
    print(
        "Top-K per anchor:",
        args.top_k,
    )
    print(
        "Mass bins:",
        args.mass_bins,
    )

    for anchor_index, anchor_formula in enumerate(
        anchor_order,
        start=1,
    ):
        neutral_mass = mass_lookup[
            anchor_formula
        ]

        candidates = enumerate_for_mass(
            neutral_mass,
            rules,
        )

        ranked = []

        for (
            formula,
            formula_mass,
            ppm,
            _,
        ) in candidates:
            abs_ppm = abs(
                float(ppm)
            )

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
                    float(formula_mass),
                    abs_ppm,
                    prior,
                    score,
                )
            )

        ranked.sort(
            key=lambda item: (
                -item[4],
                item[0],
            )
        )

        for rank, (
            formula,
            formula_mass,
            abs_ppm,
            prior,
            score,
        ) in enumerate(
            ranked[
                :args.top_k
            ],
            start=1,
        ):
            existing = universe.get(
                formula
            )

            if existing is None:
                universe[
                    formula
                ] = {
                    "molecular_formula":
                        formula,
                    "monoisotopic_mass":
                        formula_mass,
                    "first_anchor_index":
                        anchor_index,
                    "first_anchor_formula":
                        anchor_formula,
                    "best_anchor_rank":
                        rank,
                    "best_formula_score":
                        score,
                    "best_abs_mass_error_ppm":
                        abs_ppm,
                    "log_formula_prior":
                        prior,
                }

            else:
                if rank < existing[
                    "best_anchor_rank"
                ]:
                    existing[
                        "best_anchor_rank"
                    ] = rank

                if score > existing[
                    "best_formula_score"
                ]:
                    existing[
                        "best_formula_score"
                    ] = score

                if abs_ppm < existing[
                    "best_abs_mass_error_ppm"
                ]:
                    existing[
                        "best_abs_mass_error_ppm"
                    ] = abs_ppm

        if (
            anchor_index % 250 == 0
            or anchor_index
            in checkpoint_targets
        ):
            print(
                f"  anchors {anchor_index:,}/{max_anchors:,} | "
                f"unique formulas {len(universe):,}"
            )

        if anchor_index in checkpoint_targets:
            current_rows = [
                row
                for row in universe.values()
                if row[
                    "first_anchor_index"
                ]
                <= anchor_index
            ]

            current_formula_count = len(
                current_rows
            )

            current_pubchem_count = sum(
                1
                for row in current_rows
                if row[
                    "molecular_formula"
                ]
                in current_pubchem
            )

            summary_rows.append(
                {
                    "anchor_count":
                        anchor_index,
                    "unique_formula_count":
                        current_formula_count,
                    "already_in_pubchem_asset":
                        current_pubchem_count,
                    "missing_from_pubchem_asset":
                        current_formula_count
                        - current_pubchem_count,
                    "pubchem_asset_coverage_fraction":
                        (
                            current_pubchem_count
                            / current_formula_count
                            if current_formula_count
                            else 0.0
                        ),
                }
            )

    universe_frame = (
        pd.DataFrame(
            universe.values()
        )
        .sort_values(
            [
                "first_anchor_index",
                "best_anchor_rank",
                "molecular_formula",
            ],
            kind="stable",
        )
        .reset_index(
            drop=True
        )
    )

    universe_frame[
        "already_in_pubchem_asset"
    ] = (
        universe_frame[
            "molecular_formula"
        ]
        .astype(str)
        .isin(
            current_pubchem
        )
    )

    summary = pd.DataFrame(
        summary_rows
    )

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    universe_frame.to_parquet(
        args.output,
        index=False,
        compression="zstd",
    )

    summary.to_csv(
        args.summary_output,
        index=False,
    )

    print()
    print(
        "Training-derived formula universe"
    )
    print("=" * 92)
    print(
        summary.to_string(
            index=False,
            formatters={
                "pubchem_asset_coverage_fraction":
                    lambda value:
                        f"{value:.2%}",
            },
        )
    )
    print()
    print(
        "Wrote:",
        args.output,
    )
    print(
        "Wrote:",
        args.summary_output,
    )


if __name__ == "__main__":
    main()
