from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


TRAINING_PATH = Path(
    "data/processed/structure_index/training_structures.parquet"
)

COCONUT_PATH = Path(
    "data/external/coconut/coconut_approved.parquet"
)

VALIDATION_PATH = Path(
    "data/processed/splits/validation_structures.parquet"
)


def describe(values: list[int]) -> str:
    if not values:
        return "no values"

    a = np.asarray(values)

    return (
        f"median={np.median(a):.0f}, "
        f"p90={np.percentile(a, 90):.0f}, "
        f"p95={np.percentile(a, 95):.0f}, "
        f"p99={np.percentile(a, 99):.0f}, "
        f"max={np.max(a):.0f}"
    )


def main() -> None:
    training = pd.read_parquet(TRAINING_PATH)
    coconut = pd.read_parquet(COCONUT_PATH)
    validation = pd.read_parquet(VALIDATION_PATH)

    validation_keys = set(
        validation["inchikey14"]
        .dropna()
        .astype(str)
    )

    training_keys = set(
        training["inchikey14"]
        .dropna()
        .astype(str)
    )

    coconut_keys = set(
        coconut["inchikey14"]
        .dropna()
        .astype(str)
    )

    print("Catalogs")
    print("=" * 70)
    print(
        f"Training structures:     {len(training_keys):,}"
    )
    print(
        f"COCONUT structures:      {len(coconut_keys):,}"
    )
    print(
        f"Validation structures:   {len(validation_keys):,}"
    )

    training_validation_overlap = (
        training_keys & validation_keys
    )

    if training_validation_overlap:
        raise RuntimeError(
            "Training catalog contains validation structures"
        )

    coconut_truth = (
        coconut_keys & validation_keys
    )

    print()
    print("External truth availability")
    print("=" * 70)

    print(
        f"Validation truths in COCONUT: "
        f"{len(coconut_truth):,} / "
        f"{len(validation_keys):,} "
        f"({len(coconut_truth) / len(validation_keys):.3%})"
    )

    #
    # Build formula -> structure sets.
    #
    training_by_formula = (
        training.groupby("molecular_formula")[
            "inchikey14"
        ]
        .agg(lambda x: set(x.astype(str)))
        .to_dict()
    )

    coconut_by_formula = (
        coconut.groupby("molecular_formula")[
            "inchikey14"
        ]
        .agg(lambda x: set(x.astype(str)))
        .to_dict()
    )

    formulas = (
        set(training_by_formula)
        | set(coconut_by_formula)
    )

    combined_by_formula: dict[str, set[str]] = {}

    for formula in formulas:
        combined_by_formula[formula] = (
            training_by_formula.get(
                formula,
                set(),
            )
            | coconut_by_formula.get(
                formula,
                set(),
            )
        )

    #
    # For this first experiment, evaluate only validation truths
    # actually present in approved COCONUT.
    #
    # We need their formula from COCONUT itself. This isolates the
    # structure-catalog question from formula inference.
    #
    coconut_truth_rows = (
        coconut[
            coconut["inchikey14"].isin(
                coconut_truth
            )
        ]
        .drop_duplicates(
            subset=["inchikey14"]
        )
    )

    training_candidate_counts: list[int] = []
    coconut_candidate_counts: list[int] = []
    combined_candidate_counts: list[int] = []

    truth_in_training_candidates = 0
    truth_in_coconut_candidates = 0
    truth_in_combined_candidates = 0

    for row in coconut_truth_rows.itertuples(
        index=False
    ):
        truth = str(row.inchikey14)
        formula = str(row.molecular_formula)

        training_candidates = (
            training_by_formula.get(
                formula,
                set(),
            )
        )

        coconut_candidates = (
            coconut_by_formula.get(
                formula,
                set(),
            )
        )

        combined_candidates = (
            combined_by_formula.get(
                formula,
                set(),
            )
        )

        training_candidate_counts.append(
            len(training_candidates)
        )

        coconut_candidate_counts.append(
            len(coconut_candidates)
        )

        combined_candidate_counts.append(
            len(combined_candidates)
        )

        if truth in training_candidates:
            truth_in_training_candidates += 1

        if truth in coconut_candidates:
            truth_in_coconut_candidates += 1

        if truth in combined_candidates:
            truth_in_combined_candidates += 1

    evaluated = len(coconut_truth_rows)

    print()
    print("Oracle-formula candidate experiment")
    print("=" * 70)

    print(
        f"Externally recoverable truths evaluated: "
        f"{evaluated:,}"
    )

    print()
    print("Truth candidate recall:")

    print(
        f"  Training only: "
        f"{truth_in_training_candidates:,} / "
        f"{evaluated:,} "
        f"({truth_in_training_candidates / evaluated:.3%})"
    )

    print(
        f"  COCONUT only:  "
        f"{truth_in_coconut_candidates:,} / "
        f"{evaluated:,} "
        f"({truth_in_coconut_candidates / evaluated:.3%})"
    )

    print(
        f"  Combined:      "
        f"{truth_in_combined_candidates:,} / "
        f"{evaluated:,} "
        f"({truth_in_combined_candidates / evaluated:.3%})"
    )

    print()
    print("Candidate counts per query:")

    print(
        "  Training only: "
        + describe(training_candidate_counts)
    )

    print(
        "  COCONUT only:  "
        + describe(coconut_candidate_counts)
    )

    print(
        "  Combined:      "
        + describe(combined_candidate_counts)
    )

    #
    # How many COCONUT candidates are genuinely new relative
    # to training for these formulas?
    #
    novel_counts: list[int] = []

    for row in coconut_truth_rows.itertuples(
        index=False
    ):
        formula = str(row.molecular_formula)

        novel = (
            coconut_by_formula.get(
                formula,
                set(),
            )
            - training_by_formula.get(
                formula,
                set(),
            )
        )

        novel_counts.append(len(novel))

    print()
    print(
        "Novel COCONUT structures added per query formula:"
    )
    print(
        "  " + describe(novel_counts)
    )


if __name__ == "__main__":
    main()