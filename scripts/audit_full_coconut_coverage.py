from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


FULL_PATH = Path(
    "data/external/coconut/coconut_full.parquet"
)

APPROVED_PATH = Path(
    "data/external/coconut/coconut_approved.parquet"
)

TRAINING_STRUCTURES_PATH = Path(
    "data/processed/structure_index/training_structures.parquet"
)

VALIDATION_PATH = Path(
    "data/processed/splits/validation_structures.parquet"
)

AUDIT_DIR = Path(
    "data/processed/diagnostics/large_validation_audit"
)

MANIFEST_PATH = (
    AUDIT_DIR
    / "manifest.parquet"
)

HYPOTHESES_PATH = (
    AUDIT_DIR
    / "formula_hypotheses.parquet"
)

OUTPUT_PATH = (
    AUDIT_DIR
    / "full_coconut_coverage.csv"
)

FORMULA_OUTPUT_PATH = (
    AUDIT_DIR
    / "full_coconut_formula_burden.csv"
)


def pct(
    numerator: int,
    denominator: int,
) -> str:
    if denominator == 0:
        return "n/a"

    return (
        f"{numerator / denominator:.2%}"
    )


def unique_keys(
    frame: pd.DataFrame,
) -> set[str]:
    return set(
        frame[
            "inchikey14"
        ]
        .dropna()
        .astype(str)
    )


def summarize_catalog(
    label: str,
    catalog: pd.DataFrame,
    training_keys: set[str],
    validation_keys: set[str],
    audit_keys: set[str],
) -> dict[str, object]:
    keys = unique_keys(
        catalog
    )

    validation_overlap = (
        validation_keys
        & keys
    )

    audit_overlap = (
        audit_keys
        & keys
    )

    external_validation = (
        validation_overlap
        - training_keys
    )

    external_audit = (
        audit_overlap
        - training_keys
    )

    return {
        "catalog":
            label,
        "structures":
            len(
                keys
            ),
        "formulas":
            catalog[
                "molecular_formula"
            ]
            .dropna()
            .astype(str)
            .nunique(),
        "validation_truths":
            len(
                validation_overlap
            ),
        "validation_truth_pct":
            (
                len(
                    validation_overlap
                )
                / len(
                    validation_keys
                )
                if validation_keys
                else np.nan
            ),
        "validation_external_truths":
            len(
                external_validation
            ),
        "validation_external_truth_pct":
            (
                len(
                    external_validation
                )
                / len(
                    validation_keys
                )
                if validation_keys
                else np.nan
            ),
        "audit_truths":
            len(
                audit_overlap
            ),
        "audit_truth_pct":
            (
                len(
                    audit_overlap
                )
                / len(
                    audit_keys
                )
                if audit_keys
                else np.nan
            ),
        "audit_external_truths":
            len(
                external_audit
            ),
        "audit_external_truth_pct":
            (
                len(
                    external_audit
                )
                / len(
                    audit_keys
                )
                if audit_keys
                else np.nan
            ),
    }


def main() -> None:
    for required in [
        FULL_PATH,
        APPROVED_PATH,
        TRAINING_STRUCTURES_PATH,
        VALIDATION_PATH,
        MANIFEST_PATH,
        HYPOTHESES_PATH,
    ]:
        if not required.exists():
            raise FileNotFoundError(
                f"Missing required artifact: {required}"
            )

    full = pd.read_parquet(
        FULL_PATH
    )

    approved = pd.read_parquet(
        APPROVED_PATH
    )

    training = pd.read_parquet(
        TRAINING_STRUCTURES_PATH,
        columns=[
            "inchikey14",
        ],
    )

    validation = pd.read_parquet(
        VALIDATION_PATH,
        columns=[
            "inchikey14",
        ],
    )

    manifest = pd.read_parquet(
        MANIFEST_PATH,
        columns=[
            "inchikey14",
        ],
    )

    hypotheses = pd.read_parquet(
        HYPOTHESES_PATH
    )

    required_hypothesis_columns = {
        "inchikey14",
        "candidate_formula",
        "is_truth_formula",
    }

    missing = (
        required_hypothesis_columns
        - set(
            hypotheses.columns
        )
    )

    if missing:
        raise ValueError(
            "Formula hypotheses missing columns: "
            f"{sorted(missing)}"
        )

    training_keys = unique_keys(
        training
    )

    validation_keys = unique_keys(
        validation
    )

    audit_keys = unique_keys(
        manifest
    )

    summary = pd.DataFrame(
        [
            summarize_catalog(
                "approved",
                approved,
                training_keys,
                validation_keys,
                audit_keys,
            ),
            summarize_catalog(
                "full",
                full,
                training_keys,
                validation_keys,
                audit_keys,
            ),
        ]
    )

    summary.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print()
    print(
        "COCONUT catalog coverage"
    )
    print(
        "=" * 78
    )

    display_columns = [
        "catalog",
        "structures",
        "formulas",
        "validation_truths",
        "validation_truth_pct",
        "validation_external_truths",
        "validation_external_truth_pct",
        "audit_truths",
        "audit_truth_pct",
        "audit_external_truths",
        "audit_external_truth_pct",
    ]

    print(
        summary[
            display_columns
        ].to_string(
            index=False,
            formatters={
                "validation_truth_pct":
                    lambda value:
                    f"{value:.2%}",
                "validation_external_truth_pct":
                    lambda value:
                    f"{value:.2%}",
                "audit_truth_pct":
                    lambda value:
                    f"{value:.2%}",
                "audit_external_truth_pct":
                    lambda value:
                    f"{value:.2%}",
            },
        )
    )

    approved_keys = unique_keys(
        approved
    )

    full_keys = unique_keys(
        full
    )

    newly_available = (
        full_keys
        - approved_keys
    )

    newly_covered_validation = (
        validation_keys
        & newly_available
    )

    newly_covered_audit = (
        audit_keys
        & newly_available
    )

    print()
    print(
        "Gain from full COCONUT"
    )
    print(
        "-" * 78
    )
    print(
        "New structures:",
        f"{len(newly_available):,}",
    )
    print(
        "New validation truths:",
        f"{len(newly_covered_validation):,} "
        f"({pct(len(newly_covered_validation), len(validation_keys))})",
    )
    print(
        "New 10k-audit truths:",
        f"{len(newly_covered_audit):,} "
        f"({pct(len(newly_covered_audit), len(audit_keys))})",
    )

    #
    # Candidate burden under the already-frozen formula hypotheses.
    #
    # This deliberately does not rebuild formula hypotheses yet. It
    # isolates the effect of expanding the COCONUT structure catalog.
    #
    requested_formulas = set(
        hypotheses[
            "candidate_formula"
        ]
        .dropna()
        .astype(str)
    )

    def formula_counts(
        catalog: pd.DataFrame,
        label: str,
    ) -> pd.DataFrame:
        frame = catalog.loc[
            catalog[
                "molecular_formula"
            ]
            .dropna()
            .astype(str)
            .isin(
                requested_formulas
            ),
            [
                "molecular_formula",
                "inchikey14",
            ],
        ].copy()

        frame[
            "molecular_formula"
        ] = (
            frame[
                "molecular_formula"
            ]
            .astype(str)
        )

        frame[
            "inchikey14"
        ] = (
            frame[
                "inchikey14"
            ]
            .astype(str)
        )

        counts = (
            frame
            .drop_duplicates(
                subset=[
                    "molecular_formula",
                    "inchikey14",
                ]
            )
            .groupby(
                "molecular_formula"
            )[
                "inchikey14"
            ]
            .nunique()
            .rename(
                f"{label}_candidate_count"
            )
            .reset_index()
        )

        return counts

    approved_counts = formula_counts(
        approved,
        "approved",
    )

    full_counts = formula_counts(
        full,
        "full",
    )

    formula_burden = (
        pd.DataFrame(
            {
                "molecular_formula":
                    sorted(
                        requested_formulas
                    )
            }
        )
        .merge(
            approved_counts,
            on="molecular_formula",
            how="left",
        )
        .merge(
            full_counts,
            on="molecular_formula",
            how="left",
        )
    )

    for column in [
        "approved_candidate_count",
        "full_candidate_count",
    ]:
        formula_burden[
            column
        ] = (
            formula_burden[
                column
            ]
            .fillna(0)
            .astype(np.int64)
        )

    formula_burden[
        "additional_candidates"
    ] = (
        formula_burden[
            "full_candidate_count"
        ]
        - formula_burden[
            "approved_candidate_count"
        ]
    )

    formula_burden.to_csv(
        FORMULA_OUTPUT_PATH,
        index=False,
    )

    hypothesis_counts = (
        hypotheses[
            [
                "inchikey14",
                "candidate_formula",
            ]
        ]
        .drop_duplicates()
        .merge(
            formula_burden,
            left_on="candidate_formula",
            right_on="molecular_formula",
            how="left",
            validate="many_to_one",
        )
    )

    per_query = (
        hypothesis_counts.groupby(
            "inchikey14"
        )[
            [
                "approved_candidate_count",
                "full_candidate_count",
            ]
        ]
        .sum()
    )

    print()
    print(
        "Candidate burden under existing formula hypotheses"
    )
    print(
        "-" * 78
    )

    for label, column in [
        (
            "Approved COCONUT",
            "approved_candidate_count",
        ),
        (
            "Full COCONUT",
            "full_candidate_count",
        ),
    ]:
        values = per_query[
            column
        ]

        print()
        print(
            label
        )
        print(
            values.describe(
                percentiles=[
                    0.5,
                    0.9,
                    0.95,
                    0.99,
                ]
            ).to_string()
        )

    #
    # On the frozen 10k audit, determine whether the truth structure is
    # present in COCONUT under any generated formula hypothesis.
    #
    query_formulas = (
        hypotheses[
            [
                "inchikey14",
                "candidate_formula",
            ]
        ]
        .drop_duplicates()
    )

    def hypothesis_truth_coverage(
        catalog: pd.DataFrame,
    ) -> set[str]:
        candidate_pairs = (
            catalog.loc[
                catalog[
                    "molecular_formula"
                ]
                .dropna()
                .astype(str)
                .isin(
                    requested_formulas
                ),
                [
                    "inchikey14",
                    "molecular_formula",
                ],
            ]
            .drop_duplicates()
            .rename(
                columns={
                    "inchikey14":
                        "candidate_inchikey14",
                }
            )
        )

        truth_pairs = (
            query_formulas.merge(
                candidate_pairs,
                left_on=[
                    "inchikey14",
                    "candidate_formula",
                ],
                right_on=[
                    "candidate_inchikey14",
                    "molecular_formula",
                ],
                how="inner",
            )
        )

        return set(
            truth_pairs[
                "inchikey14"
            ]
            .astype(str)
        )

    approved_hypothesis_truths = (
        hypothesis_truth_coverage(
            approved
        )
    )

    full_hypothesis_truths = (
        hypothesis_truth_coverage(
            full
        )
    )

    print()
    print(
        "10k audit truth structure under generated formula hypotheses"
    )
    print(
        "-" * 78
    )
    print(
        "Approved COCONUT:",
        f"{len(approved_hypothesis_truths):,}/"
        f"{len(audit_keys):,} "
        f"({pct(len(approved_hypothesis_truths), len(audit_keys))})",
    )
    print(
        "Full COCONUT:",
        f"{len(full_hypothesis_truths):,}/"
        f"{len(audit_keys):,} "
        f"({pct(len(full_hypothesis_truths), len(audit_keys))})",
    )
    print(
        "Additional truths recovered:",
        f"{len(full_hypothesis_truths - approved_hypothesis_truths):,}",
    )

    print()
    print(
        "Wrote:",
        OUTPUT_PATH,
    )
    print(
        "Wrote:",
        FORMULA_OUTPUT_PATH,
    )


if __name__ == "__main__":
    main()
