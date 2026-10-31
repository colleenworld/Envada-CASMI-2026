from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent

ROOT = (
    PROJECT_ROOT
    / "data/processed/blind_pseudotest"
)

NOVELTY_PATH = (
    ROOT
    / "novelty_strata.parquet"
)

PUBCHEM_ROOT = (
    PROJECT_ROOT
    / "data/processed/pubchem_formula_index"
)

PUBCHEM_BURDEN_PATH = (
    PUBCHEM_ROOT
    / "burden.parquet"
)

DIRECT_PUBCHEM_PATH = (
    PROJECT_ROOT
    / "data/processed/diagnostics/"
    "pubchem_validation_coverage.parquet"
)

COCONUT_APPROVED_PATH = (
    PROJECT_ROOT
    / "data/external/coconut/coconut_approved.parquet"
)

OUTPUT_PATH = (
    ROOT
    / "missing_candidate_truths.parquet"
)

SUMMARY_PATH = (
    ROOT
    / "missing_candidate_truths_summary.csv"
)


MISSING_STRATA = {
    "pubchem_formula_absent",
    "formula_present_truth_structure_absent",
    "coconut_only",
}


def main() -> None:
    for path in [
        NOVELTY_PATH,
        PUBCHEM_BURDEN_PATH,
        COCONUT_APPROVED_PATH,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"Missing required artifact: {path}"
            )

    novelty = pd.read_parquet(
        NOVELTY_PATH
    ).copy()

    required = {
        "molecule_id",
        "inchikey14",
        "molecular_formula",
        "database_stratum",
        "in_pubchem_asset",
        "pubchem_formula_present",
        "in_approved_coconut",
        "truth_rank",
        "outcome",
    }

    missing = (
        required
        - set(
            novelty.columns
        )
    )

    if missing:
        raise ValueError(
            "Novelty table missing columns: "
            f"{sorted(missing)}"
        )

    missing_truth = novelty.loc[
        novelty[
            "database_stratum"
        ].isin(
            MISSING_STRATA
        )
    ].copy()

    missing_truth[
        "molecule_id"
    ] = (
        missing_truth[
            "molecule_id"
        ]
        .astype(str)
    )

    missing_truth[
        "inchikey14"
    ] = (
        missing_truth[
            "inchikey14"
        ]
        .astype(str)
    )

    missing_truth[
        "molecular_formula"
    ] = (
        missing_truth[
            "molecular_formula"
        ]
        .astype(str)
    )

    burden = pd.read_parquet(
        PUBCHEM_BURDEN_PATH
    ).copy()

    burden[
        "molecular_formula"
    ] = (
        burden[
            "molecular_formula"
        ]
        .astype(str)
    )

    burden_columns = [
        "molecular_formula",
        "candidate_count",
        "hit_max_records",
        "fetch_failed",
    ]

    missing_burden_columns = (
        set(
            burden_columns
        )
        - set(
            burden.columns
        )
    )

    if missing_burden_columns:
        raise ValueError(
            "PubChem burden missing columns: "
            f"{sorted(missing_burden_columns)}"
        )

    burden = (
        burden[
            burden_columns
        ]
        .drop_duplicates(
            subset=[
                "molecular_formula",
            ]
        )
    )

    result = missing_truth.merge(
        burden,
        on="molecular_formula",
        how="left",
        validate="many_to_one",
    )

    result[
        "pubchem_candidate_count"
    ] = (
        pd.to_numeric(
            result[
                "candidate_count"
            ],
            errors="coerce",
        )
    )

    result[
        "formula_hit_import_cap"
    ] = (
        result[
            "hit_max_records"
        ]
        .fillna(False)
        .astype(bool)
    )

    result[
        "formula_fetch_failed"
    ] = (
        result[
            "fetch_failed"
        ]
        .fillna(False)
        .astype(bool)
    )

    result.drop(
        columns=[
            "candidate_count",
            "hit_max_records",
            "fetch_failed",
        ],
        inplace=True,
    )

    direct_available = (
        DIRECT_PUBCHEM_PATH.exists()
    )

    result[
        "direct_pubchem_lookup_available"
    ] = direct_available

    result[
        "truth_found_in_pubchem_direct"
    ] = pd.NA

    result[
        "direct_pubchem_formula_matches"
    ] = pd.NA

    if direct_available:
        direct = pd.read_parquet(
            DIRECT_PUBCHEM_PATH
        ).copy()

        needed = {
            "inchikey14",
            "pubchem_found",
            "pubchem_formula_matches",
        }

        if needed <= set(
            direct.columns
        ):
            direct[
                "inchikey14"
            ] = (
                direct[
                    "inchikey14"
                ]
                .astype(str)
            )

            direct = (
                direct[
                    [
                        "inchikey14",
                        "pubchem_found",
                        "pubchem_formula_matches",
                    ]
                ]
                .drop_duplicates(
                    subset=[
                        "inchikey14",
                    ],
                    keep="first",
                )
                .rename(
                    columns={
                        "pubchem_found":
                            "_direct_pubchem_found",
                        "pubchem_formula_matches":
                            "_direct_pubchem_formula_matches",
                    }
                )
            )

            result = result.merge(
                direct,
                on="inchikey14",
                how="left",
                validate="many_to_one",
            )

            result[
                "truth_found_in_pubchem_direct"
            ] = result[
                "_direct_pubchem_found"
            ]

            result[
                "direct_pubchem_formula_matches"
            ] = result[
                "_direct_pubchem_formula_matches"
            ]

            result.drop(
                columns=[
                    "_direct_pubchem_found",
                    "_direct_pubchem_formula_matches",
                ],
                inplace=True,
            )

    result[
        "asset_failure_type"
    ] = "unknown"

    formula_absent = ~(
        result[
            "pubchem_formula_present"
        ]
        .fillna(False)
        .astype(bool)
    )

    structure_absent = (
        result[
            "pubchem_formula_present"
        ]
        .fillna(False)
        .astype(bool)
        & ~(
            result[
                "in_pubchem_asset"
            ]
            .fillna(False)
            .astype(bool)
        )
    )

    result.loc[
        formula_absent,
        "asset_failure_type",
    ] = (
        "formula_missing_from_offline_asset"
    )

    result.loc[
        structure_absent,
        "asset_failure_type",
    ] = (
        "formula_present_truth_structure_absent"
    )

    result[
        "recovery_class"
    ] = "needs_external_verification"

    # The current bulk-SDF importer is exhaustive over the selected
    # formula set: there is no per-formula record cap and fetch_failed
    # is false for completed bulk imports. Keep explicit guards anyway.
    capped = (
        result[
            "formula_hit_import_cap"
        ]
        .fillna(False)
    )

    failed = (
        result[
            "formula_fetch_failed"
        ]
        .fillna(False)
    )

    result.loc[
        capped,
        "recovery_class",
    ] = (
        "recoverable_by_removing_import_cap"
    )

    result.loc[
        failed,
        "recovery_class",
    ] = (
        "recoverable_by_repairing_failed_import"
    )

    direct_found = (
        result[
            "truth_found_in_pubchem_direct"
        ]
        .fillna(False)
        .astype(bool)
    )

    direct_formula_match = (
        result[
            "direct_pubchem_formula_matches"
        ]
        .fillna(False)
        .astype(bool)
    )

    result.loc[
        formula_absent
        & direct_found
        & direct_formula_match
        & ~capped
        & ~failed,
        "recovery_class",
    ] = (
        "recoverable_by_expanding_formula_asset"
    )

    result.loc[
        structure_absent
        & direct_found
        & direct_formula_match
        & ~capped
        & ~failed,
        "recovery_class",
    ] = (
        "pubchem_asset_identity_mismatch"
    )

    result.loc[
        direct_available
        & ~direct_found
        & ~capped
        & ~failed,
        "recovery_class",
    ] = (
        "not_found_in_pubchem_direct"
    )

    result.loc[
        result[
            "in_approved_coconut"
        ]
        .fillna(False)
        .astype(bool),
        "recovery_class",
    ] = (
        "recoverable_from_coconut"
    )

    result[
        "likely_action"
    ] = (
        result[
            "recovery_class"
        ]
        .map(
            {
                "recoverable_by_expanding_formula_asset":
                    "expand_formula_set_and_rebuild_pubchem_asset",
                "recoverable_by_removing_import_cap":
                    "remove_pubchem_import_cap_and_rebuild",
                "recoverable_by_repairing_failed_import":
                    "repair_pubchem_import_and_rebuild",
                "pubchem_asset_identity_mismatch":
                    "audit_pubchem_key_canonicalization_and_bulk_extraction",
                "recoverable_from_coconut":
                    "ensure_coconut_candidate_is_in_inference_catalog",
                "not_found_in_pubchem_direct":
                    "seek_additional_catalogs_or_de_novo_generation",
                "needs_external_verification":
                    "run_or_complete_direct_pubchem_validation_lookup",
            }
        )
        .fillna(
            "investigate"
        )
    )

    result.to_parquet(
        OUTPUT_PATH,
        index=False,
        compression="zstd",
    )

    summary = (
        result.groupby(
            [
                "recovery_class",
                "likely_action",
            ],
            dropna=False,
        )
        .agg(
            queries=(
                "molecule_id",
                "size",
            ),
            formulas=(
                "molecular_formula",
                "nunique",
            ),
            mean_pubchem_candidates=(
                "pubchem_candidate_count",
                "mean",
            ),
            direct_pubchem_found=(
                "truth_found_in_pubchem_direct",
                lambda values:
                    int(
                        pd.Series(
                            values
                        )
                        .fillna(False)
                        .astype(bool)
                        .sum()
                    ),
            ),
        )
        .reset_index()
        .sort_values(
            [
                "queries",
                "recovery_class",
            ],
            ascending=[
                False,
                True,
            ],
        )
    )

    summary[
        "fraction_of_missing"
    ] = (
        summary[
            "queries"
        ]
        / len(
            result
        )
    )

    summary.to_csv(
        SUMMARY_PATH,
        index=False,
    )

    print()
    print(
        "Missing candidate truth diagnostic"
    )
    print(
        "=" * 100
    )
    print(
        "Missing-candidate queries:",
        f"{len(result):,}",
    )
    print(
        "Unique formulas:",
        f"{result['molecular_formula'].nunique():,}",
    )
    print(
        "Direct PubChem cache available:",
        direct_available,
    )
    print(
        "Import cap hits:",
        int(
            result[
                "formula_hit_import_cap"
            ].sum()
        ),
    )
    print(
        "Import fetch failures:",
        int(
            result[
                "formula_fetch_failed"
            ].sum()
        ),
    )
    print()
    print(
        summary[
            [
                "recovery_class",
                "queries",
                "fraction_of_missing",
                "formulas",
                "mean_pubchem_candidates",
                "direct_pubchem_found",
                "likely_action",
            ]
        ]
        .to_string(
            index=False,
            formatters={
                "fraction_of_missing":
                    lambda value:
                        f"{value:.2%}",
                "mean_pubchem_candidates":
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
        "By original database stratum"
    )
    print(
        "----------------------------"
    )
    print(
        result[
            "database_stratum"
        ]
        .value_counts()
        .to_string()
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

    if not direct_available:
        print()
        print(
            "NOTE: the cached direct PubChem validation lookup is absent. "
            "The diagnostic can still identify offline-asset omissions, "
            "but cannot yet distinguish those from structures genuinely "
            "absent from PubChem."
        )


if __name__ == "__main__":
    main()
