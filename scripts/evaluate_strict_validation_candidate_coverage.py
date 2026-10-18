from __future__ import annotations

from pathlib import Path

import pandas as pd

from casmi26.evaluation.strict_validation import (
    build_allowed_structure_catalog,
    resolve_structure_metadata,
)


PROJECT_ROOT = (
    Path(__file__).resolve().parent.parent
)

TRAIN_PATH = (
    PROJECT_ROOT
    / "data/raw/train.parquet"
)

TRAIN_STRUCTURES_PATH = (
    PROJECT_ROOT
    / "data/processed/splits/train_structures.parquet"
)

VALIDATION_STRUCTURES_PATH = (
    PROJECT_ROOT
    / "data/processed/splits/validation_structures.parquet"
)

COCONUT_PATH = (
    PROJECT_ROOT
    / "data/external/coconut/coconut_approved.parquet"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data/processed/diagnostics"
)

OUTPUT_PATH = (
    OUTPUT_DIR
    / "strict_validation_candidate_coverage.parquet"
)


def load_keys(
    path: Path,
) -> set[str]:
    frame = pd.read_parquet(
        path,
        columns=["inchikey14"],
    )

    return set(
        frame[
            "inchikey14"
        ].astype(str)
    )


def main() -> None:
    train_keys = load_keys(
        TRAIN_STRUCTURES_PATH
    )
    validation_keys = load_keys(
        VALIDATION_STRUCTURES_PATH
    )

    overlap = (
        train_keys
        & validation_keys
    )

    if overlap:
        raise RuntimeError(
            "Train and validation manifests overlap. "
            f"Examples: {sorted(overlap)[:20]}"
        )

    print(
        "Training structures:",
        f"{len(train_keys):,}",
    )
    print(
        "Validation structures:",
        f"{len(validation_keys):,}",
    )

    metadata = pd.read_parquet(
        TRAIN_PATH,
        columns=[
            "inchikey14",
            "normalized_smiles",
            "molecular_formula",
        ],
    )

    metadata[
        "inchikey14"
    ] = (
        metadata[
            "inchikey14"
        ].astype(str)
    )

    training_rows = (
        metadata.loc[
            metadata[
                "inchikey14"
            ].isin(train_keys)
        ]
        .copy()
    )

    validation_rows = (
        metadata.loc[
            metadata[
                "inchikey14"
            ].isin(validation_keys)
        ]
        .copy()
    )

    observed_validation_keys = set(
        validation_rows[
            "inchikey14"
        ]
    )

    missing_validation = sorted(
        validation_keys
        - observed_validation_keys
    )

    if missing_validation:
        raise RuntimeError(
            "Validation manifest contains structures not found "
            "in train.parquet. "
            f"Examples: {missing_validation[:20]}"
        )

    validation_truth = (
        resolve_structure_metadata(
            validation_rows,
            smiles_column="normalized_smiles",
        )
        .rename(
            columns={
                "normalized_smiles":
                    "truth_smiles",
                "molecular_formula":
                    "truth_formula",
            }
        )
    )

    coconut = pd.read_parquet(
        COCONUT_PATH,
        columns=[
            "inchikey14",
            "smiles",
            "molecular_formula",
        ],
    )

    catalog, conflicts = (
        build_allowed_structure_catalog(
            training_rows,
            coconut,
        )
    )

    leaked_from_training = sorted(
        set(
            catalog.loc[
                catalog[
                    "in_training"
                ],
                "inchikey14",
            ]
        )
        & validation_keys
    )

    if leaked_from_training:
        raise RuntimeError(
            "Validation structures leaked into the training side "
            "of the allowed catalog. "
            f"Examples: {leaked_from_training[:20]}"
        )

    training_formulas = set(
        training_rows[
            "molecular_formula"
        ]
        .dropna()
        .astype(str)
    )

    coconut_formulas = set(
        coconut[
            "molecular_formula"
        ]
        .dropna()
        .astype(str)
    )

    allowed_formulas = (
        training_formulas
        | coconut_formulas
    )

    catalog_for_truth = (
        catalog[
            [
                "inchikey14",
                "molecular_formula",
                "in_training",
                "in_coconut",
            ]
        ]
        .rename(
            columns={
                "molecular_formula":
                    "candidate_catalog_formula",
            }
        )
    )

    diagnostic = (
        validation_truth
        .merge(
            catalog_for_truth,
            on="inchikey14",
            how="left",
            validate="one_to_one",
        )
    )

    diagnostic[
        "truth_formula_in_training"
    ] = (
        diagnostic[
            "truth_formula"
        ].isin(
            training_formulas
        )
    )

    diagnostic[
        "truth_formula_in_coconut"
    ] = (
        diagnostic[
            "truth_formula"
        ].isin(
            coconut_formulas
        )
    )

    diagnostic[
        "truth_formula_in_allowed_index"
    ] = (
        diagnostic[
            "truth_formula"
        ].isin(
            allowed_formulas
        )
    )

    diagnostic[
        "truth_structure_in_allowed_catalog"
    ] = (
        diagnostic[
            "candidate_catalog_formula"
        ].notna()
    )

    diagnostic[
        "truth_structure_formula_matches"
    ] = (
        diagnostic[
            "candidate_catalog_formula"
        ]
        == diagnostic[
            "truth_formula"
        ]
    )

    diagnostic[
        "truth_is_exact_formula_candidate"
    ] = (
        diagnostic[
            "truth_formula_in_allowed_index"
        ]
        & diagnostic[
            "truth_structure_in_allowed_catalog"
        ]
        & diagnostic[
            "truth_structure_formula_matches"
        ]
    )

    diagnostic[
        "truth_in_coconut"
    ] = (
        diagnostic[
            "in_coconut"
        ]
        .fillna(False)
        .astype(bool)
    )

    diagnostic[
        "truth_in_training_catalog"
    ] = (
        diagnostic[
            "in_training"
        ]
        .fillna(False)
        .astype(bool)
    )

    diagnostic[
        "excluded_cross_catalog_conflict"
    ] = (
        diagnostic[
            "inchikey14"
        ].isin(
            conflicts
        )
    )

    diagnostic = diagnostic.sort_values(
        "inchikey14"
    ).reset_index(
        drop=True
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    diagnostic.to_parquet(
        OUTPUT_PATH,
        index=False,
    )

    total = len(
        diagnostic
    )

    def report(
        label: str,
        column: str,
    ) -> None:
        count = int(
            diagnostic[
                column
            ].sum()
        )

        fraction = (
            count / total
            if total
            else 0.0
        )

        print(
            f"{label:<42}"
            f"{count:>8,} / {total:,} "
            f"({fraction:.2%})"
        )

    print()
    print(
        "Strict validation candidate coverage"
    )
    print(
        "===================================="
    )

    report(
        "Truth formula present in training:",
        "truth_formula_in_training",
    )
    report(
        "Truth formula present in COCONUT:",
        "truth_formula_in_coconut",
    )
    report(
        "Truth formula present in allowed index:",
        "truth_formula_in_allowed_index",
    )
    report(
        "Truth structure present in COCONUT:",
        "truth_in_coconut",
    )
    report(
        "Truth structure present in allowed catalog:",
        "truth_structure_in_allowed_catalog",
    )
    report(
        "Truth structure has matching formula:",
        "truth_structure_formula_matches",
    )
    report(
        "Truth is exact-formula candidate:",
        "truth_is_exact_formula_candidate",
    )

    print()
    print(
        "Validation structures on training side:",
        int(
            diagnostic[
                "truth_in_training_catalog"
            ].sum()
        ),
    )

    print(
        "Validation structures excluded by "
        "cross-catalog conflict:",
        int(
            diagnostic[
                "excluded_cross_catalog_conflict"
            ].sum()
        ),
    )

    print()
    print(
        "Allowed catalog structures:",
        f"{len(catalog):,}",
    )

    print(
        "Cross-catalog conflicts:",
        f"{len(conflicts):,}",
    )

    print()
    print(
        f"Wrote {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()
