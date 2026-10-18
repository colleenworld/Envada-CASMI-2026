from __future__ import annotations

import pandas as pd


def resolve_structure_metadata(
    frame: pd.DataFrame,
    *,
    smiles_column: str,
) -> pd.DataFrame:
    """
    Resolve one formula/SMILES representation per InChIKey14 by choosing
    the most frequently observed pair.
    """
    required = {
        "inchikey14",
        smiles_column,
        "molecular_formula",
    }

    missing = required - set(frame.columns)

    if missing:
        raise ValueError(
            "Structure metadata is missing required columns: "
            f"{sorted(missing)}"
        )

    clean = frame[
        [
            "inchikey14",
            smiles_column,
            "molecular_formula",
        ]
    ].dropna().copy()

    clean["inchikey14"] = (
        clean["inchikey14"]
        .astype(str)
    )
    clean[smiles_column] = (
        clean[smiles_column]
        .astype(str)
    )
    clean["molecular_formula"] = (
        clean["molecular_formula"]
        .astype(str)
    )

    counts = (
        clean.groupby(
            [
                "inchikey14",
                smiles_column,
                "molecular_formula",
            ],
            as_index=False,
        )
        .size()
        .sort_values(
            [
                "inchikey14",
                "size",
                "molecular_formula",
                smiles_column,
            ],
            ascending=[
                True,
                False,
                True,
                True,
            ],
        )
    )

    return (
        counts
        .drop_duplicates(
            subset=["inchikey14"],
            keep="first",
        )
        .drop(columns=["size"])
        .reset_index(drop=True)
    )


def build_allowed_structure_catalog(
    training: pd.DataFrame,
    coconut: pd.DataFrame,
) -> tuple[pd.DataFrame, set[str]]:
    """
    Build a strict-validation candidate catalog from training-split
    structures plus external COCONUT structures.

    Structures where the two sources disagree about molecular formula are
    excluded, matching the production candidate-builder policy.
    """
    training_resolved = resolve_structure_metadata(
        training,
        smiles_column="normalized_smiles",
    )

    coconut_resolved = resolve_structure_metadata(
        coconut.rename(
            columns={
                "smiles": "normalized_smiles",
            }
        ),
        smiles_column="normalized_smiles",
    )

    overlap = training_resolved[
        [
            "inchikey14",
            "molecular_formula",
        ]
    ].merge(
        coconut_resolved[
            [
                "inchikey14",
                "molecular_formula",
            ]
        ],
        on="inchikey14",
        how="inner",
        suffixes=(
            "_training",
            "_coconut",
        ),
    )

    conflicts = set(
        overlap.loc[
            overlap[
                "molecular_formula_training"
            ]
            != overlap[
                "molecular_formula_coconut"
            ],
            "inchikey14",
        ].astype(str)
    )

    training_resolved = (
        training_resolved.loc[
            ~training_resolved[
                "inchikey14"
            ].isin(conflicts)
        ]
        .copy()
    )

    coconut_resolved = (
        coconut_resolved.loc[
            ~coconut_resolved[
                "inchikey14"
            ].isin(conflicts)
        ]
        .copy()
    )

    training_resolved[
        "in_training"
    ] = True
    training_resolved[
        "in_coconut"
    ] = False

    coconut_resolved[
        "in_training"
    ] = False
    coconut_resolved[
        "in_coconut"
    ] = True

    combined = pd.concat(
        [
            training_resolved,
            coconut_resolved,
        ],
        ignore_index=True,
    )

    catalog = (
        combined
        .sort_values(
            [
                "inchikey14",
                "in_training",
            ],
            ascending=[
                True,
                False,
            ],
        )
        .drop_duplicates(
            subset=["inchikey14"],
            keep="first",
        )
        .reset_index(drop=True)
    )

    return catalog, conflicts
