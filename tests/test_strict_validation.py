import pandas as pd

from casmi26.evaluation.strict_validation import (
    build_allowed_structure_catalog,
    resolve_structure_metadata,
)


def test_resolve_structure_metadata_prefers_most_common_pair():
    frame = pd.DataFrame(
        {
            "inchikey14": [
                "A",
                "A",
                "A",
                "B",
            ],
            "normalized_smiles": [
                "CCO",
                "CCO",
                "CC",
                "CCC",
            ],
            "molecular_formula": [
                "C2H6O",
                "C2H6O",
                "C2H6",
                "C3H8",
            ],
        }
    )

    result = resolve_structure_metadata(
        frame,
        smiles_column="normalized_smiles",
    )

    a = result.loc[
        result[
            "inchikey14"
        ]
        == "A"
    ].iloc[0]

    assert a[
        "normalized_smiles"
    ] == "CCO"

    assert a[
        "molecular_formula"
    ] == "C2H6O"


def test_allowed_catalog_uses_training_plus_coconut():
    training = pd.DataFrame(
        {
            "inchikey14": [
                "TRAIN",
            ],
            "normalized_smiles": [
                "CC",
            ],
            "molecular_formula": [
                "C2H6",
            ],
        }
    )

    coconut = pd.DataFrame(
        {
            "inchikey14": [
                "EXTERNAL",
            ],
            "smiles": [
                "CCO",
            ],
            "molecular_formula": [
                "C2H6O",
            ],
        }
    )

    catalog, conflicts = (
        build_allowed_structure_catalog(
            training,
            coconut,
        )
    )

    assert set(
        catalog[
            "inchikey14"
        ]
    ) == {
        "TRAIN",
        "EXTERNAL",
    }

    assert conflicts == set()


def test_conflicting_structure_is_excluded():
    training = pd.DataFrame(
        {
            "inchikey14": [
                "CONFLICT",
            ],
            "normalized_smiles": [
                "CC",
            ],
            "molecular_formula": [
                "C2H6",
            ],
        }
    )

    coconut = pd.DataFrame(
        {
            "inchikey14": [
                "CONFLICT",
            ],
            "smiles": [
                "CC",
            ],
            "molecular_formula": [
                "C2H4",
            ],
        }
    )

    catalog, conflicts = (
        build_allowed_structure_catalog(
            training,
            coconut,
        )
    )

    assert catalog.empty
    assert conflicts == {
        "CONFLICT",
    }
