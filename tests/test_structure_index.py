import numpy as np
import pytest

from casmi26.chemistry.structure_index import (
    build_structure_index,
    structures_for_formulas,
)


def test_build_structure_index() -> None:
    index = build_structure_index(
        inchikey14=[
            "STRUCTURE-B",
            "STRUCTURE-A",
            "STRUCTURE-C",
        ],
        smiles=[
            "CC",
            "C",
            "CCC",
        ],
        formulas=[
            "C2H6",
            "CH4",
            "C3H8",
        ],
        masses=[
            30.046950,
            16.031300,
            44.062600,
        ],
    )

    assert len(index) == 3

    assert index.inchikey14.tolist() == [
        "STRUCTURE-B",
        "STRUCTURE-C",
        "STRUCTURE-A",
    ]

    assert index.formulas.tolist() == [
        "C2H6",
        "C3H8",
        "CH4",
    ]


def test_build_structure_index_deduplicates() -> None:
    index = build_structure_index(
        inchikey14=[
            "STRUCTURE-A",
            "STRUCTURE-A",
        ],
        smiles=[
            "CCO",
            "CCO",
        ],
        formulas=[
            "C2H6O",
            "C2H6O",
        ],
        masses=[
            46.041865,
            46.041865,
        ],
    )

    assert len(index) == 1
    assert index.inchikey14[0] == "STRUCTURE-A"


def test_build_structure_index_rejects_inconsistent_metadata() -> None:
    with pytest.raises(
        ValueError,
        match="Inconsistent metadata",
    ):
        build_structure_index(
            inchikey14=[
                "STRUCTURE-A",
                "STRUCTURE-A",
            ],
            smiles=[
                "CCO",
                "COC",
            ],
            formulas=[
                "C2H6O",
                "C2H6O",
            ],
            masses=[
                46.041865,
                46.041865,
            ],
        )


def test_build_structure_index_rejects_unequal_lengths() -> None:
    with pytest.raises(
        ValueError,
        match="equal length",
    ):
        build_structure_index(
            inchikey14=[
                "STRUCTURE-A",
                "STRUCTURE-B",
            ],
            smiles=[
                "CCO",
            ],
            formulas=[
                "C2H6O",
                "CH4",
            ],
            masses=[
                46.041865,
                16.031300,
            ],
        )


def test_structures_for_one_formula() -> None:
    index = build_structure_index(
        inchikey14=[
            "A",
            "B",
            "C",
        ],
        smiles=[
            "CCO",
            "COC",
            "CCC",
        ],
        formulas=[
            "C2H6O",
            "C2H6O",
            "C3H8",
        ],
        masses=[
            46.041865,
            46.041865,
            44.062600,
        ],
    )

    result = structures_for_formulas(
        index,
        {"C2H6O"},
    )

    assert index.inchikey14[
        result
    ].tolist() == [
        "A",
        "B",
    ]


def test_structures_for_multiple_formulas() -> None:
    index = build_structure_index(
        inchikey14=[
            "A",
            "B",
            "C",
        ],
        smiles=[
            "CCO",
            "COC",
            "CCC",
        ],
        formulas=[
            "C2H6O",
            "C2H6O",
            "C3H8",
        ],
        masses=[
            46.041865,
            46.041865,
            44.062600,
        ],
    )

    result = structures_for_formulas(
        index,
        {
            "C2H6O",
            "C3H8",
        },
    )

    assert set(
        index.inchikey14[
            result
        ].tolist()
    ) == {
        "A",
        "B",
        "C",
    }


def test_structures_for_unknown_formula() -> None:
    index = build_structure_index(
        inchikey14=["A"],
        smiles=["CCO"],
        formulas=["C2H6O"],
        masses=[46.041865],
    )

    result = structures_for_formulas(
        index,
        {"C99H99"},
    )

    assert np.array_equal(
        result,
        np.empty(
            0,
            dtype=np.int64,
        ),
    )


def test_structures_for_empty_formula_set() -> None:
    index = build_structure_index(
        inchikey14=["A"],
        smiles=["CCO"],
        formulas=["C2H6O"],
        masses=[46.041865],
    )

    result = structures_for_formulas(
        index,
        set(),
    )

    assert len(result) == 0