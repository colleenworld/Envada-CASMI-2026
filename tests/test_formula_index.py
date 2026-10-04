import numpy as np
import pytest

from casmi26.chemistry.formula_index import (
    build_formula_index,
    search_formula_index,
)


def test_build_formula_index_sorts_by_mass() -> None:
    index = build_formula_index(
        ["B", "A", "C"],
        [200.0, 100.0, 300.0],
    )

    assert index.formulas.tolist() == [
        "A",
        "B",
        "C",
    ]

    assert index.masses.tolist() == [
        100.0,
        200.0,
        300.0,
    ]


def test_build_formula_index_deduplicates() -> None:
    index = build_formula_index(
        ["A", "A", "B"],
        [100.0, 100.0, 200.0],
    )

    assert index.formulas.tolist() == [
        "A",
        "B",
    ]


def test_build_formula_index_rejects_inconsistent_mass() -> None:
    with pytest.raises(
        ValueError,
        match="Inconsistent mass",
    ):
        build_formula_index(
            ["A", "A"],
            [100.0, 101.0],
        )


def test_search_formula_index() -> None:
    index = build_formula_index(
        ["A", "B", "C"],
        [
            99.999,
            100.0005,
            100.01,
        ],
    )

    result = search_formula_index(
        index,
        neutral_mass=100.0,
        tolerance_ppm=10.0,
    )

    assert np.array_equal(
        result,
        np.array([0, 1]),
    )


def test_search_formula_index_zero_matches() -> None:
    index = build_formula_index(
        ["A"],
        [200.0],
    )

    result = search_formula_index(
        index,
        neutral_mass=100.0,
        tolerance_ppm=10.0,
    )

    assert len(result) == 0


def test_search_formula_index_rejects_negative_tolerance() -> None:
    index = build_formula_index(
        ["A"],
        [100.0],
    )

    with pytest.raises(
        ValueError,
        match="non-negative",
    ):
        search_formula_index(
            index,
            neutral_mass=100.0,
            tolerance_ppm=-1.0,
        )