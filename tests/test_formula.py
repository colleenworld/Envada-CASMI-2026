import pytest

from casmi26.chemistry.formula import (
    mass_error_da,
    mass_error_ppm,
    monoisotopic_mass,
    parse_formula,
)


def test_parse_formula() -> None:
    assert parse_formula(
        "C20H30O2"
    ) == {
        "C": 20,
        "H": 30,
        "O": 2,
    }


def test_parse_formula_implicit_counts() -> None:
    assert parse_formula(
        "CH4O"
    ) == {
        "C": 1,
        "H": 4,
        "O": 1,
    }


def test_parse_formula_two_letter_elements() -> None:
    assert parse_formula(
        "C2H5Cl"
    ) == {
        "C": 2,
        "H": 5,
        "Cl": 1,
    }


def test_repeated_elements_are_combined() -> None:
    assert parse_formula(
        "CH3CH2OH"
    ) == {
        "C": 2,
        "H": 6,
        "O": 1,
    }


def test_rejects_unsupported_element() -> None:
    with pytest.raises(
        ValueError,
        match="Unsupported element",
    ):
        parse_formula(
            "C2H5Xe"
        )


def test_rejects_invalid_formula() -> None:
    with pytest.raises(
        ValueError,
        match="Unsupported molecular formula",
    ):
        parse_formula(
            "C6H12O6+"
        )


def test_monoisotopic_mass_glucose() -> None:
    mass = monoisotopic_mass(
        "C6H12O6"
    )

    assert mass == pytest.approx(
        180.06338810418,
        abs=1e-9,
    )


def test_mass_error_da() -> None:
    assert mass_error_da(
        100.01,
        100.00,
    ) == pytest.approx(
        0.01
    )


def test_mass_error_ppm() -> None:
    assert mass_error_ppm(
        100.001,
        100.0,
    ) == pytest.approx(
        10.0
    )