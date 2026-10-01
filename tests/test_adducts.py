import pytest

from casmi26.chemistry.adducts import (
    ADDUCTS,
    neutral_mass,
)


@pytest.mark.parametrize(
    ("adduct", "neutral", "expected_precursor"),
    [
        (
            "[M+H]+",
            300.0,
            301.007276466621,
        ),
        (
            "[M-H]-",
            300.0,
            298.992723533379,
        ),
        (
            "[M+Na]+",
            300.0,
            322.989218,
        ),
        (
            "[M+K]+",
            300.0,
            338.963158,
        ),
        (
            "[M+NH4]+",
            300.0,
            318.033823,
        ),
    ],
)
def test_precursor_conversion(
    adduct,
    neutral,
    expected_precursor,
):
    definition = ADDUCTS[adduct]

    assert definition.precursor_mz(
        neutral
    ) == pytest.approx(
        expected_precursor,
        abs=1e-9,
    )


@pytest.mark.parametrize(
    "adduct",
    list(ADDUCTS),
)
def test_round_trip(adduct):
    definition = ADDUCTS[adduct]

    neutral = 500.123456

    precursor = definition.precursor_mz(
        neutral
    )

    recovered = definition.neutral_mass(
        precursor
    )

    assert recovered == pytest.approx(
        neutral,
        abs=1e-9,
    )


def test_unknown_adduct_returns_none():
    assert neutral_mass(
        300.0,
        "[something-we-dont-support]",
    ) is None