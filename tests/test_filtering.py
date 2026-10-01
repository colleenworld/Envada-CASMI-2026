import numpy as np
import pytest

from casmi26.retrieval.filtering import (
    neutral_mass_mask,
    precursor_adduct_mask,
)

def test_neutral_mass_mask_matches_within_tolerance():
    candidate_masses = np.array(
        [
            499.98,
            499.995,
            500.0,
            500.005,
            500.02,
        ],
        dtype=np.float64,
    )

    result = neutral_mass_mask(
        query_neutral_mass=500.0,
        candidate_neutral_masses=candidate_masses,
        tolerance_da=0.01,
    )

    assert result.tolist() == [
        False,
        True,
        True,
        True,
        False,
    ]


def test_neutral_mass_mask_tolerance_is_inclusive():
    candidate_masses = np.array(
        [
            499.99,
            500.01,
        ],
        dtype=np.float64,
    )

    result = neutral_mass_mask(
        query_neutral_mass=500.0,
        candidate_neutral_masses=candidate_masses,
        tolerance_da=0.01,
    )

    assert result.tolist() == [
        True,
        True,
    ]


def test_neutral_mass_mask_rejects_nonfinite_candidates():
    candidate_masses = np.array(
        [
            np.nan,
            np.inf,
            -np.inf,
            500.0,
        ],
        dtype=np.float64,
    )

    result = neutral_mass_mask(
        query_neutral_mass=500.0,
        candidate_neutral_masses=candidate_masses,
    )

    assert result.tolist() == [
        False,
        False,
        False,
        True,
    ]


def test_neutral_mass_mask_rejects_nonfinite_query():
    candidate_masses = np.array(
        [
            500.0,
            500.001,
        ],
        dtype=np.float64,
    )

    result = neutral_mass_mask(
        query_neutral_mass=np.nan,
        candidate_neutral_masses=candidate_masses,
    )

    assert result.tolist() == [
        False,
        False,
    ]


def test_neutral_mass_mask_rejects_negative_tolerance():
    candidate_masses = np.array(
        [500.0],
        dtype=np.float64,
    )

    with pytest.raises(
        ValueError,
        match="tolerance_da must be non-negative",
    ):
        neutral_mass_mask(
            query_neutral_mass=500.0,
            candidate_neutral_masses=candidate_masses,
            tolerance_da=-0.01,
        )

def test_matches_same_adduct_within_tolerance():
    mask = precursor_adduct_mask(
        query_adduct="[M+H]+",
        query_precursor_mz=300.0,
        candidate_adducts=np.array(
            ["[M+H]+", "[M+H]+", "[M+Na]+"],
        ),
        candidate_precursor_mzs=np.array(
            [300.005, 300.02, 300.005],
        ),
        tolerance_da=0.01,
    )

    assert mask.tolist() == [
        True,
        False,
        False,
    ]


def test_tolerance_is_inclusive():
    mask = precursor_adduct_mask(
        query_adduct="[M+H]+",
        query_precursor_mz=300.0,
        candidate_adducts=np.array(["[M+H]+"]),
        candidate_precursor_mzs=np.array([300.01]),
        tolerance_da=0.01,
    )

    assert mask.tolist() == [True]


def test_non_finite_precursor_does_not_match():
    mask = precursor_adduct_mask(
        query_adduct="[M+H]+",
        query_precursor_mz=300.0,
        candidate_adducts=np.array(
            ["[M+H]+", "[M+H]+"],
        ),
        candidate_precursor_mzs=np.array(
            [np.nan, np.inf],
        ),
    )

    assert not mask.any()


def test_different_adduct_does_not_match():
    mask = precursor_adduct_mask(
        query_adduct="[M+H]+",
        query_precursor_mz=300.0,
        candidate_adducts=np.array(["[M+Na]+"]),
        candidate_precursor_mzs=np.array([300.0]),
    )

    assert mask.tolist() == [False]


def test_rejects_negative_tolerance():
    with pytest.raises(
        ValueError,
        match="non-negative",
    ):
        precursor_adduct_mask(
            query_adduct="[M+H]+",
            query_precursor_mz=300.0,
            candidate_adducts=np.array(["[M+H]+"]),
            candidate_precursor_mzs=np.array([300.0]),
            tolerance_da=-0.01,
        )


def test_rejects_different_lengths():
    with pytest.raises(
        ValueError,
        match="same length",
    ):
        precursor_adduct_mask(
            query_adduct="[M+H]+",
            query_precursor_mz=300.0,
            candidate_adducts=np.array(
                ["[M+H]+", "[M+H]+"],
            ),
            candidate_precursor_mzs=np.array([300.0]),
        )