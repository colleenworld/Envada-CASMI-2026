import numpy as np
import pytest

from casmi26.retrieval.mass_index import (
    build_neutral_mass_index,
    search_neutral_mass,
    search_neutral_masses,
)


def test_builds_sorted_partitions():
    masses = np.array(
        [
            300.0,
            100.0,
            200.0,
            150.0,
        ]
    )

    modes = np.array(
        [
            "positive",
            "positive",
            "positive",
            "negative",
        ]
    )

    index = build_neutral_mass_index(
        masses,
        modes,
    )

    positive = index.partitions[
        "positive"
    ]

    np.testing.assert_array_equal(
        positive.masses,
        [
            100.0,
            200.0,
            300.0,
        ],
    )

    np.testing.assert_array_equal(
        positive.row_ids,
        [
            1,
            2,
            0,
        ],
    )

    negative = index.partitions[
        "negative"
    ]

    np.testing.assert_array_equal(
        negative.masses,
        [
            150.0,
        ],
    )

    np.testing.assert_array_equal(
        negative.row_ids,
        [
            3,
        ],
    )


def test_excludes_non_finite_masses():
    masses = np.array(
        [
            100.0,
            np.nan,
            200.0,
            np.inf,
        ]
    )

    modes = np.array(
        [
            "positive",
            "positive",
            "positive",
            "positive",
        ]
    )

    index = build_neutral_mass_index(
        masses,
        modes,
    )

    partition = index.partitions[
        "positive"
    ]

    np.testing.assert_array_equal(
        partition.masses,
        [
            100.0,
            200.0,
        ],
    )

    np.testing.assert_array_equal(
        partition.row_ids,
        [
            0,
            2,
        ],
    )


def test_searches_within_tolerance():
    masses = np.array(
        [
            99.98,
            99.99,
            100.00,
            100.01,
            100.02,
        ]
    )

    modes = np.array(
        [
            "positive",
            "positive",
            "positive",
            "positive",
            "positive",
        ]
    )

    index = build_neutral_mass_index(
        masses,
        modes,
    )

    result = search_neutral_mass(
        index=index,
        neutral_mass=100.0,
        mode="positive",
        tolerance_da=0.01,
    )

    np.testing.assert_array_equal(
        np.sort(result),
        [
            1,
            2,
            3,
        ],
    )


def test_search_respects_mode():
    masses = np.array(
        [
            100.0,
            100.0,
        ]
    )

    modes = np.array(
        [
            "positive",
            "negative",
        ]
    )

    index = build_neutral_mass_index(
        masses,
        modes,
    )

    positive = search_neutral_mass(
        index=index,
        neutral_mass=100.0,
        mode="positive",
        tolerance_da=0.01,
    )

    negative = search_neutral_mass(
        index=index,
        neutral_mass=100.0,
        mode="negative",
        tolerance_da=0.01,
    )

    np.testing.assert_array_equal(
        positive,
        [0],
    )

    np.testing.assert_array_equal(
        negative,
        [1],
    )


def test_returns_empty_for_unknown_mode():
    index = build_neutral_mass_index(
        [100.0],
        ["positive"],
    )

    result = search_neutral_mass(
        index=index,
        neutral_mass=100.0,
        mode="negative",
        tolerance_da=0.01,
    )

    assert len(result) == 0


def test_returns_empty_for_nan_query_mass():
    index = build_neutral_mass_index(
        [100.0],
        ["positive"],
    )

    result = search_neutral_mass(
        index=index,
        neutral_mass=np.nan,
        mode="positive",
        tolerance_da=0.01,
    )

    assert len(result) == 0


def test_rejects_negative_tolerance():
    index = build_neutral_mass_index(
        [100.0],
        ["positive"],
    )

    with pytest.raises(
        ValueError,
        match="non-negative",
    ):
        search_neutral_mass(
            index=index,
            neutral_mass=100.0,
            mode="positive",
            tolerance_da=-0.01,
        )


def test_multi_search_returns_union():
    masses = np.array(
        [
            100.000,
            100.005,
            200.000,
            200.005,
            300.000,
        ]
    )

    modes = np.array(
        [
            "positive",
            "positive",
            "negative",
            "negative",
            "positive",
        ]
    )

    index = build_neutral_mass_index(
        masses,
        modes,
    )

    result = search_neutral_masses(
        index=index,
        neutral_masses=[
            100.0,
            200.0,
        ],
        modes=[
            "positive",
            "negative",
        ],
        tolerance_da=0.01,
    )

    np.testing.assert_array_equal(
        result,
        [
            0,
            1,
            2,
            3,
        ],
    )


def test_multi_search_removes_duplicate_rows():
    masses = np.array(
        [
            100.000,
            100.005,
            100.020,
        ]
    )

    modes = np.array(
        [
            "positive",
            "positive",
            "positive",
        ]
    )

    index = build_neutral_mass_index(
        masses,
        modes,
    )

    result = search_neutral_masses(
        index=index,
        neutral_masses=[
            100.000,
            100.005,
        ],
        modes=[
            "positive",
            "positive",
        ],
        tolerance_da=0.01,
    )

    np.testing.assert_array_equal(
        result,
        [
            0,
            1,
        ],
    )


def test_rejects_mismatched_build_lengths():
    with pytest.raises(
        ValueError,
        match="same length",
    ):
        build_neutral_mass_index(
            neutral_masses=[
                100.0,
                200.0,
            ],
            modes=[
                "positive",
            ],
        )


def test_rejects_mismatched_search_lengths():
    index = build_neutral_mass_index(
        [100.0],
        ["positive"],
    )

    with pytest.raises(
        ValueError,
        match="same length",
    ):
        search_neutral_masses(
            index=index,
            neutral_masses=[
                100.0,
                200.0,
            ],
            modes=[
                "positive",
            ],
            tolerance_da=0.01,
        )