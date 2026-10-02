import numpy as np

from casmi26.retrieval.mass_index import (
    build_neutral_mass_index,
)

from casmi26.retrieval.candidates import (
    merge_candidate_ids,
    opposite_polarity_mass_candidates,
    reference_rows_to_candidate_ids,
    same_polarity_mass_candidates,
    top_k_candidate_ids,
)

def test_top_k_candidate_ids():
    scores = np.array(
        [
            0.20,
            0.90,
            0.50,
            0.80,
            0.10,
        ],
        dtype=np.float64,
    )

    result = top_k_candidate_ids(
        scores,
        k=3,
    )

    np.testing.assert_array_equal(
        result,
        np.array(
            [1, 3, 2],
            dtype=np.int64,
        ),
    )


def test_top_k_candidate_ids_ignores_non_finite_scores():
    scores = np.array(
        [
            -np.inf,
            0.90,
            np.nan,
            0.80,
            -np.inf,
        ],
        dtype=np.float64,
    )

    result = top_k_candidate_ids(
        scores,
        k=25,
    )

    np.testing.assert_array_equal(
        result,
        np.array(
            [1, 3],
            dtype=np.int64,
        ),
    )


def test_top_k_candidate_ids_handles_k_larger_than_candidates():
    scores = np.array(
        [
            0.20,
            0.90,
            0.50,
        ],
        dtype=np.float64,
    )

    result = top_k_candidate_ids(
        scores,
        k=25,
    )

    np.testing.assert_array_equal(
        result,
        np.array(
            [1, 2, 0],
            dtype=np.int64,
        ),
    )


def test_top_k_candidate_ids_is_deterministic_for_ties():
    scores = np.array(
        [
            0.50,
            0.90,
            0.90,
            0.90,
            0.20,
        ],
        dtype=np.float64,
    )

    result = top_k_candidate_ids(
        scores,
        k=2,
    )

    np.testing.assert_array_equal(
        result,
        np.array(
            [1, 2],
            dtype=np.int64,
        ),
    )


def test_top_k_candidate_ids_zero_k():
    scores = np.array(
        [0.50, 0.90],
        dtype=np.float64,
    )

    result = top_k_candidate_ids(
        scores,
        k=0,
    )

    assert result.dtype == np.int64
    assert len(result) == 0


def test_top_k_candidate_ids_rejects_negative_k():
    scores = np.array(
        [0.50, 0.90],
        dtype=np.float64,
    )

    try:
        top_k_candidate_ids(
            scores,
            k=-1,
        )
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Expected ValueError"
        )

def test_opposite_polarity_mass_candidates_ignores_unknown_mass():
    reference_masses = np.array(
        [100.000],
        dtype=np.float64,
    )

    reference_modes = np.array(
        ["negative"],
        dtype=object,
    )

    reference_candidate_ids = np.array(
        [10],
        dtype=np.int64,
    )

    index = build_neutral_mass_index(
        neutral_masses=reference_masses,
        modes=reference_modes,
    )

    result = opposite_polarity_mass_candidates(
        index=index,
        neutral_masses=np.array(
            [np.nan],
            dtype=np.float64,
        ),
        modes=np.array(
            ["positive"],
            dtype=object,
        ),
        reference_candidate_ids=(
            reference_candidate_ids
        ),
        tolerance_da=0.01,
    )

    assert len(result) == 0

def test_opposite_polarity_mass_candidates_reverse_direction():
    reference_masses = np.array(
        [
            100.000,
            100.004,
        ],
        dtype=np.float64,
    )

    reference_modes = np.array(
        [
            "positive",
            "negative",
        ],
        dtype=object,
    )

    reference_candidate_ids = np.array(
        [10, 20],
        dtype=np.int64,
    )

    index = build_neutral_mass_index(
        neutral_masses=reference_masses,
        modes=reference_modes,
    )

    result = opposite_polarity_mass_candidates(
        index=index,
        neutral_masses=np.array(
            [100.002],
            dtype=np.float64,
        ),
        modes=np.array(
            ["negative"],
            dtype=object,
        ),
        reference_candidate_ids=(
            reference_candidate_ids
        ),
        tolerance_da=0.01,
    )

    np.testing.assert_array_equal(
        result,
        np.array([10]),
    )

def test_opposite_polarity_mass_candidates():
    reference_masses = np.array(
        [
            100.000,
            100.004,
            100.006,
            200.000,
        ],
        dtype=np.float64,
    )

    reference_modes = np.array(
        [
            "positive",
            "negative",
            "negative",
            "negative",
        ],
        dtype=object,
    )

    reference_candidate_ids = np.array(
        [10, 20, 30, 40],
        dtype=np.int64,
    )

    index = build_neutral_mass_index(
        neutral_masses=reference_masses,
        modes=reference_modes,
    )

    result = opposite_polarity_mass_candidates(
        index=index,
        neutral_masses=np.array(
            [100.000],
            dtype=np.float64,
        ),
        modes=np.array(
            ["positive"],
            dtype=object,
        ),
        reference_candidate_ids=(
            reference_candidate_ids
        ),
        tolerance_da=0.01,
    )

    np.testing.assert_array_equal(
        result,
        np.array([20, 30]),
    )

def test_same_polarity_mass_candidates_ignores_unknown_mass():
    reference_masses = np.array(
        [100.000],
        dtype=np.float64,
    )

    reference_modes = np.array(
        ["positive"],
        dtype=object,
    )

    reference_candidate_ids = np.array(
        [10],
        dtype=np.int64,
    )

    index = build_neutral_mass_index(
        neutral_masses=reference_masses,
        modes=reference_modes,
    )

    result = same_polarity_mass_candidates(
        index=index,
        neutral_masses=np.array(
            [np.nan],
            dtype=np.float64,
        ),
        modes=np.array(
            ["positive"],
            dtype=object,
        ),
        reference_candidate_ids=(
            reference_candidate_ids
        ),
        tolerance_da=0.01,
    )

    assert len(result) == 0

def test_same_polarity_mass_candidates_unions_query_spectra():
    reference_masses = np.array(
        [
            100.000,
            200.000,
            300.000,
        ],
        dtype=np.float64,
    )

    reference_modes = np.array(
        [
            "positive",
            "negative",
            "positive",
        ],
        dtype=object,
    )

    reference_candidate_ids = np.array(
        [10, 20, 30],
        dtype=np.int64,
    )

    index = build_neutral_mass_index(
        neutral_masses=reference_masses,
        modes=reference_modes,
    )

    result = same_polarity_mass_candidates(
        index=index,
        neutral_masses=np.array(
            [100.002, 200.004],
            dtype=np.float64,
        ),
        modes=np.array(
            ["positive", "negative"],
            dtype=object,
        ),
        reference_candidate_ids=(
            reference_candidate_ids
        ),
        tolerance_da=0.01,
    )

    np.testing.assert_array_equal(
        result,
        np.array([10, 20]),
    )

def test_same_polarity_mass_candidates():
    reference_masses = np.array(
        [
            100.000,
            100.005,
            100.006,
            100.004,
            200.000,
        ],
        dtype=np.float64,
    )

    reference_modes = np.array(
        [
            "positive",
            "positive",
            "positive",
            "negative",
            "positive",
        ],
        dtype=object,
    )

    reference_candidate_ids = np.array(
        [
            10,
            10,
            20,
            30,
            40,
        ],
        dtype=np.int64,
    )

    index = build_neutral_mass_index(
        neutral_masses=reference_masses,
        modes=reference_modes,
    )

    result = same_polarity_mass_candidates(
        index=index,
        neutral_masses=np.array(
            [100.000],
            dtype=np.float64,
        ),
        modes=np.array(
            ["positive"],
            dtype=object,
        ),
        reference_candidate_ids=(
            reference_candidate_ids
        ),
        tolerance_da=0.01,
    )

    np.testing.assert_array_equal(
        result,
        np.array([10, 20]),
    )

def test_reference_rows_to_candidate_ids():
    reference_candidate_ids = np.array(
        [10, 10, 20, 30, 30, 40],
        dtype=np.int64,
    )

    result = reference_rows_to_candidate_ids(
        np.array([0, 1, 2, 4]),
        reference_candidate_ids,
    )

    np.testing.assert_array_equal(
        result,
        np.array([10, 20, 30]),
    )


def test_reference_rows_to_candidate_ids_empty():
    reference_candidate_ids = np.array(
        [10, 20, 30],
        dtype=np.int64,
    )

    result = reference_rows_to_candidate_ids(
        np.array([], dtype=np.int64),
        reference_candidate_ids,
    )

    assert result.dtype == np.int64
    assert len(result) == 0

def test_merge_candidate_ids_unions_sources():
    result = merge_candidate_ids(
        np.array([1, 2, 3]),
        np.array([3, 4, 5]),
        np.array([5, 6]),
    )

    np.testing.assert_array_equal(
        result,
        np.array([1, 2, 3, 4, 5, 6]),
    )


def test_merge_candidate_ids_removes_duplicates():
    result = merge_candidate_ids(
        np.array([3, 3, 3]),
        np.array([3, 4, 4]),
    )

    np.testing.assert_array_equal(
        result,
        np.array([3, 4]),
    )


def test_merge_candidate_ids_accepts_empty_sources():
    result = merge_candidate_ids(
        np.array([], dtype=np.int64),
        np.array([7, 8]),
        np.array([], dtype=np.int64),
    )

    np.testing.assert_array_equal(
        result,
        np.array([7, 8]),
    )


def test_merge_candidate_ids_all_empty():
    result = merge_candidate_ids(
        np.array([], dtype=np.int64),
        np.array([], dtype=np.int64),
    )

    assert result.dtype == np.int64
    assert len(result) == 0


def test_merge_candidate_ids_accepts_iterables():
    result = merge_candidate_ids(
        [5, 2, 5],
        (3, 2),
    )

    np.testing.assert_array_equal(
        result,
        np.array([2, 3, 5]),
    )