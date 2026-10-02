import numpy as np

from casmi26.retrieval.ranking import (
    build_candidate_features,
    candidate_membership,
    candidate_scores,
    candidate_spectral_ranks,
)
def test_candidate_scores():
    scores = np.array(
        [
            0.10,
            0.25,
            0.80,
            -np.inf,
            0.60,
        ],
        dtype=np.float32,
    )

    result = candidate_scores(
        [2, 4, 0],
        scores,
    )

    np.testing.assert_allclose(
        result,
        np.array(
            [
                0.80,
                0.60,
                0.10,
            ],
            dtype=np.float32,
        ),
    )


def test_candidate_scores_preserves_negative_infinity():
    scores = np.array(
        [
            0.5,
            -np.inf,
            0.3,
        ],
        dtype=np.float32,
    )

    result = candidate_scores(
        [1, 2],
        scores,
    )

    assert np.isneginf(
        result[0]
    )

    assert result[1] == np.float32(
        0.3
    )


def test_candidate_scores_empty():
    scores = np.array(
        [0.1, 0.2],
        dtype=np.float32,
    )

    result = candidate_scores(
        [],
        scores,
    )

    assert len(result) == 0
    assert result.dtype == np.float32


def test_candidate_scores_rejects_negative_id():
    scores = np.array(
        [0.1, 0.2],
        dtype=np.float32,
    )

    try:
        candidate_scores(
            [-1],
            scores,
        )
    except ValueError as error:
        assert (
            str(error)
            == "candidate IDs must be non-negative"
        )
    else:
        raise AssertionError(
            "Expected ValueError"
        )


def test_candidate_scores_rejects_out_of_range_id():
    scores = np.array(
        [0.1, 0.2],
        dtype=np.float32,
    )

    try:
        candidate_scores(
            [2],
            scores,
        )
    except ValueError as error:
        assert (
            str(error)
            == "candidate ID exceeds score vector"
        )
    else:
        raise AssertionError(
            "Expected ValueError"
        )


def test_build_candidate_features_with_scores():
    scores = np.array(
        [
            0.10,
            0.20,
            0.30,
            0.40,
            0.50,
            0.60,
        ],
        dtype=np.float32,
    )

    result = build_candidate_features(
        hybrid_candidate_ids=[
            5,
            2,
            4,
        ],
        same_candidate_ids=[
            2,
            5,
        ],
        opposite_candidate_ids=[
            4,
        ],
        spectral_candidate_ids=[
            5,
            4,
        ],
        scores=scores,
    )

    np.testing.assert_allclose(
        result[
            "same_polarity_cosine"
        ],
        np.array(
            [
                0.60,
                0.30,
                0.50,
            ],
            dtype=np.float32,
        ),
    )

def test_candidate_membership():
    candidate_ids = np.array(
        [10, 20, 30, 40],
        dtype=np.int64,
    )

    members = np.array(
        [20, 40],
        dtype=np.int64,
    )

    result = candidate_membership(
        candidate_ids,
        members,
    )

    np.testing.assert_array_equal(
        result,
        np.array(
            [False, True, False, True]
        ),
    )


def test_candidate_membership_empty_members():
    candidate_ids = np.array(
        [10, 20, 30],
        dtype=np.int64,
    )

    result = candidate_membership(
        candidate_ids,
        [],
    )

    np.testing.assert_array_equal(
        result,
        np.array(
            [False, False, False]
        ),
    )


def test_candidate_membership_empty_candidates():
    result = candidate_membership(
        np.array(
            [],
            dtype=np.int64,
        ),
        [10, 20],
    )

    assert result.dtype == bool
    assert len(result) == 0


def test_candidate_spectral_ranks():
    candidate_ids = np.array(
        [10, 20, 30, 40],
        dtype=np.int64,
    )

    spectral_ids = np.array(
        [30, 10, 40],
        dtype=np.int64,
    )

    result = candidate_spectral_ranks(
        candidate_ids,
        spectral_ids,
    )

    np.testing.assert_equal(
        result,
        np.array(
            [2.0, np.nan, 1.0, 3.0]
        ),
    )


def test_candidate_spectral_ranks_empty():
    candidate_ids = np.array(
        [10, 20],
        dtype=np.int64,
    )

    result = candidate_spectral_ranks(
        candidate_ids,
        [],
    )

    assert np.isnan(
        result
    ).all()


def test_build_candidate_features():
    result = build_candidate_features(
        hybrid_candidate_ids=[
            10,
            20,
            30,
            40,
            50,
        ],
        same_candidate_ids=[
            10,
            20,
            30,
        ],
        opposite_candidate_ids=[
            20,
            40,
        ],
        spectral_candidate_ids=[
            50,
            30,
            10,
        ],
    )

    np.testing.assert_array_equal(
        result["candidate_id"],
        np.array(
            [10, 20, 30, 40, 50]
        ),
    )

    np.testing.assert_array_equal(
        result["in_same_mass"],
        np.array(
            [
                True,
                True,
                True,
                False,
                False,
            ]
        ),
    )

    np.testing.assert_array_equal(
        result["in_opposite_mass"],
        np.array(
            [
                False,
                True,
                False,
                True,
                False,
            ]
        ),
    )

    np.testing.assert_array_equal(
        result["in_spectral_top25"],
        np.array(
            [
                True,
                False,
                True,
                False,
                True,
            ]
        ),
    )

    np.testing.assert_equal(
        result["spectral_rank"],
        np.array(
            [
                3.0,
                np.nan,
                2.0,
                np.nan,
                1.0,
            ]
        ),
    )


def test_build_candidate_features_preserves_hybrid_order():
    result = build_candidate_features(
        hybrid_candidate_ids=[
            50,
            10,
            40,
        ],
        same_candidate_ids=[
            10,
            40,
        ],
        opposite_candidate_ids=[],
        spectral_candidate_ids=[
            50,
            40,
        ],
    )

    np.testing.assert_array_equal(
        result["candidate_id"],
        np.array(
            [50, 10, 40]
        ),
    )