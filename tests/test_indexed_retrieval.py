import numpy as np
import pytest
from scipy import sparse
import pandas as pd

from casmi26.retrieval.indexed import (
    aggregate_query_structure_scores,
    cosine_similarity_block,
    search_structure_blockwise,
    search_structures_blockwise,
    search_structures_indexed_blockwise,
)

def test_indexed_batched_search_leaves_unmatched_candidates_at_negative_infinity():
    query_vectors = sparse.csr_matrix(
        np.array(
            [
                [1.0, 0.0],
            ],
            dtype=np.float32,
        )
    )

    reference_vectors = sparse.csr_matrix(
        np.array(
            [
                [1.0, 0.0],
                [1.0, 0.0],
            ],
            dtype=np.float32,
        )
    )

    scores = (
        search_structures_indexed_blockwise(
            query_vectors=query_vectors,
            query_structure_ids=[
                0,
            ],
            query_modes=[
                "positive",
            ],
            reference_vectors=reference_vectors,
            reference_candidate_ids=[
                0,
                1,
            ],
            reference_modes=[
                "positive",
                "negative",
            ],
            query_structure_count=1,
            candidate_count=2,
            block_size=2,
        )
    )

    assert scores[
        0,
        0,
    ] == pytest.approx(
        1.0
    )

    assert np.isneginf(
        scores[
            0,
            1,
        ]
    )

def test_indexed_batched_search_matches_dictionary_search():
    query_vectors = sparse.csr_matrix(
        np.array(
            [
                [1.0, 0.0, 0.0],
                [0.8, 0.6, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.8, 0.6],
            ],
            dtype=np.float32,
        )
    )

    query_keys = np.array(
        [
            "QUERY_A",
            "QUERY_A",
            "QUERY_B",
            "QUERY_B",
        ]
    )

    query_modes = np.array(
        [
            "positive",
            "positive",
            "negative",
            "negative",
        ]
    )

    reference_vectors = sparse.csr_matrix(
        np.array(
            [
                [1.0, 0.0, 0.0],
                [0.8, 0.6, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.8, 0.6],
                [0.6, 0.8, 0.0],
                [0.0, 0.6, 0.8],
            ],
            dtype=np.float32,
        )
    )

    reference_metadata = pd.DataFrame(
        {
            "inchikey14": [
                "CANDIDATE_X",
                "CANDIDATE_X",
                "CANDIDATE_Y",
                "CANDIDATE_Y",
                "CANDIDATE_Z",
                "CANDIDATE_Z",
            ],
            "ionization_mode": [
                "positive",
                "positive",
                "negative",
                "negative",
                "positive",
                "negative",
            ],
        }
    )

    # Existing implementation.
    dictionary_results = (
        search_structures_blockwise(
            query_vectors=query_vectors,
            query_structure_keys=query_keys,
            query_modes=query_modes,
            reference_vectors=reference_vectors,
            reference_metadata=reference_metadata,
            block_size=2,
        )
    )

    # Integer IDs for query structures.
    query_id_by_key = {
        "QUERY_A": 0,
        "QUERY_B": 1,
    }

    query_structure_ids = np.array(
        [
            query_id_by_key[key]
            for key in query_keys
        ],
        dtype=np.int64,
    )

    # Integer IDs for candidate structures.
    candidate_keys = np.array(
        [
            "CANDIDATE_X",
            "CANDIDATE_Y",
            "CANDIDATE_Z",
        ]
    )

    candidate_id_by_key = {
        key: candidate_id
        for candidate_id, key in enumerate(
            candidate_keys
        )
    }

    reference_candidate_ids = np.array(
        [
            candidate_id_by_key[key]
            for key in reference_metadata[
                "inchikey14"
            ]
        ],
        dtype=np.int64,
    )

    indexed_scores = (
        search_structures_indexed_blockwise(
            query_vectors=query_vectors,
            query_structure_ids=(
                query_structure_ids
            ),
            query_modes=query_modes,
            reference_vectors=reference_vectors,
            reference_candidate_ids=(
                reference_candidate_ids
            ),
            reference_modes=(
                reference_metadata[
                    "ionization_mode"
                ].to_numpy()
            ),
            query_structure_count=2,
            candidate_count=3,
            block_size=2,
        )
    )

    assert indexed_scores.shape == (
        2,
        3,
    )

    for query_key, query_id in (
        query_id_by_key.items()
    ):
        for (
            candidate_key,
            candidate_score,
        ) in dictionary_results[
            query_key
        ].items():
            candidate_id = (
                candidate_id_by_key[
                    candidate_key
                ]
            )

            assert indexed_scores[
                query_id,
                candidate_id,
            ] == pytest.approx(
                candidate_score
            )

def test_batched_search_matches_single_structure_search():
    query_vectors = sparse.csr_matrix(
        np.array(
            [
                [1.0, 0.0, 0.0],
                [0.8, 0.6, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.8, 0.6],
            ],
            dtype=np.float32,
        )
    )

    query_keys = np.array(
        [
            "QUERY_A",
            "QUERY_A",
            "QUERY_B",
            "QUERY_B",
        ]
    )

    query_modes = np.array(
        [
            "positive",
            "positive",
            "negative",
            "negative",
        ]
    )

    reference_vectors = sparse.csr_matrix(
        np.array(
            [
                [1.0, 0.0, 0.0],
                [0.8, 0.6, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.8, 0.6],
                [0.6, 0.8, 0.0],
                [0.0, 0.6, 0.8],
            ],
            dtype=np.float32,
        )
    )

    reference_metadata = pd.DataFrame(
        {
            "inchikey14": [
                "CANDIDATE_X",
                "CANDIDATE_X",
                "CANDIDATE_Y",
                "CANDIDATE_Y",
                "CANDIDATE_Z",
                "CANDIDATE_Z",
            ],
            "ionization_mode": [
                "positive",
                "positive",
                "negative",
                "negative",
                "positive",
                "negative",
            ],
        }
    )

    batched = search_structures_blockwise(
        query_vectors=query_vectors,
        query_structure_keys=query_keys,
        query_modes=query_modes,
        reference_vectors=reference_vectors,
        reference_metadata=reference_metadata,
        block_size=2,
    )

    for query_key in [
        "QUERY_A",
        "QUERY_B",
    ]:
        query_mask = (
            query_keys == query_key
        )

        single = search_structure_blockwise(
            query_vectors=(
                query_vectors[
                    query_mask
                ]
            ),
            query_modes=(
                query_modes[
                    query_mask
                ]
            ),
            reference_vectors=(
                reference_vectors
            ),
            reference_metadata=(
                reference_metadata
            ),
            block_size=2,
        )

        assert (
            batched[query_key].keys()
            == single.keys()
        )

        for candidate_key in single:
            assert (
                batched[
                    query_key
                ][
                    candidate_key
                ]
                == pytest.approx(
                    single[
                        candidate_key
                    ]
                )
            )

def test_cosine_similarity_block():
    queries = sparse.csr_matrix(
        np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
            ],
            dtype=np.float32,
        )
    )

    candidates = sparse.csr_matrix(
        np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 0.0],
            ],
            dtype=np.float32,
        )
    )

    scores = cosine_similarity_block(
        queries,
        candidates,
    )

    np.testing.assert_allclose(
        scores,
        np.array(
            [
                [1.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
            ]
        ),
    )


def test_aggregates_multiple_query_spectra():
    similarities = np.array(
        [
            [0.9, 0.2],
            [0.7, 0.8],
        ]
    )

    results = (
        aggregate_query_structure_scores(
            similarities,
            np.array(["A", "A"]),
            np.array(["X", "Y"]),
        )
    )

    assert results["A"]["X"] == pytest.approx(
        0.9
    )

    assert results["A"]["Y"] == pytest.approx(
        0.8
    )


def test_aggregates_multiple_candidate_spectra():
    similarities = np.array(
        [
            [0.2, 0.9, 0.4],
        ]
    )

    results = (
        aggregate_query_structure_scores(
            similarities,
            np.array(["A"]),
            np.array(["X", "X", "Y"]),
        )
    )

    assert results["A"]["X"] == pytest.approx(
        0.9
    )

    assert results["A"]["Y"] == pytest.approx(
        0.4
    )


def test_aggregates_both_sides():
    similarities = np.array(
        [
            [0.1, 0.7, 0.3],
            [0.8, 0.4, 0.6],
        ]
    )

    results = (
        aggregate_query_structure_scores(
            similarities,
            np.array(["A", "A"]),
            np.array(["X", "X", "Y"]),
        )
    )

    assert results["A"]["X"] == pytest.approx(
        0.8
    )

    assert results["A"]["Y"] == pytest.approx(
        0.6
    )


def test_handles_multiple_query_structures():
    similarities = np.array(
        [
            [0.9, 0.1],
            [0.2, 0.8],
        ]
    )

    results = (
        aggregate_query_structure_scores(
            similarities,
            np.array(["A", "B"]),
            np.array(["X", "Y"]),
        )
    )

    assert results["A"]["X"] == pytest.approx(
        0.9
    )

    assert results["B"]["Y"] == pytest.approx(
        0.8
    )


def test_rejects_dimension_mismatch():
    queries = sparse.csr_matrix(
        (2, 3)
    )

    candidates = sparse.csr_matrix(
        (2, 4)
    )

    with pytest.raises(
        ValueError,
        match="dimensions differ",
    ):
        cosine_similarity_block(
            queries,
            candidates,
        )


def test_rejects_key_shape_mismatch():
    similarities = np.zeros(
        (2, 3)
    )

    with pytest.raises(
        ValueError,
        match="do not match",
    ):
        aggregate_query_structure_scores(
            similarities,
            np.array(["A"]),
            np.array(["X", "Y", "Z"]),
        )

def test_blockwise_search_aggregates_candidates():
    query_vectors = sparse.csr_matrix(
        np.array(
            [
                [1.0, 0.0],
                [0.8, 0.6],
            ],
            dtype=np.float32,
        )
    )

    reference_vectors = sparse.csr_matrix(
        np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [0.8, 0.6],
                [0.6, 0.8],
            ],
            dtype=np.float32,
        )
    )

    reference_metadata = pd.DataFrame(
        {
            "inchikey14": [
                "X",
                "X",
                "Y",
                "Z",
            ],
            "ionization_mode": [
                "positive",
                "positive",
                "positive",
                "negative",
            ],
        }
    )

    scores = search_structure_blockwise(
        query_vectors=query_vectors,
        query_modes=[
            "positive",
            "positive",
        ],
        reference_vectors=reference_vectors,
        reference_metadata=reference_metadata,
        block_size=2,
    )

    assert scores["X"] == pytest.approx(
        1.0
    )

    assert scores["Y"] == pytest.approx(
        1.0
    )

    # Negative reference must not be compared with
    # positive query spectra.
    assert "Z" not in scores


def test_blockwise_search_handles_multiple_polarities():
    query_vectors = sparse.csr_matrix(
        np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
            ],
            dtype=np.float32,
        )
    )

    reference_vectors = sparse.csr_matrix(
        np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
            ],
            dtype=np.float32,
        )
    )

    reference_metadata = pd.DataFrame(
        {
            "inchikey14": [
                "POS",
                "NEG",
            ],
            "ionization_mode": [
                "positive",
                "negative",
            ],
        }
    )

    scores = search_structure_blockwise(
        query_vectors=query_vectors,
        query_modes=[
            "positive",
            "negative",
        ],
        reference_vectors=reference_vectors,
        reference_metadata=reference_metadata,
        block_size=1,
    )

    assert scores["POS"] == pytest.approx(
        1.0
    )

    assert scores["NEG"] == pytest.approx(
        1.0
    )