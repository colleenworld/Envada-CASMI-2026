import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from casmi26.retrieval.index import (
    load_retrieval_index,
    save_retrieval_index,
)


def test_round_trip(tmp_path):
    metadata = pd.DataFrame(
        {
            "inchikey14": [
                "A",
                "B",
                "C",
            ],
            "ingest_lib": [
                "gnps",
                "riken",
                "mona",
            ],
        }
    )

    vectors = sparse.csr_matrix(
        np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
    )

    save_retrieval_index(
        tmp_path,
        metadata,
        vectors,
    )

    loaded_metadata, loaded_vectors = (
        load_retrieval_index(tmp_path)
    )

    pd.testing.assert_frame_equal(
        loaded_metadata,
        metadata,
    )

    np.testing.assert_array_equal(
        loaded_vectors.toarray(),
        vectors.toarray(),
    )


def test_preserves_sparse_matrix(tmp_path):
    metadata = pd.DataFrame(
        {
            "inchikey14": ["A"],
        }
    )

    vectors = sparse.csr_matrix(
        np.array(
            [[0.0, 1.0, 0.0]],
            dtype=np.float32,
        )
    )

    save_retrieval_index(
        tmp_path,
        metadata,
        vectors,
    )

    _, loaded = load_retrieval_index(
        tmp_path
    )

    assert sparse.isspmatrix_csr(
        loaded
    )

    assert loaded.nnz == 1


def test_rejects_row_count_mismatch(
    tmp_path,
):
    metadata = pd.DataFrame(
        {
            "inchikey14": [
                "A",
                "B",
            ],
        }
    )

    vectors = sparse.csr_matrix(
        (3, 5),
        dtype=np.float32,
    )

    with pytest.raises(
        ValueError,
        match="same number of rows",
    ):
        save_retrieval_index(
            tmp_path,
            metadata,
            vectors,
        )


def test_rejects_dense_vectors(
    tmp_path,
):
    metadata = pd.DataFrame(
        {
            "inchikey14": ["A"],
        }
    )

    vectors = np.zeros(
        (1, 5),
        dtype=np.float32,
    )

    with pytest.raises(
        ValueError,
        match="CSR",
    ):
        save_retrieval_index(
            tmp_path,
            metadata,
            vectors,
        )