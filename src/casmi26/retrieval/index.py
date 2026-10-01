from pathlib import Path

import pandas as pd
from scipy import sparse


def save_retrieval_index(
    output_dir: Path,
    metadata: pd.DataFrame,
    vectors: sparse.csr_matrix,
) -> None:
    if not sparse.isspmatrix_csr(vectors):
        raise ValueError(
            "vectors must be a CSR matrix"
        )

    if len(metadata) != vectors.shape[0]:
        raise ValueError(
            "metadata and vectors must contain "
            "the same number of rows"
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    metadata.to_parquet(
        output_dir / "metadata.parquet",
        index=False,
    )

    sparse.save_npz(
        output_dir / "vectors.npz",
        vectors,
        compressed=True,
    )


def load_retrieval_index(
    index_dir: Path,
) -> tuple[pd.DataFrame, sparse.csr_matrix]:
    metadata = pd.read_parquet(
        index_dir / "metadata.parquet"
    )

    vectors = sparse.load_npz(
        index_dir / "vectors.npz"
    ).tocsr()

    if len(metadata) != vectors.shape[0]:
        raise ValueError(
            "retrieval index is inconsistent: "
            "metadata/vector row counts differ"
        )

    return metadata, vectors