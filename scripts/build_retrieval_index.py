from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy import sparse

from casmi26.spectra.binning import bin_spectrum


TRAIN_PATH = Path("data/raw/train.parquet")

OUTPUT_DIR = Path(
    "data/processed/retrieval_index"
)

CHUNK_DIR = OUTPUT_DIR / "_chunks"

METADATA_PATH = (
    OUTPUT_DIR / "metadata.parquet"
)

VECTORS_PATH = (
    OUTPUT_DIR / "vectors.npz"
)

BATCH_SIZE = 10_000

MAX_MZ = 1500.0
BIN_WIDTH = 1.0


METADATA_COLUMNS = [
    "inchikey14",
    "ingest_lib",
    "ionization_mode",
    "adduct",
    "precursor_mz",
]


def bin_batch(
    frame: pd.DataFrame,
) -> sparse.csr_matrix:
    vectors = [
        bin_spectrum(
            mzs,
            intensities,
            max_mz=MAX_MZ,
            bin_width=BIN_WIDTH,
        )
        for mzs, intensities in zip(
            frame["ms2_mzs"],
            frame[
                "ms2_normalized_intensities"
            ],
            strict=True,
        )
    ]

    dense = np.asarray(
        vectors,
        dtype=np.float32,
    )

    return sparse.csr_matrix(
        dense
    )


def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if CHUNK_DIR.exists():
        shutil.rmtree(
            CHUNK_DIR
        )

    CHUNK_DIR.mkdir()

    parquet_file = pq.ParquetFile(
        TRAIN_PATH
    )

    total_rows = (
        parquet_file.metadata.num_rows
    )

    print(
        f"Input spectra: {total_rows:,}"
    )

    columns = [
        *METADATA_COLUMNS,
        "ms2_mzs",
        "ms2_normalized_intensities",
    ]

    metadata_chunks = []
    vector_paths = []

    processed = 0

    for chunk_number, batch in enumerate(
        parquet_file.iter_batches(
            batch_size=BATCH_SIZE,
            columns=columns,
        ),
        start=1,
    ):
        frame = batch.to_pandas()

        vectors = bin_batch(
            frame
        )

        metadata = (
            frame[
                METADATA_COLUMNS
            ]
            .reset_index(drop=True)
        )

        vector_path = (
            CHUNK_DIR
            / f"vectors-{chunk_number:05d}.npz"
        )

        sparse.save_npz(
            vector_path,
            vectors,
            compressed=True,
        )

        vector_paths.append(
            vector_path
        )

        metadata_chunks.append(
            metadata
        )

        processed += len(frame)

        print(
            f"\rProcessed "
            f"{processed:,}/{total_rows:,} "
            f"({processed / total_rows:.1%})",
            end="",
            flush=True,
        )

    print()

    if processed != total_rows:
        raise RuntimeError(
            "Index row count does not match "
            "input row count"
        )

    print("Combining metadata...")

    all_metadata = pd.concat(
        metadata_chunks,
        ignore_index=True,
    )

    print("Combining sparse vectors...")

    matrices = [
        sparse.load_npz(path)
        for path in vector_paths
    ]

    all_vectors = sparse.vstack(
        matrices,
        format="csr",
        dtype=np.float32,
    )

    if len(all_metadata) != (
        all_vectors.shape[0]
    ):
        raise RuntimeError(
            "Metadata/vector row counts "
            "do not match"
        )

    print("Writing metadata...")

    all_metadata.to_parquet(
        METADATA_PATH,
        index=False,
    )

    print("Writing sparse matrix...")

    sparse.save_npz(
        VECTORS_PATH,
        all_vectors,
        compressed=True,
    )

    print("Removing temporary chunks...")

    shutil.rmtree(
        CHUNK_DIR
    )

    print()
    print("Retrieval index complete")
    print("------------------------")
    print(
        f"Rows:       "
        f"{all_vectors.shape[0]:,}"
    )
    print(
        f"Dimensions: "
        f"{all_vectors.shape[1]:,}"
    )
    print(
        f"Non-zero:   "
        f"{all_vectors.nnz:,}"
    )
    density = (
            all_vectors.nnz
            / (
                    all_vectors.shape[0]
                    * all_vectors.shape[1]
            )
    )

    print(
        f"Density:    {density:.2%}"
    )
    print(
        f"Metadata:   {METADATA_PATH}"
    )
    print(
        f"Vectors:    {VECTORS_PATH}"
    )


if __name__ == "__main__":
    main()