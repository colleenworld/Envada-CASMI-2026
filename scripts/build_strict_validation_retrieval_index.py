from __future__ import annotations

from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy import sparse

from casmi26.retrieval.split_index import (
    filter_to_training_structures,
)
from casmi26.spectra.binning import bin_spectrum


TRAIN_PATH = Path("data/raw/train.parquet")

TRAIN_STRUCTURES_PATH = Path(
    "data/processed/splits/train_structures.parquet"
)

VALIDATION_STRUCTURES_PATH = Path(
    "data/processed/splits/validation_structures.parquet"
)

OUTPUT_DIR = Path(
    "data/processed/retrieval_index_strict_validation"
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


def load_structure_keys(
    path: Path,
) -> set[str]:
    frame = pd.read_parquet(
        path,
        columns=[
            "inchikey14",
        ],
    )

    return set(
        frame[
            "inchikey14"
        ]
        .astype(str)
    )


def main() -> None:
    training_keys = load_structure_keys(
        TRAIN_STRUCTURES_PATH
    )

    validation_keys = load_structure_keys(
        VALIDATION_STRUCTURES_PATH
    )

    overlap = (
        training_keys
        & validation_keys
    )

    if overlap:
        raise RuntimeError(
            "Train/validation split is not structure-disjoint. "
            f"Examples: {sorted(overlap)[:20]}"
        )

    print(
        "Training structures:",
        f"{len(training_keys):,}",
    )

    print(
        "Validation structures excluded:",
        f"{len(validation_keys):,}",
    )

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

    total_input_rows = (
        parquet_file.metadata.num_rows
    )

    print(
        "Input spectra:",
        f"{total_input_rows:,}",
    )

    columns = [
        *METADATA_COLUMNS,
        "ms2_mzs",
        "ms2_normalized_intensities",
    ]

    metadata_chunks = []
    vector_paths = []

    processed_input_rows = 0
    indexed_rows = 0

    for chunk_number, batch in enumerate(
        parquet_file.iter_batches(
            batch_size=BATCH_SIZE,
            columns=columns,
        ),
        start=1,
    ):
        frame = batch.to_pandas()

        processed_input_rows += len(
            frame
        )

        frame = filter_to_training_structures(
            frame,
            training_keys,
        )

        if frame.empty:
            print(
                f"\rProcessed "
                f"{processed_input_rows:,}/"
                f"{total_input_rows:,} input spectra | "
                f"indexed {indexed_rows:,}",
                end="",
                flush=True,
            )
            continue

        vectors = bin_batch(
            frame
        )

        metadata = (
            frame[
                METADATA_COLUMNS
            ]
            .reset_index(
                drop=True
            )
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

        indexed_rows += len(
            frame
        )

        print(
            f"\rProcessed "
            f"{processed_input_rows:,}/"
            f"{total_input_rows:,} input spectra | "
            f"indexed {indexed_rows:,}",
            end="",
            flush=True,
        )

    print()

    if (
        processed_input_rows
        != total_input_rows
    ):
        raise RuntimeError(
            "Input row count does not match parquet metadata"
        )

    if not metadata_chunks:
        raise RuntimeError(
            "Strict-validation retrieval index would be empty"
        )

    print(
        "Combining metadata..."
    )

    all_metadata = pd.concat(
        metadata_chunks,
        ignore_index=True,
    )

    print(
        "Combining sparse vectors..."
    )

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
            "Metadata/vector row counts do not match"
        )

    indexed_keys = set(
        all_metadata[
            "inchikey14"
        ]
        .astype(str)
    )

    leaked_validation = sorted(
        indexed_keys
        & validation_keys
    )

    if leaked_validation:
        raise RuntimeError(
            "Validation structures leaked into the strict "
            "retrieval index. "
            f"Examples: {leaked_validation[:20]}"
        )

    unexpected_keys = sorted(
        indexed_keys
        - training_keys
    )

    if unexpected_keys:
        raise RuntimeError(
            "Retrieval index contains structures outside "
            "the training split. "
            f"Examples: {unexpected_keys[:20]}"
        )

    print(
        "Writing metadata..."
    )

    all_metadata.to_parquet(
        METADATA_PATH,
        index=False,
    )

    print(
        "Writing sparse matrix..."
    )

    sparse.save_npz(
        VECTORS_PATH,
        all_vectors,
        compressed=True,
    )

    shutil.rmtree(
        CHUNK_DIR
    )

    print()
    print(
        "Strict-validation retrieval index complete"
    )
    print(
        "------------------------------------------"
    )
    print(
        "Rows:",
        f"{all_vectors.shape[0]:,}",
    )
    print(
        "Structures:",
        f"{len(indexed_keys):,}",
    )
    print(
        "Dimensions:",
        f"{all_vectors.shape[1]:,}",
    )
    print(
        "Validation structures present:",
        len(
            indexed_keys
            & validation_keys
        ),
    )
    print(
        "Metadata:",
        METADATA_PATH,
    )
    print(
        "Vectors:",
        VECTORS_PATH,
    )


if __name__ == "__main__":
    main()
