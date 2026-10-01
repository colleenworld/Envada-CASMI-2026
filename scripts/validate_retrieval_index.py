from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from casmi26.retrieval.index import (
    load_retrieval_index,
)
from casmi26.spectra.binning import (
    bin_spectrum,
)


INDEX_DIR = Path(
    "data/processed/retrieval_index"
)

TRAIN_PATH = Path(
    "data/raw/train.parquet"
)

SAMPLE_SIZE = 100


def main() -> None:
    print("Loading retrieval index...")

    metadata, vectors = load_retrieval_index(
        INDEX_DIR
    )

    print()
    print("Index")
    print("-----")
    print(
        f"Metadata rows: {len(metadata):,}"
    )
    print(
        f"Vector shape:  {vectors.shape}"
    )
    print(
        f"Vector dtype:  {vectors.dtype}"
    )
    print(
        f"Non-zero:      {vectors.nnz:,}"
    )

    # Pick 100 deterministic rows spread evenly across the
    # entire training dataset.
    indices = np.linspace(
        0,
        len(metadata) - 1,
        SAMPLE_SIZE,
        dtype=int,
    )

    wanted = set(
        int(index)
        for index in indices
    )

    parquet_file = pq.ParquetFile(
        TRAIN_PATH
    )

    columns = [
        "inchikey14",
        "ingest_lib",
        "ionization_mode",
        "adduct",
        "precursor_mz",
        "ms2_mzs",
        "ms2_normalized_intensities",
    ]

    current_row = 0
    checked = 0

    print()
    print(
        f"Validating {SAMPLE_SIZE} rows..."
    )

    for batch in parquet_file.iter_batches(
        batch_size=10_000,
        columns=columns,
    ):
        frame = batch.to_pandas()

        start = current_row
        stop = start + len(frame)

        batch_indices = [
            index
            for index in wanted
            if start <= index < stop
        ]

        for global_index in batch_indices:
            local_index = (
                global_index - start
            )

            raw = frame.iloc[
                local_index
            ]

            indexed = metadata.iloc[
                global_index
            ]

            # Verify metadata alignment.
            assert (
                raw["inchikey14"]
                == indexed["inchikey14"]
            ), (
                f"inchikey14 mismatch "
                f"at row {global_index}"
            )

            assert (
                raw["ingest_lib"]
                == indexed["ingest_lib"]
            ), (
                f"ingest_lib mismatch "
                f"at row {global_index}"
            )

            assert (
                raw["ionization_mode"]
                == indexed["ionization_mode"]
            ), (
                f"ionization_mode mismatch "
                f"at row {global_index}"
            )

            assert (
                raw["adduct"]
                == indexed["adduct"]
            ), (
                f"adduct mismatch "
                f"at row {global_index}"
            )

            # Recreate the vector directly from the raw spectrum.
            raw_vector = bin_spectrum(
                raw["ms2_mzs"],
                raw[
                    "ms2_normalized_intensities"
                ],
                max_mz=1500.0,
                bin_width=1.0,
            )

            indexed_vector = (
                vectors[
                    global_index
                ]
                .toarray()
                .ravel()
            )

            np.testing.assert_allclose(
                indexed_vector,
                raw_vector,
                rtol=1e-6,
                atol=1e-7,
                err_msg=(
                    f"Spectrum mismatch "
                    f"at row {global_index}"
                ),
            )

            checked += 1

        current_row = stop

    print()
    print(
        f"Validated {checked} rows successfully."
    )

    if checked != SAMPLE_SIZE:
        raise RuntimeError(
            f"Expected to validate "
            f"{SAMPLE_SIZE} rows, "
            f"but validated {checked}"
        )

    print(
        "Metadata alignment: PASS"
    )
    print(
        "Spectrum vectors:   PASS"
    )


if __name__ == "__main__":
    main()