from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


ROOT = Path("data/processed/mist/validation_500_v2")

HYPOTHESES_PATH = ROOT / "mist_multispectrum_hypotheses.parquet"
MANIFEST_PATH = ROOT / "manifest.parquet"
TRAIN_PATH = Path("data/raw/train.parquet")
OUTPUT_MGF = ROOT / "spectra_multispectrum.mgf"


def main():
    hypotheses = pd.read_parquet(HYPOTHESES_PATH)
    manifest = pd.read_parquet(MANIFEST_PATH)

    required = {
        "hypothesis_spec",
        "query_inchikey14",
        "candidate_formula",
        "source_train_row_id",
        "source_precursor_mz",
        "source_raw_peak_count",
        "source_mist_usable_peak_count",
    }

    missing_columns = required - set(hypotheses.columns)

    if missing_columns:
        raise ValueError(
            f"Hypothesis file missing columns: "
            f"{sorted(missing_columns)}"
        )

    #
    # Use the complete frozen 500-query universe so that the same
    # filtering context is used everywhere.
    #
    query_ids = set(
        manifest["query_inchikey14"]
    )

    parquet = pq.ParquetFile(TRAIN_PATH)

    columns = [
        "inchikey14",
        "ingest_lib",
        "molecular_formula",
        "ionization_mode",
        "adduct",
        "precursor_mz",
        "instrument_type",
        "num_peaks",
        "ms2_mzs",
        "ms2_normalized_intensities",
    ]

    chunks = []
    row_offset = 0

    for batch in parquet.iter_batches(
        batch_size=100_000,
        columns=columns,
    ):
        df = batch.to_pandas()

        #
        # Stable absolute train.parquet row identifier.
        # This MUST be assigned before filtering.
        #
        df["train_row_id"] = np.arange(
            row_offset,
            row_offset + len(df),
            dtype=np.int64,
        )

        row_offset += len(df)

        matched = df[
            df["inchikey14"].isin(query_ids)
            & (df["adduct"] == "[M+H]+")
        ].copy()

        if not matched.empty:
            chunks.append(matched)

    if not chunks:
        raise RuntimeError(
            "No source spectra found."
        )

    spectra = pd.concat(
        chunks,
        ignore_index=True,
    )

    if spectra["train_row_id"].duplicated().any():
        raise ValueError(
            "Duplicate train_row_id values found."
        )

    spectra_by_id = spectra.set_index(
        "train_row_id"
    )

    requested_ids = set(
        hypotheses["source_train_row_id"]
        .astype(int)
    )

    available_ids = set(
        spectra_by_id.index.astype(int)
    )

    missing = requested_ids - available_ids

    if missing:
        raise ValueError(
            f"Missing {len(missing)} selected source spectra. "
            f"Example IDs: {sorted(missing)[:10]}"
        )

    #
    # Verify that the stable row IDs resolve to exactly the
    # spectra selected by the hypothesis builder.
    #
    query_mismatches = 0
    precursor_mismatches = 0
    peak_count_mismatches = 0

    for row in hypotheses.itertuples(index=False):
        train_row_id = int(
            row.source_train_row_id
        )

        spec = spectra_by_id.loc[
            train_row_id
        ]

        if spec.inchikey14 != row.query_inchikey14:
            query_mismatches += 1

        if not np.isclose(
            float(spec.precursor_mz),
            float(row.source_precursor_mz),
            rtol=0,
            atol=1e-10,
        ):
            precursor_mismatches += 1

        actual_peaks = len(
            spec.ms2_mzs
        )

        if actual_peaks != int(
            row.source_raw_peak_count
        ):
            peak_count_mismatches += 1

    print("Pre-export consistency")
    print("----------------------")
    print("Hypotheses:", len(hypotheses))
    print("Query mismatches:", query_mismatches)
    print(
        "Precursor mismatches:",
        precursor_mismatches,
    )
    print(
        "Peak-count mismatches:",
        peak_count_mismatches,
    )

    if query_mismatches:
        raise ValueError(
            "Source/query mismatches found."
        )

    if precursor_mismatches:
        raise ValueError(
            "Source precursor mismatches found."
        )

    if peak_count_mismatches:
        raise ValueError(
            "Source peak-count mismatches found."
        )

    written = 0
    unique_source_ids = set()

    with OUTPUT_MGF.open("w") as f:
        for row in hypotheses.itertuples(
            index=False
        ):
            train_row_id = int(
                row.source_train_row_id
            )

            spec = spectra_by_id.loc[
                train_row_id
            ]

            raw_peak_count = len(
                spec.ms2_mzs
            )

            if int(row.source_mist_usable_peak_count) <= 0:
                raise ValueError(
                    "Selected source spectrum has no "
                    "MIST-usable peaks: "
                    f"{row.hypothesis_spec}; "
                    f"train_row_id={train_row_id}"
                )

            if (
                spec.inchikey14
                != row.query_inchikey14
            ):
                raise ValueError(
                    "Selected source spectrum "
                    "belongs to wrong query: "
                    f"{row.hypothesis_spec}"
                )

            unique_source_ids.add(
                train_row_id
            )

            f.write("BEGIN IONS\n")
            f.write(
                f"TITLE={row.hypothesis_spec}\n"
            )
            f.write(
                f"FEATURE_ID="
                f"{row.hypothesis_spec}\n"
            )
            f.write(
                f"PEPMASS="
                f"{float(spec.precursor_mz):.8f}\n"
            )
            f.write("CHARGE=1+\n")

            #
            # Candidate formula only — never oracle truth.
            #
            f.write(
                f"FORMULA="
                f"{row.candidate_formula}\n"
            )

            f.write("ADDUCT=[M+H]+\n")
            f.write("IONMODE=positive\n")

            for mz, intensity in zip(
                spec.ms2_mzs,
                spec.ms2_normalized_intensities,
            ):
                f.write(
                    f"{float(mz):.8f} "
                    f"{float(intensity):.8f}\n"
                )

            f.write("END IONS\n\n")

            written += 1

    print("\nMGF export")
    print("----------")
    print(
        "Hypotheses:",
        len(hypotheses),
    )
    print(
        "MGF blocks written:",
        written,
    )
    print(
        "Unique source spectra used:",
        len(unique_source_ids),
    )

    print(
        "Minimum MIST-usable peak count:",
        int(
            hypotheses[
                "source_mist_usable_peak_count"
            ].min()
        ),
    )
    print(
        "Minimum selected raw peak count:",
        int(
            hypotheses[
                "source_raw_peak_count"
            ].min()
        ),
    )

    print(f"\nWrote {OUTPUT_MGF}")


if __name__ == "__main__":
    main()