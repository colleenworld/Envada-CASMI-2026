from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


ROOT = Path("data/processed/mist/validation_500")

FAILURES_PATH = ROOT / "formula_failure_diagnosis.parquet"
TRAIN_PATH = Path("data/raw/train.parquet")

OUTPUT_PATH = ROOT / "formula_failure_source_metadata.parquet"


def main():
    failures = pd.read_parquet(FAILURES_PATH)

    query_ids = set(failures["query_inchikey14"])

    parquet = pq.ParquetFile(TRAIN_PATH)

    columns = [
        "ingest_lib",
        "inchikey14",
        "molecular_formula",
        "ionization_mode",
        "adduct",
        "adduct_orig",
        "precursor_mz",
        "precursor_error_ppm",
        "num_peaks",
        "instrument_type",
    ]

    chunks = []

    for batch in parquet.iter_batches(
        batch_size=100_000,
        columns=columns,
    ):
        df = batch.to_pandas()

        matched = df[
            df["inchikey14"].isin(query_ids)
        ]

        if not matched.empty:
            chunks.append(matched)

    raw = pd.concat(
        chunks,
        ignore_index=True,
    )

    #
    # The validation benchmark selected one [M+H]+ spectrum per
    # structure, choosing highest num_peaks and then precursor_mz.
    # Reproduce that deterministic selection here.
    #
    selected = raw[
        raw["adduct"] == "[M+H]+"
    ].copy()

    selected = selected.sort_values(
        [
            "inchikey14",
            "num_peaks",
            "precursor_mz",
        ],
        ascending=[True, False, True],
        kind="stable",
    )

    selected = (
        selected
        .drop_duplicates(
            "inchikey14",
            keep="first",
        )
        .rename(
            columns={
                "inchikey14": "query_inchikey14",
            }
        )
    )

    diagnosis = failures.merge(
        selected,
        on="query_inchikey14",
        how="left",
        suffixes=("_diagnosis", "_raw"),
    )

    diagnosis.to_parquet(
        OUTPUT_PATH,
        index=False,
    )

    print("Failures:", len(failures))
    print(
        "Matched source rows:",
        diagnosis["ingest_lib"].notna().sum(),
    )

    print("\nFailures by source library:")
    print(
        diagnosis["ingest_lib"]
        .value_counts(dropna=False)
        .to_string()
    )

    print("\nOriginal adduct strings:")
    print(
        diagnosis["adduct_orig"]
        .value_counts(dropna=False)
        .head(30)
        .to_string()
    )

    print("\nSource precursor_error_ppm:")
    errors = pd.to_numeric(
        diagnosis["precursor_error_ppm"],
        errors="coerce",
    )

    print(
        errors.abs()
        .describe(
            percentiles=[
                0.5,
                0.75,
                0.9,
                0.95,
                0.99,
            ]
        )
        .to_string()
    )

    print("\nLargest source precursor errors:")

    tmp = diagnosis.copy()

    tmp["source_abs_precursor_error_ppm"] = (
        pd.to_numeric(
            tmp["precursor_error_ppm"],
            errors="coerce",
        ).abs()
    )

    print(
        tmp.sort_values(
            "source_abs_precursor_error_ppm",
            ascending=False,
        )
        .head(30)
        [
            [
                "query_inchikey14",
                "molecular_formula",
                "ingest_lib",
                "adduct",
                "adduct_orig",
                "precursor_mz_raw",
                "precursor_error_ppm",
                "truth_mass_error_ppm",
                "category",
            ]
        ]
        .to_string(index=False)
    )

    print(f"\nWrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()