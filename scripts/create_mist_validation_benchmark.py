from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


RAW_TRAIN = Path("data/raw/train.parquet")

VALIDATION_PATH = Path(
    "data/processed/splits/validation_structures.parquet"
)

COCONUT_PATH = Path(
    "data/external/coconut/coconut_approved.parquet"
)

OUTPUT_DIR = Path(
    "data/processed/mist/validation_500"
)

MANIFEST_PATH = OUTPUT_DIR / "manifest.parquet"

TARGET_SIZE = 500
SEED = 20261005


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    validation = pd.read_parquet(VALIDATION_PATH)
    validation_ids = set(validation["inchikey14"])

    coconut = pd.read_parquet(
        COCONUT_PATH,
        columns=[
            "inchikey14",
            "molecular_formula",
            "smiles",
        ],
    )

    coconut_ids = set(coconut["inchikey14"])

    #
    # This benchmark deliberately targets validation structures whose
    # identities are available in our approved external structure
    # catalog.
    #
    recoverable_ids = validation_ids & coconut_ids

    print("Validation structures:", len(validation_ids))
    print(
        "Validation structures in approved COCONUT:",
        len(recoverable_ids),
    )

    #
    # Stream train.parquet instead of loading the ~3 GB file.
    #
    parquet = pq.ParquetFile(RAW_TRAIN)

    columns = [
        "inchikey14",
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

    for batch in parquet.iter_batches(
        batch_size=100_000,
        columns=columns,
    ):
        df = batch.to_pandas()

        mask = (
            df["inchikey14"].isin(recoverable_ids)
            & (df["adduct"] == "[M+H]+")
        )

        if mask.any():
            chunks.append(df.loc[mask].copy())

    if not chunks:
        raise RuntimeError(
            "No eligible [M+H]+ validation spectra found."
        )

    spectra = pd.concat(
        chunks,
        ignore_index=True,
    )

    print(
        "Eligible [M+H]+ spectra:",
        len(spectra),
    )
    print(
        "Eligible structures:",
        spectra["inchikey14"].nunique(),
    )

    #
    # Deterministic representative spectrum:
    # use the spectrum with the most peaks.
    #
    spectra = spectra.sort_values(
        [
            "inchikey14",
            "num_peaks",
            "precursor_mz",
        ],
        ascending=[True, False, True],
        kind="stable",
    )

    representatives = (
        spectra
        .drop_duplicates(
            "inchikey14",
            keep="first",
        )
        .reset_index(drop=True)
    )

    if len(representatives) < TARGET_SIZE:
        raise RuntimeError(
            f"Only {len(representatives)} eligible structures; "
            f"cannot select {TARGET_SIZE}."
        )

    #
    # Random selection is allowed here because the random seed and
    # rules are frozen before looking at downstream performance.
    #
    selected = (
        representatives
        .sample(
            n=TARGET_SIZE,
            random_state=SEED,
        )
        .sort_values("inchikey14")
        .reset_index(drop=True)
    )

    selected.insert(
        0,
        "spec",
        "mistval_" + selected["inchikey14"],
    )

    selected = selected.rename(
        columns={
            "inchikey14": "query_inchikey14",
            "molecular_formula": "truth_formula",
        }
    )

    selected.to_parquet(
        MANIFEST_PATH,
        index=False,
    )

    print("\nFrozen MIST validation benchmark")
    print("--------------------------------")
    print("Queries:", len(selected))
    print(
        "Unique structures:",
        selected["query_inchikey14"].nunique(),
    )
    print(
        "Adducts:",
        selected["adduct"].value_counts().to_dict(),
    )

    print("\nPeak counts:")
    print(
        selected["num_peaks"]
        .describe(
            percentiles=[0.5, 0.9, 0.95, 0.99]
        )
        .to_string()
    )

    print(f"\nWrote {MANIFEST_PATH}")


if __name__ == "__main__":
    main()