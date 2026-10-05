from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


ROOT = Path("data/processed/mist/validation_500")

FAILURES_PATH = ROOT / "formula_failure_diagnosis.parquet"
TRAIN_PATH = Path("data/raw/train.parquet")

PROTON_MASS = 1.007276466621
PPM = 10.0


def main():
    failures = pd.read_parquet(FAILURES_PATH)

    indexed = failures[
        failures["truth_formula_in_index"]
    ].copy()

    query_ids = set(indexed["query_inchikey14"])

    truth_mass = dict(
        zip(
            indexed["query_inchikey14"],
            indexed["truth_formula_mass"],
        )
    )

    parquet = pq.ParquetFile(TRAIN_PATH)

    columns = [
        "inchikey14",
        "ingest_lib",
        "adduct",
        "adduct_orig",
        "precursor_mz",
        "precursor_error_ppm",
        "num_peaks",
    ]

    chunks = []

    for batch in parquet.iter_batches(
        batch_size=100_000,
        columns=columns,
    ):
        df = batch.to_pandas()

        matched = df[
            df["inchikey14"].isin(query_ids)
            & (df["adduct"] == "[M+H]+")
        ].copy()

        if not matched.empty:
            chunks.append(matched)

    spectra = pd.concat(
        chunks,
        ignore_index=True,
    )

    spectra["truth_mass"] = spectra["inchikey14"].map(
        truth_mass
    )

    spectra["implied_neutral_mass"] = (
        spectra["precursor_mz"] - PROTON_MASS
    )

    spectra["computed_truth_ppm"] = (
        (
            spectra["truth_mass"]
            - spectra["implied_neutral_mass"]
        )
        / spectra["implied_neutral_mass"]
        * 1e6
    )

    spectra["within_10ppm"] = (
        spectra["computed_truth_ppm"].abs() <= PPM
    )

    summary_rows = []

    for query_key, group in spectra.groupby(
        "inchikey14",
        sort=False,
    ):
        best = group.loc[
            group["computed_truth_ppm"].abs().idxmin()
        ]

        summary_rows.append(
            {
                "query_inchikey14": query_key,
                "hplus_spectrum_count": len(group),
                "any_within_10ppm": bool(
                    group["within_10ppm"].any()
                ),
                "best_abs_ppm": float(
                    group["computed_truth_ppm"]
                    .abs()
                    .min()
                ),
                "best_ppm": float(
                    best["computed_truth_ppm"]
                ),
                "best_source": best["ingest_lib"],
                "best_precursor_mz": best["precursor_mz"],
                "best_num_peaks": best["num_peaks"],
            }
        )

    summary = pd.DataFrame(summary_rows)

    result = failures.merge(
        summary,
        on="query_inchikey14",
        how="left",
    )

    failures_count = len(failures)

    recovered = result[
        result["any_within_10ppm"] == True
    ]

    print("Frozen validation formula failures:", failures_count)

    print(
        "Failures with indexed truth formula:",
        int(failures["truth_formula_in_index"].sum()),
    )

    print(
        "Recovered by another [M+H]+ spectrum at 10 ppm:",
        len(recovered),
        "/",
        failures_count,
    )

    print(
        "Remaining failures:",
        failures_count - len(recovered),
    )

    print("\nRecovery by original failure category:")
    print(
        result.groupby("category")["any_within_10ppm"]
        .agg(["count", "sum"])
        .to_string()
    )

    print("\nBest absolute ppm using all H+ spectra:")
    print(
        result["best_abs_ppm"]
        .dropna()
        .describe(
            percentiles=[
                0.25,
                0.5,
                0.75,
                0.9,
                0.95,
                0.99,
            ]
        )
        .to_string()
    )

    print("\nRecovered examples:")
    print(
        recovered.sort_values("best_abs_ppm")
        [
            [
                "query_inchikey14",
                "truth_formula",
                "candidate_formula_count",
                "truth_mass_error_ppm",
                "hplus_spectrum_count",
                "best_abs_ppm",
                "best_source",
                "best_num_peaks",
            ]
        ]
        .head(30)
        .to_string(index=False)
    )

    print("\nStill not recovered:")
    print(
        result[
            result["any_within_10ppm"] != True
        ]
        [
            [
                "query_inchikey14",
                "truth_formula",
                "category",
                "hplus_spectrum_count",
                "truth_mass_error_ppm",
                "best_abs_ppm",
            ]
        ]
        .sort_values(
            "best_abs_ppm",
            na_position="last",
        )
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()