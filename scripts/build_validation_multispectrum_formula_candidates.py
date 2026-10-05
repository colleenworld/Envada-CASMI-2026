from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


ROOT = Path("data/processed/mist/validation_500")

MANIFEST_PATH = ROOT / "manifest.parquet"
TRAIN_PATH = Path("data/raw/train.parquet")
FORMULA_INDEX_PATH = Path(
    "data/processed/formula_index/training_coconut_formulas.parquet"
)

OUTPUT_PATH = ROOT / "multispectrum_formula_candidates.parquet"

PROTON_MASS = 1.007276466621
PPM = 10.0


def main():
    manifest = pd.read_parquet(MANIFEST_PATH)

    query_ids = set(manifest["query_inchikey14"])

    formulas = pd.read_parquet(FORMULA_INDEX_PATH)

    formula_masses = formulas["monoisotopic_mass"].to_numpy()
    order = np.argsort(formula_masses)

    masses_sorted = formula_masses[order]
    formulas_sorted = formulas.iloc[order].reset_index(drop=True)

    # Collect every H+ spectrum for the frozen 500 query structures.
    parquet = pq.ParquetFile(TRAIN_PATH)

    columns = [
        "inchikey14",
        "ingest_lib",
        "adduct",
        "precursor_mz",
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

    print("Queries:", len(manifest))
    print("H+ spectra:", len(spectra))
    print(
        "Queries with H+ spectra:",
        spectra["inchikey14"].nunique(),
    )

    candidate_rows = []

    for query_key, group in spectra.groupby(
        "inchikey14",
        sort=False,
    ):
        spectrum_count = len(group)

        # formula -> list of ppm errors from spectra supporting it
        support = {}

        for row in group.itertuples(index=False):
            neutral_mass = row.precursor_mz - PROTON_MASS

            tol = neutral_mass * PPM / 1e6

            lo = np.searchsorted(
                masses_sorted,
                neutral_mass - tol,
                side="left",
            )
            hi = np.searchsorted(
                masses_sorted,
                neutral_mass + tol,
                side="right",
            )

            candidates = formulas_sorted.iloc[lo:hi]

            for cand in candidates.itertuples(index=False):
                ppm_error = (
                    (cand.monoisotopic_mass - neutral_mass)
                    / neutral_mass
                    * 1e6
                )

                support.setdefault(
                    cand.molecular_formula,
                    [],
                ).append(ppm_error)

        for formula, ppm_errors in support.items():
            ppm_errors = np.asarray(
                ppm_errors,
                dtype=float,
            )

            candidate_rows.append(
                {
                    "query_inchikey14": query_key,
                    "candidate_formula": formula,
                    "spectrum_count": spectrum_count,
                    "formula_support_count": len(ppm_errors),
                    "formula_support_fraction":
                        len(ppm_errors) / spectrum_count,
                    "mass_error_best_abs_ppm":
                        np.abs(ppm_errors).min(),
                    "mass_error_median_abs_ppm":
                        np.median(np.abs(ppm_errors)),
                    "mass_error_mean_abs_ppm":
                        np.mean(np.abs(ppm_errors)),
                    "mass_error_signed_median_ppm":
                        np.median(ppm_errors),
                }
            )

    candidates = pd.DataFrame(candidate_rows)

    truth = manifest[
        [
            "query_inchikey14",
            "truth_formula",
        ]
    ].drop_duplicates()

    candidates = candidates.merge(
        truth,
        on="query_inchikey14",
        how="left",
    )

    candidates["is_truth_formula"] = (
        candidates["candidate_formula"]
        == candidates["truth_formula"]
    )

    candidates = candidates.sort_values(
        [
            "query_inchikey14",
            "formula_support_count",
            "mass_error_best_abs_ppm",
            "candidate_formula",
        ],
        ascending=[True, False, True, True],
        kind="stable",
    ).reset_index(drop=True)

    candidates.to_parquet(
        OUTPUT_PATH,
        index=False,
    )

    per_query = (
        candidates.groupby("query_inchikey14")
        .size()
        .reindex(
            manifest["query_inchikey14"],
            fill_value=0,
        )
    )

    truth_recovered = (
        candidates.loc[
            candidates["is_truth_formula"],
            "query_inchikey14",
        ]
        .nunique()
    )

    print("\nMulti-spectrum formula candidates")
    print("---------------------------------")
    print("Rows:", len(candidates))
    print(
        "Queries with >=1 formula candidate:",
        int((per_query > 0).sum()),
        "/",
        len(manifest),
    )
    print(
        "Queries with truth formula:",
        truth_recovered,
        "/",
        len(manifest),
    )

    print("\nCandidate formulas/query:")
    print(
        per_query.describe(
            percentiles=[
                0.5,
                0.75,
                0.9,
                0.95,
                0.99,
            ]
        ).to_string()
    )

    print("\nTruth formula support:")
    truth_rows = candidates[
        candidates["is_truth_formula"]
    ]

    if not truth_rows.empty:
        print(
            truth_rows[
                "formula_support_fraction"
            ]
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

    print(f"\nWrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()