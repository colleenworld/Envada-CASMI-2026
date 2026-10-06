from pathlib import Path
import re

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


ROOT = Path("data/processed/mist/validation_500_v2")

MANIFEST_PATH = ROOT / "manifest.parquet"

FORMULA_CANDIDATES_PATH = (
    ROOT / "multispectrum_formula_candidates.parquet"
)

FORMULA_INDEX_PATH = Path(
    "data/processed/formula_index/training_coconut_formulas.parquet"
)

TRAIN_PATH = Path("data/raw/train.parquet")

OUTPUT_PATH = ROOT / "mist_multispectrum_hypotheses.parquet"
LABELS_PATH = ROOT / "mist_multispectrum_labels.tsv"

PROTON_MASS = 1.007276466621

def is_mist_processable(row) -> bool:
    mzs = np.asarray(row["ms2_mzs"], dtype=float)
    intensities = np.asarray(
        row["ms2_normalized_intensities"],
        dtype=float,
    )

    if mzs.size == 0 or intensities.size == 0:
        return False

    if mzs.size != intensities.size:
        return False

    finite = (
        np.isfinite(mzs)
        & np.isfinite(intensities)
    )

    if not finite.any():
        return False

    mzs = mzs[finite]
    intensities = intensities[finite]

    #
    # Mirrors process_spec_file():
    #
    # merged_spec = merged_spec[
    #     merged_spec[:, 0] <= parentmass + 1
    # ]
    #
    usable = (
        mzs
        <= float(row["precursor_mz"]) + 1.0
    )

    if not usable.any():
        return False

    #
    # Avoid an all-zero spectrum as well.
    #
    return (
        np.max(intensities[usable]) > 0
    )

def formula_token(formula: str) -> str:
    return re.sub(
        r"[^A-Za-z0-9]+",
        "_",
        formula,
    )


def main():
    manifest = pd.read_parquet(MANIFEST_PATH)

    formula_candidates = pd.read_parquet(
        FORMULA_CANDIDATES_PATH
    )

    formula_index = pd.read_parquet(
        FORMULA_INDEX_PATH
    )

    query_ids = set(
        manifest["query_inchikey14"]
    )

    #
    # Formula -> monoisotopic mass.
    #
    formula_mass = dict(
        zip(
            formula_index["molecular_formula"],
            formula_index["monoisotopic_mass"],
        )
    )

    #
    # Read every [M+H]+ spectrum for the full frozen set of
    # 500 v2 queries.
    #
    # IMPORTANT:
    # train_row_id is assigned from the absolute row position
    # in train.parquet BEFORE filtering. This makes it stable
    # across scripts and avoids the earlier positional-ID bug.
    #
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

        df["train_row_id"] = np.arange(
            row_offset,
            row_offset + len(df),
            dtype=np.int64,
        )

        row_offset += len(df)

        mask = (
            df["inchikey14"].isin(query_ids)
            & (df["adduct"] == "[M+H]+")
        )

        if mask.any():
            chunks.append(
                df.loc[mask].copy()
            )

    if not chunks:
        raise RuntimeError(
            "No [M+H]+ spectra found for validation v2 queries."
        )

    spectra = pd.concat(
        chunks,
        ignore_index=True,
    ).reset_index(drop=True)

    spectra["neutral_mass"] = (
        spectra["precursor_mz"]
        - PROTON_MASS
    )

    print("Queries:", len(manifest))
    print("H+ spectra:", len(spectra))
    print(
        "Formula hypotheses:",
        len(formula_candidates),
    )

    #
    # Group spectra once for efficient lookup.
    #
    spectra_by_query = {
        key: group
        for key, group in spectra.groupby(
            "inchikey14",
            sort=False,
        )
    }

    rows = []

    for cand in formula_candidates.itertuples(
        index=False
    ):
        formula = cand.candidate_formula

        theoretical_mass = formula_mass.get(
            formula
        )

        if theoretical_mass is None:
            raise ValueError(
                f"Formula missing from index: {formula}"
            )

        group = spectra_by_query.get(
            cand.query_inchikey14
        )

        if group is None or group.empty:
            raise ValueError(
                "No [M+H]+ spectra for "
                f"{cand.query_inchikey14}"
            )

        #
        # Formula generation used all [M+H]+ spectra.
        #
        # MIST, however, cannot use the known failure case where
        # a one-peak spectrum becomes empty during preprocessing.
        #
        # Therefore select the smallest-|ppm| spectrum only from
        # spectra containing >1 raw MS/MS peak.
        #
        processable_mask = group.apply(
            is_mist_processable,
            axis=1,
        )

        processable = group[
            processable_mask
        ].copy()

        if processable.empty:
            raise ValueError(
                "No MIST-processable [M+H]+ spectrum for "
                f"{cand.query_inchikey14}"
            )

        ppm_errors = (
            (
                theoretical_mass
                - processable["neutral_mass"]
            )
            / processable["neutral_mass"]
            * 1e6
        )

        best_idx = ppm_errors.abs().idxmin()

        best = processable.loc[best_idx]

        best_ppm = float(
            ppm_errors.loc[best_idx]
        )

        hypothesis_spec = (
            f"mistval2_"
            f"{cand.query_inchikey14}"
            f"__f_{formula_token(formula)}"
        )

        rows.append(
            {
                "hypothesis_spec":
                    hypothesis_spec,

                "query_inchikey14":
                    cand.query_inchikey14,

                "truth_formula":
                    cand.truth_formula,

                "candidate_formula":
                    formula,

                "is_truth_formula":
                    bool(cand.is_truth_formula),

                #
                # Multi-spectrum formula evidence.
                #
                "spectrum_count":
                    cand.spectrum_count,

                "formula_support_count":
                    cand.formula_support_count,

                "formula_support_fraction":
                    cand.formula_support_fraction,

                "mass_error_best_abs_ppm":
                    cand.mass_error_best_abs_ppm,

                "mass_error_median_abs_ppm":
                    cand.mass_error_median_abs_ppm,

                "mass_error_mean_abs_ppm":
                    cand.mass_error_mean_abs_ppm,

                "mass_error_signed_median_ppm":
                    cand.mass_error_signed_median_ppm,

                #
                # Stable source-spectrum identity.
                #
                "source_train_row_id":
                    int(best.train_row_id),

                #
                # Selected spectrum metadata.
                #
                "source_ingest_lib":
                    best.ingest_lib,

                "source_precursor_mz":
                    float(best.precursor_mz),

                "source_instrument_type":
                    best.instrument_type,

                "source_raw_peak_count":
                    len(best.ms2_mzs),

                "source_mist_usable_peak_count":
                    int(
                        np.sum(
                            np.asarray(
                                best.ms2_mzs,
                                dtype=float,
                            )
                            <= float(best.precursor_mz) + 1.0
                        )
                    ),

                #
                # Formula error for the specific spectrum that
                # will actually be supplied to MIST.
                #
                "selected_mass_error_ppm":
                    best_ppm,
            }
        )

    hypotheses = pd.DataFrame(rows)

    if (
            hypotheses[
                "source_mist_usable_peak_count"
            ] <= 0
    ).any():
        raise ValueError(
            "Selected spectrum with no "
            "MIST-usable peaks."
        )

    if hypotheses["hypothesis_spec"].duplicated().any():
        duplicates = (
            hypotheses.loc[
                hypotheses[
                    "hypothesis_spec"
                ].duplicated(keep=False),
                "hypothesis_spec",
            ]
            .unique()
            .tolist()
        )

        raise ValueError(
            f"Duplicate hypothesis IDs: {duplicates[:10]}"
        )

    hypotheses.to_parquet(
        OUTPUT_PATH,
        index=False,
    )


    #
    # MIST labels.
    #
    labels = pd.DataFrame({
        "spec": hypotheses["hypothesis_spec"],
        "formula": hypotheses["candidate_formula"],
        "ionization": "[M+H]+",
        "dataset": "casmi_validation_500_v2_multispectrum",
        "compound": hypotheses["hypothesis_spec"],
        "parentmass": hypotheses["source_precursor_mz"],
        "instrument": hypotheses["source_instrument_type"],
    })

    labels.to_csv(
        LABELS_PATH,
        sep="\t",
        index=False,
    )

    #
    # Diagnostics.
    #
    print("\nMIST multi-spectrum hypotheses")
    print("-----------------------------")

    print(
        "Hypotheses:",
        len(hypotheses),
    )

    print(
        "Queries:",
        hypotheses[
            "query_inchikey14"
        ].nunique(),
        "/ 500",
    )

    print(
        "Truth formula hypotheses:",
        int(
            hypotheses[
                "is_truth_formula"
            ].sum()
        ),
        "/ 500",
    )

    print(
        "Unique source spectra selected:",
        hypotheses[
            "source_train_row_id"
        ].nunique(),
    )

    print(
        "\nSelected raw peak counts:"
    )

    print(
        hypotheses[
            "source_raw_peak_count"
        ]
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

    print(
        "\nSelected spectrum mass error:"
    )

    print(
        hypotheses[
            "selected_mass_error_ppm"
        ]
        .abs()
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

    #
    # mass_error_best_abs_ppm was computed using all spectra.
    # selected_mass_error_ppm uses only MIST-processable spectra.
    #
    # Therefore selected error can legitimately be larger.
    #
    delta = (
        hypotheses[
            "selected_mass_error_ppm"
        ].abs()
        - hypotheses[
            "mass_error_best_abs_ppm"
        ]
    )

    print(
        "\nHypotheses where processability changed "
        "the selected mass error:",
        int((delta > 1e-9).sum()),
    )

    print(
        "Largest extra ppm from choosing a "
        "processable spectrum:",
        float(delta.max()),
    )

    print(
        "\nSelected spectra with <=1 raw peak:",
        int(
            (
                hypotheses[
                    "source_raw_peak_count"
                ] <= 1
            ).sum()
        ),
    )

    print(f"\nWrote {OUTPUT_PATH}")
    print(f"Wrote {LABELS_PATH}")


if __name__ == "__main__":
    main()