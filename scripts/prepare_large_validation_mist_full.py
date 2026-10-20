from __future__ import annotations

import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from casmi26.pubchem.inference_asset import load_pubchem_candidates_for_formulas


SUPPORTED_ADDUCTS = {
    "[M+H]+": "[M+H]+",
    "[M+Na]+": "[M+Na]+",
    "[M+K]+": "[M+K]+",
    "[M+NH4]+": "[M+H3N+H]+",
}


def format_peaks(mzs, intensities) -> list[str]:
    mzs = np.asarray(mzs, dtype=float)
    intensities = np.asarray(intensities, dtype=float)

    if len(mzs) != len(intensities):
        return []

    lines = []
    for mz, intensity in zip(mzs, intensities, strict=True):
        if (
            np.isfinite(mz)
            and np.isfinite(intensity)
            and intensity > 0
        ):
            lines.append(
                f"{float(mz):.8f} {float(intensity):.8f}"
            )

    return lines


def main() -> None:
    project_root = Path(
        os.environ.get(
            "CASMI26_PROJECT_ROOT",
            Path(__file__).resolve().parent.parent,
        )
    )

    benchmark_dir = (
        project_root
        / "data/processed/diagnostics/large_validation_audit/"
        "true_formula_neighbor_benchmark"
    )

    sample_path = benchmark_dir / "sample.parquet"
    train_path = project_root / "data/raw/train.parquet"
    pubchem_root = (
        project_root
        / "data/processed/pubchem_formula_index"
    )

    output_dir = benchmark_dir / "mist_full"
    output_dir.mkdir(parents=True, exist_ok=True)

    mist_sample_path = output_dir / "sample.parquet"
    mgf_path = output_dir / "benchmark.mgf"
    labels_path = output_dir / "labels.tsv"
    form_to_smi_path = output_dir / "form_to_smi.pkl"

    sample = pd.read_parquet(sample_path)
    selected_keys = set(
        sample["inchikey14"].astype(str)
    )

    parquet = pq.ParquetFile(train_path)

    columns = [
        "inchikey14",
        "precursor_mz",
        "adduct",
        "ionization_mode",
        "ms2_mzs",
        "ms2_normalized_intensities",
    ]

    chunks = []

    for batch in parquet.iter_batches(
        batch_size=100_000,
        columns=columns,
    ):
        frame = batch.to_pandas()

        matched = frame.loc[
            frame["inchikey14"].astype(str).isin(selected_keys)
            & frame["adduct"].isin(SUPPORTED_ADDUCTS)
            & (
                frame["ionization_mode"].astype(str)
                == "positive"
            )
        ].copy()

        if not matched.empty:
            chunks.append(matched)

    if not chunks:
        raise RuntimeError(
            "No MIST-supported positive spectra found"
        )

    spectra = pd.concat(
        chunks,
        ignore_index=True,
    )

    spectrum_rows = []

    for structure_key, group in spectra.groupby(
        "inchikey14",
        sort=False,
    ):
        candidates = []

        for row in group.itertuples(index=False):
            peak_lines = format_peaks(
                row.ms2_mzs,
                row.ms2_normalized_intensities,
            )

            if not peak_lines:
                continue

            candidates.append(
                (
                    len(peak_lines),
                    float(row.precursor_mz),
                    str(row.adduct),
                    row,
                    peak_lines,
                )
            )

        if not candidates:
            continue

        candidates.sort(
            key=lambda item: (
                -item[0],
                item[1],
                item[2],
            )
        )

        _, _, _, row, peak_lines = candidates[0]

        spectrum_rows.append(
            {
                "inchikey14": str(structure_key),
                "precursor_mz": float(row.precursor_mz),
                "adduct": str(row.adduct),
                "mist_ionization": SUPPORTED_ADDUCTS[
                    str(row.adduct)
                ],
                "peak_lines": peak_lines,
            }
        )

    selected_spectra = pd.DataFrame(
        spectrum_rows
    )

    mist_sample = sample.merge(
        selected_spectra,
        on="inchikey14",
        how="inner",
        validate="one_to_one",
    )

    mist_sample = mist_sample.reset_index(drop=True)

    mist_sample["spec"] = [
        f"large_mist_{index:05d}"
        for index in range(len(mist_sample))
    ]

    formulas = sorted(
        set(
            mist_sample[
                "truth_formula"
            ].astype(str)
        )
    )

    candidates = load_pubchem_candidates_for_formulas(
        pubchem_root,
        formulas,
    )

    candidate_pairs = set(
        zip(
            candidates["molecular_formula"].astype(str),
            candidates["inchikey14"].astype(str),
            strict=True,
        )
    )

    truth_present = [
        (
            str(row.truth_formula),
            str(row.inchikey14),
        )
        in candidate_pairs
        for row in mist_sample.itertuples(index=False)
    ]

    mist_sample = mist_sample.loc[
        truth_present
    ].reset_index(drop=True)

    formulas = sorted(
        mist_sample[
            "truth_formula"
        ].astype(str)
        .unique()
    )

    candidates = candidates.loc[
        candidates[
            "molecular_formula"
        ]
        .astype(str)
        .isin(formulas)
    ].copy()

    form_to_smi = {}

    for formula, group in candidates.groupby(
        "molecular_formula",
        sort=False,
    ):
        form_to_smi[str(formula)] = [
            {
                "smi": str(row.smiles),
                "ikey": str(row.inchikey14),
            }
            for row in group.itertuples(index=False)
        ]

    with form_to_smi_path.open("wb") as handle:
        pickle.dump(
            form_to_smi,
            handle,
        )

    labels = []

    with mgf_path.open("w") as handle:
        for row in mist_sample.itertuples(index=False):
            handle.write("BEGIN IONS\n")
            handle.write(f"TITLE={row.spec}\n")
            handle.write(f"FEATURE_ID={row.spec}\n")
            handle.write(
                f"PEPMASS={float(row.precursor_mz):.8f}\n"
            )
            handle.write("CHARGE=1+\n")
            handle.write(
                f"FORMULA={row.truth_formula}\n"
            )
            handle.write(
                f"ADDUCT={row.mist_ionization}\n"
            )
            handle.write("IONMODE=positive\n")

            for peak_line in row.peak_lines:
                handle.write(peak_line + "\n")

            handle.write("END IONS\n\n")

            labels.append(
                {
                    "spec": row.spec,
                    "formula": str(row.truth_formula),
                    "ionization": str(
                        row.mist_ionization
                    ),
                }
            )

    pd.DataFrame(labels).to_csv(
        labels_path,
        sep="\t",
        index=False,
    )

    mist_sample = mist_sample.drop(
        columns=["peak_lines"]
    )

    mist_sample.to_parquet(
        mist_sample_path,
        index=False,
    )

    print(
        "Large true-formula MIST benchmark prepared"
    )
    print(
        "=========================================="
    )
    print(
        "Original benchmark structures:",
        f"{len(sample):,}",
    )
    print(
        "MIST-supported positive structures:",
        f"{len(mist_sample):,}",
    )
    print(
        "Unique formulas:",
        f"{len(formulas):,}",
    )
    print(
        "PubChem formula/structure candidates:",
        f"{len(candidates):,}",
    )
    print()
    print("Wrote:")
    print(mist_sample_path)
    print(mgf_path)
    print(labels_path)
    print(form_to_smi_path)


if __name__ == "__main__":
    main()
