from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

from casmi26.evaluation.strict_validation import (
    resolve_structure_metadata,
)


PROJECT_ROOT = (
    Path(__file__).resolve().parent.parent
)

TRAIN_PATH = (
    PROJECT_ROOT
    / "data/raw/train.parquet"
)

VALIDATION_STRUCTURES_PATH = (
    PROJECT_ROOT
    / "data/processed/splits/validation_structures.parquet"
)

PUBCHEM_CANDIDATES_PATH = (
    PROJECT_ROOT
    / "data/processed/diagnostics/"
    "pubchem_validation_formula_candidates.parquet"
)

PUBCHEM_BURDEN_PATH = (
    PROJECT_ROOT
    / "data/processed/diagnostics/"
    "pubchem_validation_formula_candidates_burden.parquet"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data/processed/diagnostics/"
    "pubchem_mist_benchmark"
)

SAMPLE_PATH = (
    OUTPUT_DIR
    / "sample.parquet"
)

CANDIDATES_PATH = (
    OUTPUT_DIR
    / "candidates.parquet"
)

MGF_PATH = (
    OUTPUT_DIR
    / "benchmark.mgf"
)

LABELS_PATH = (
    OUTPUT_DIR
    / "benchmark_labels.tsv"
)

FORM_TO_SMI_PATH = (
    OUTPUT_DIR
    / "form_to_smi.pkl"
)

SUPPORTED_ADDUCTS = {
    "[M+H]+": "[M+H]+",
    "[M+Na]+": "[M+Na]+",
    "[M+K]+": "[M+K]+",
    "[M+NH4]+": "[M+H3N+H]+",
}

BUCKETS = [
    ("le100", 0, 100),
    ("101to500", 101, 500),
    ("501to1000", 501, 1000),
    ("1001to5000", 1001, 5000),
]


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--per-bucket",
        type=int,
        default=25,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=20261019,
    )

    return parser.parse_args()


def format_peaks(
    mzs,
    intensities,
) -> list[str]:
    mzs = np.asarray(
        mzs,
        dtype=np.float64,
    )

    intensities = np.asarray(
        intensities,
        dtype=np.float64,
    )

    if len(mzs) != len(intensities):
        raise ValueError(
            "m/z and intensity arrays differ in length"
        )

    lines = []

    for mz, intensity in zip(
        mzs,
        intensities,
        strict=True,
    ):
        if not (
            np.isfinite(mz)
            and np.isfinite(intensity)
        ):
            continue

        if intensity <= 0:
            continue

        lines.append(
            f"{mz:.8f} {intensity:.8f}"
        )

    return lines


def load_validation_keys() -> set[str]:
    return set(
        pd.read_parquet(
            VALIDATION_STRUCTURES_PATH,
            columns=["inchikey14"],
        )[
            "inchikey14"
        ]
        .astype(str)
    )


def choose_positive_spectrum(
    group: pd.DataFrame,
) -> pd.Series | None:
    supported = group.loc[
        group[
            "adduct"
        ].isin(
            SUPPORTED_ADDUCTS
        )
        & (
            group[
                "ionization_mode"
            ].astype(str)
            == "positive"
        )
    ].copy()

    if supported.empty:
        return None

    supported[
        "_priority"
    ] = (
        supported[
            "adduct"
        ]
        .map(
            {
                "[M+H]+": 0,
                "[M+Na]+": 1,
                "[M+NH4]+": 2,
                "[M+K]+": 3,
            }
        )
        .fillna(99)
    )

    supported = supported.sort_values(
        [
            "_priority",
        ]
    )

    return supported.iloc[0]


def load_candidates_for_formulas(
    formulas: list[str],
) -> pd.DataFrame:
    dataset = ds.dataset(
        PUBCHEM_CANDIDATES_PATH,
        format="parquet",
    )

    table = dataset.to_table(
        columns=[
            "molecular_formula",
            "pubchem_cid",
            "full_inchikey",
            "inchikey14",
            "smiles",
        ],
        filter=ds.field(
            "molecular_formula"
        ).isin(
            formulas
        ),
    )

    return (
        table.to_pandas()
        .drop_duplicates(
            subset=[
                "molecular_formula",
                "inchikey14",
            ]
        )
        .reset_index(drop=True)
    )


def main() -> None:
    args = parse_args()

    if args.per_bucket <= 0:
        raise ValueError(
            "--per-bucket must be positive"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    validation_keys = (
        load_validation_keys()
    )

    train = pd.read_parquet(
        TRAIN_PATH,
        columns=[
            "inchikey14",
            "spectrum_id",
            "precursor_mz",
            "adduct",
            "ionization_mode",
            "ms2_mzs",
            "ms2_normalized_intensities",
            "normalized_smiles",
            "molecular_formula",
        ],
    )

    train[
        "inchikey14"
    ] = (
        train[
            "inchikey14"
        ].astype(str)
    )

    validation = (
        train.loc[
            train[
                "inchikey14"
            ].isin(
                validation_keys
            )
        ]
        .copy()
    )

    truth = resolve_structure_metadata(
        validation,
        smiles_column="normalized_smiles",
    ).rename(
        columns={
            "normalized_smiles":
                "truth_smiles",
            "molecular_formula":
                "truth_formula",
        }
    )

    burden = pd.read_parquet(
        PUBCHEM_BURDEN_PATH,
        columns=[
            "molecular_formula",
            "candidate_count",
            "hit_max_records",
            "fetch_failed",
        ],
    ).rename(
        columns={
            "molecular_formula":
                "truth_formula",
        }
    )

    truth = truth.merge(
        burden,
        on="truth_formula",
        how="left",
        validate="many_to_one",
    )

    truth = truth.loc[
        ~truth[
            "fetch_failed"
        ].fillna(False)
    ].copy()

    #
    # Pick one checkpoint-supported positive spectrum per structure.
    #
    spectrum_rows = []

    grouped = validation.groupby(
        "inchikey14",
        sort=False,
    )

    for inchikey14 in truth[
        "inchikey14"
    ]:
        if inchikey14 not in grouped.groups:
            continue

        selected = choose_positive_spectrum(
            grouped.get_group(
                inchikey14
            )
        )

        if selected is None:
            continue

        peaks = format_peaks(
            selected.ms2_mzs,
            selected.ms2_normalized_intensities,
        )

        if not peaks:
            continue

        spectrum_rows.append(
            {
                "inchikey14":
                    inchikey14,
                "spectrum_id":
                    f"validation_{inchikey14}",
                "precursor_mz":
                    float(
                        selected.precursor_mz
                    ),
                "adduct":
                    str(
                        selected.adduct
                    ),
                "mist_ionization":
                    SUPPORTED_ADDUCTS[
                        str(
                            selected.adduct
                        )
                    ],
                "ms2_mzs":
                    selected.ms2_mzs,
                "ms2_normalized_intensities":
                    selected.ms2_normalized_intensities,
            }
        )

    spectra = pd.DataFrame(
        spectrum_rows
    )

    eligible = truth.merge(
        spectra,
        on="inchikey14",
        how="inner",
        validate="one_to_one",
    )

    #
    # Confirm truth presence in the actual capped formula-candidate file.
    #
    eligible_formulas = sorted(
        eligible[
            "truth_formula"
        ].astype(str).unique()
    )

    eligible_candidates = (
        load_candidates_for_formulas(
            eligible_formulas
        )
    )

    candidate_pairs = set(
        zip(
            eligible_candidates[
                "molecular_formula"
            ].astype(str),
            eligible_candidates[
                "inchikey14"
            ].astype(str),
            strict=True,
        )
    )

    eligible[
        "truth_candidate_present"
    ] = [
        (
            str(formula),
            str(inchikey14),
        )
        in candidate_pairs
        for formula, inchikey14
        in zip(
            eligible[
                "truth_formula"
            ],
            eligible[
                "inchikey14"
            ],
            strict=True,
        )
    ]

    eligible = eligible.loc[
        eligible[
            "truth_candidate_present"
        ]
    ].copy()

    rng = np.random.default_rng(
        args.seed
    )

    sample_parts = []

    print(
        "Eligible positive truth-present structures:",
        f"{len(eligible):,}",
    )

    for (
        bucket_name,
        low,
        high,
    ) in BUCKETS:
        bucket = eligible.loc[
            (
                eligible[
                    "candidate_count"
                ]
                >= low
            )
            & (
                eligible[
                    "candidate_count"
                ]
                <= high
            )
        ].copy()

        take = min(
            args.per_bucket,
            len(bucket),
        )

        if take:
            indices = rng.choice(
                bucket.index.to_numpy(),
                size=take,
                replace=False,
            )

            selected = (
                bucket.loc[
                    indices
                ]
                .copy()
            )

            selected[
                "burden_bucket"
            ] = bucket_name

            sample_parts.append(
                selected
            )

        print(
            f"{bucket_name:>10}: "
            f"eligible {len(bucket):,}, "
            f"selected {take:,}"
        )

    if not sample_parts:
        raise RuntimeError(
            "No benchmark molecules were selected"
        )

    sample = (
        pd.concat(
            sample_parts,
            ignore_index=True,
        )
        .sort_values(
            [
                "burden_bucket",
                "inchikey14",
            ]
        )
        .reset_index(drop=True)
    )

    sample[
        "spec"
    ] = [
        f"benchmark_{i:04d}"
        for i in range(
            len(sample)
        )
    ]

    selected_formulas = sorted(
        sample[
            "truth_formula"
        ].astype(str).unique()
    )

    candidates = (
        load_candidates_for_formulas(
            selected_formulas
        )
    )

    selected_pairs = (
        sample[
            [
                "spec",
                "truth_formula",
            ]
        ]
        .drop_duplicates()
    )

    benchmark_candidates = (
        selected_pairs.merge(
            candidates,
            left_on="truth_formula",
            right_on="molecular_formula",
            how="inner",
            validate="many_to_many",
        )
        .drop(
            columns=[
                "molecular_formula",
            ]
        )
    )

    benchmark_candidates.to_parquet(
        CANDIDATES_PATH,
        index=False,
    )

    sample.to_parquet(
        SAMPLE_PATH,
        index=False,
    )

    mgf_blocks = []

    label_rows = []

    for row in sample.itertuples(
        index=False
    ):
        peak_lines = format_peaks(
            row.ms2_mzs,
            row.ms2_normalized_intensities,
        )

        block = [
            "BEGIN IONS",
            f"TITLE={row.spec}",
            (
                "PEPMASS="
                f"{float(row.precursor_mz):.8f}"
            ),
            "IONMODE=Positive",
            *peak_lines,
            "END IONS",
        ]

        mgf_blocks.append(
            "\n".join(
                block
            )
        )

        label_rows.append(
            {
                "spec":
                    row.spec,
                "formula":
                    row.truth_formula,
                "ionization":
                    row.mist_ionization,
            }
        )

    MGF_PATH.write_text(
        "\n\n".join(
            mgf_blocks
        )
        + "\n"
    )

    pd.DataFrame(
        label_rows
    ).to_csv(
        LABELS_PATH,
        sep="\t",
        index=False,
    )

    formula_structures = (
        candidates[
            [
                "molecular_formula",
                "inchikey14",
                "smiles",
            ]
        ]
        .drop_duplicates(
            subset=[
                "molecular_formula",
                "inchikey14",
            ]
        )
    )

    form_to_smi = {}

    for formula, group in (
        formula_structures.groupby(
            "molecular_formula",
            sort=False,
        )
    ):
        form_to_smi[
            str(formula)
        ] = [
            {
                "smi":
                    str(row.smiles),
                "ikey":
                    str(row.inchikey14),
            }
            for row in group.itertuples(
                index=False
            )
        ]

    with FORM_TO_SMI_PATH.open(
        "wb"
    ) as handle:
        pickle.dump(
            form_to_smi,
            handle,
        )

    print()
    print(
        "Strict PubChem/MIST benchmark prepared"
    )
    print(
        "======================================"
    )
    print(
        "Molecules:",
        len(sample),
    )
    print(
        "Unique formulas:",
        len(
            selected_formulas
        ),
    )
    print(
        "Candidate rows:",
        len(
            benchmark_candidates
        ),
    )
    print(
        "Unique candidate structures:",
        benchmark_candidates[
            "inchikey14"
        ].nunique(),
    )
    print()

    counts = (
        sample.groupby(
            "burden_bucket"
        )
        .size()
    )

    print(
        counts.to_string()
    )

    print()
    print(
        f"Wrote {SAMPLE_PATH}"
    )
    print(
        f"Wrote {CANDIDATES_PATH}"
    )
    print(
        f"Wrote {MGF_PATH}"
    )
    print(
        f"Wrote {LABELS_PATH}"
    )
    print(
        f"Wrote {FORM_TO_SMI_PATH}"
    )


if __name__ == "__main__":
    main()
