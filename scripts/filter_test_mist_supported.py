from __future__ import annotations

import os
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(
    os.environ.get(
        "CASMI26_PROJECT_ROOT",
        Path(__file__).resolve().parent.parent,
    )
)

TEST_ROOT = PROJECT_ROOT / "data/processed/test"

HYPOTHESES_PATH = Path(
    os.environ.get(
        "CASMI26_FORMULA_HYPOTHESES_PATH",
        TEST_ROOT / "multispectrum_formula_hypotheses.parquet",
    )
)

LABELS_PATH = TEST_ROOT / "mist/test_multiformula_labels.tsv"
OUTPUT_PATH = TEST_ROOT / "mist/test_multiformula_labels_mist_supported.tsv"

SUPPORTED_ADDUCTS = {
    "[M+H]+",
    "[M+Na]+",
    "[M+NH4]+",
    "[M+K]+",
}


def main() -> None:
    hypotheses = pd.read_parquet(HYPOTHESES_PATH)
    labels = pd.read_csv(LABELS_PATH, sep="\t")

    supported_specs = set(
        hypotheses.loc[
            hypotheses["selected_adduct"].astype(str).isin(
                SUPPORTED_ADDUCTS
            ),
            "hypothesis_spec",
        ].astype(str)
    )

    filtered = labels.loc[
        labels["spec"].astype(str).isin(supported_specs)
    ].copy()

    filtered["ionization"] = filtered["ionization"].replace(
        {"[M+NH4]+": "[M+H3N+H]+"}
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    filtered.to_csv(OUTPUT_PATH, sep="\t", index=False)

    supported_molecules = hypotheses.loc[
        hypotheses["hypothesis_spec"].astype(str).isin(
            set(filtered["spec"].astype(str))
        ),
        "molecule_id",
    ].nunique()

    print("MIST-supported hypotheses:", len(filtered))
    print("Molecules represented:", supported_molecules)
    print("Ionization modes:")
    print(filtered["ionization"].value_counts().to_string())
    print("Wrote:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
