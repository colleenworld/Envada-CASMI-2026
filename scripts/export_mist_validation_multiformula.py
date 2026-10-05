from pathlib import Path
import re

import numpy as np
import pandas as pd


ROOT = Path("data/processed/mist/validation_500")

MANIFEST_PATH = ROOT / "manifest.parquet"
FORMULA_INDEX_PATH = Path(
    "data/processed/formula_index/training_coconut_formulas.parquet"
)

LABELS_PATH = ROOT / "mist_multiformula_labels.tsv"
MAPPING_PATH = ROOT / "mist_multiformula_mapping.parquet"

PPM = 10.0
PROTON_MASS = 1.007276466621


def formula_token(formula: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", formula)


def main():
    manifest = pd.read_parquet(MANIFEST_PATH)
    formulas = pd.read_parquet(FORMULA_INDEX_PATH)

    masses = formulas["monoisotopic_mass"].to_numpy()
    order = np.argsort(masses)

    masses_sorted = masses[order]
    formulas_sorted = formulas.iloc[order].reset_index(drop=True)

    label_rows = []
    mapping_rows = []

    for row in manifest.itertuples(index=False):
        if row.adduct != "[M+H]+":
            raise ValueError(
                f"Unexpected adduct {row.adduct!r} for {row.spec}"
            )

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

        candidates = formulas_sorted.iloc[lo:hi].copy()

        candidates["abs_mass_error"] = (
            candidates["monoisotopic_mass"] - neutral_mass
        ).abs()

        candidates = candidates.sort_values(
            ["abs_mass_error", "molecular_formula"],
            kind="stable",
        ).reset_index(drop=True)

        for formula_idx, cand in candidates.iterrows():
            candidate_formula = cand["molecular_formula"]

            hypothesis_spec = (
                f"{row.spec}__f{formula_idx:03d}_"
                f"{formula_token(candidate_formula)}"
            )

            ppm_error = (
                (cand["monoisotopic_mass"] - neutral_mass)
                / neutral_mass
                * 1e6
            )

            label_rows.append(
                {
                    "spec": hypothesis_spec,
                    "formula": candidate_formula,
                    "ionization": row.adduct,
                    "dataset": "casmi_validation_500",
                    "compound": hypothesis_spec,
                    "parentmass": row.precursor_mz,
                    "instrument": row.instrument_type,
                }
            )

            mapping_rows.append(
                {
                    "hypothesis_spec": hypothesis_spec,
                    "original_spec": row.spec,
                    "query_inchikey14": row.query_inchikey14,
                    "truth_formula": row.truth_formula,
                    "candidate_formula": candidate_formula,
                    "candidate_formula_mass":
                        cand["monoisotopic_mass"],
                    "neutral_mass": neutral_mass,
                    "mass_error_ppm": ppm_error,
                    "is_truth_formula":
                        candidate_formula == row.truth_formula,
                }
            )

    labels = pd.DataFrame(label_rows)
    mapping = pd.DataFrame(mapping_rows)

    labels.to_csv(
        LABELS_PATH,
        sep="\t",
        index=False,
    )

    mapping.to_parquet(
        MAPPING_PATH,
        index=False,
    )

    per_query = (
        mapping.groupby("original_spec")
        .size()
        .reindex(manifest["spec"], fill_value=0)
    )

    truth_available = (
        mapping.groupby("original_spec")["is_truth_formula"]
        .any()
        .reindex(manifest["spec"], fill_value=False)
    )

    print("Queries:", len(manifest))
    print("Hypotheses:", len(mapping))
    print(
        "Queries with >=1 formula candidate:",
        int((per_query > 0).sum()),
    )
    print(
        "Queries with truth formula:",
        int(truth_available.sum()),
        "/",
        len(manifest),
    )

    print("\nFormula hypotheses/query:")
    print(
        per_query.describe(
            percentiles=[0.5, 0.75, 0.9, 0.95, 0.99]
        ).to_string()
    )

    print(f"\nWrote {LABELS_PATH}")
    print(f"Wrote {MAPPING_PATH}")


if __name__ == "__main__":
    main()