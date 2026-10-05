from pathlib import Path

import pandas as pd


ROOT = Path("data/processed/mist/validation_500")

MANIFEST_PATH = ROOT / "manifest.parquet"
MAPPING_PATH = ROOT / "mist_multiformula_mapping.parquet"
FORMULA_INDEX_PATH = Path(
    "data/processed/formula_index/training_coconut_formulas.parquet"
)

PROTON_MASS = 1.007276466621


def main():
    manifest = pd.read_parquet(MANIFEST_PATH)
    mapping = pd.read_parquet(MAPPING_PATH)
    formulas = pd.read_parquet(FORMULA_INDEX_PATH)

    formula_mass = dict(
        zip(
            formulas["molecular_formula"],
            formulas["monoisotopic_mass"],
        )
    )

    candidate_counts = (
        mapping.groupby("original_spec")
        .size()
        .rename("candidate_formula_count")
    )

    truth_available = (
        mapping.groupby("original_spec")["is_truth_formula"]
        .any()
        .rename("truth_formula_available")
    )

    rows = []

    for row in manifest.itertuples(index=False):
        neutral_mass = row.precursor_mz - PROTON_MASS

        truth_mass = formula_mass.get(row.truth_formula)

        if truth_mass is None:
            truth_ppm = None
            truth_in_index = False
        else:
            truth_ppm = (
                (truth_mass - neutral_mass)
                / neutral_mass
                * 1e6
            )
            truth_in_index = True

        count = int(candidate_counts.get(row.spec, 0))
        available = bool(truth_available.get(row.spec, False))

        if available:
            category = "truth_available"
        elif not truth_in_index:
            category = "truth_formula_not_in_index"
        elif count == 0:
            category = "no_formula_candidates"
        elif abs(truth_ppm) > 10:
            category = "truth_outside_10ppm"
        else:
            category = "other"

        rows.append(
            {
                "spec": row.spec,
                "query_inchikey14": row.query_inchikey14,
                "truth_formula": row.truth_formula,
                "precursor_mz": row.precursor_mz,
                "neutral_mass": neutral_mass,
                "truth_formula_mass": truth_mass,
                "truth_mass_error_ppm": truth_ppm,
                "truth_formula_in_index": truth_in_index,
                "candidate_formula_count": count,
                "truth_formula_available": available,
                "category": category,
            }
        )

    out = pd.DataFrame(rows)

    failures = out[
        ~out["truth_formula_available"]
    ].copy()

    print("Formula-stage diagnosis")
    print("-----------------------")
    print("Queries:", len(out))
    print("Truth formula available:", int(out["truth_formula_available"].sum()))
    print("Failures:", len(failures))

    print("\nFailure categories:")
    print(
        failures["category"]
        .value_counts(dropna=False)
        .to_string()
    )

    print("\nTruth mass error among failures with indexed formula:")
    indexed = failures[
        failures["truth_formula_in_index"]
        & failures["truth_mass_error_ppm"].notna()
    ]

    if not indexed.empty:
        print(
            indexed["truth_mass_error_ppm"]
            .abs()
            .describe(
                percentiles=[0.5, 0.75, 0.9, 0.95, 0.99]
            )
            .to_string()
        )

    print("\nLargest truth mass errors:")
    print(
        indexed.assign(
            abs_ppm=indexed["truth_mass_error_ppm"].abs()
        )
        .sort_values("abs_ppm", ascending=False)
        .head(25)
        [
            [
                "query_inchikey14",
                "truth_formula",
                "candidate_formula_count",
                "truth_mass_error_ppm",
                "category",
            ]
        ]
        .to_string(index=False)
    )

    output_path = ROOT / "formula_failure_diagnosis.parquet"
    failures.to_parquet(output_path, index=False)

    print(f"\nWrote {output_path}")


if __name__ == "__main__":
    main()