from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent

BENCHMARK_DIR = (
    PROJECT_ROOT
    / "data/processed/diagnostics/pubchem_mist_benchmark"
)

SAMPLE_PATH = BENCHMARK_DIR / "sample.parquet"
PRUNING_PATH = BENCHMARK_DIR / "neighbor_pruning_ranked.parquet"

OUTPUT_DIR = BENCHMARK_DIR / "pruned"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FORM_TO_SMI_PATH = OUTPUT_DIR / "form_to_smi.pkl"
SELECTED_PATH = OUTPUT_DIR / "selected_candidates.parquet"
FORMULA_SUMMARY_PATH = OUTPUT_DIR / "formula_union_summary.parquet"

SHORTLIST_BY_BUCKET = {
    "le100": None,
    "101to500": None,
    "501to1000": 250,
    "1001to5000": 100,
}


def main() -> None:
    sample = pd.read_parquet(SAMPLE_PATH)
    ranked = pd.read_parquet(PRUNING_PATH)

    required_sample = {
        "spec",
        "truth_formula",
        "burden_bucket",
        "candidate_count",
    }
    missing_sample = required_sample - set(sample.columns)
    if missing_sample:
        raise ValueError(
            "Benchmark sample is missing columns: "
            f"{sorted(missing_sample)}"
        )

    required_ranked = {
        "spec",
        "candidate_inchikey14",
        "candidate_smiles",
        "neighbor_prior_score",
        "prior_rank",
    }
    missing_ranked = required_ranked - set(ranked.columns)
    if missing_ranked:
        raise ValueError(
            "Neighbor-pruning ranking is missing columns: "
            f"{sorted(missing_ranked)}"
        )

    meta = sample[
        [
            "spec",
            "truth_formula",
            "burden_bucket",
            "candidate_count",
        ]
    ].copy()

    unknown_buckets = sorted(
        set(meta["burden_bucket"].astype(str))
        - set(SHORTLIST_BY_BUCKET)
    )
    if unknown_buckets:
        raise ValueError(
            f"Unknown burden buckets: {unknown_buckets}"
        )

    ranked = ranked.merge(
        meta,
        on="spec",
        how="inner",
        validate="many_to_one",
    )

    selected_parts = []

    for spec, group in ranked.groupby("spec", sort=False):
        bucket = str(group["burden_bucket"].iloc[0])
        limit = SHORTLIST_BY_BUCKET[bucket]

        group = group.sort_values(
            ["prior_rank", "candidate_inchikey14"],
            ascending=[True, True],
        )

        if limit is not None:
            group = group.head(limit)

        selected_parts.append(group)

    selected = pd.concat(selected_parts, ignore_index=True)

    #
    # MIST's retrieval database is keyed by molecular formula, not by
    # spectrum. If two benchmark spectra share a formula, their
    # spectrum-specific shortlists cannot be represented separately in
    # one HDF. Use the union for that formula and report the inflation.
    #
    spec_formula = meta[
        ["spec", "truth_formula"]
    ].drop_duplicates()

    selected = selected.merge(
        spec_formula,
        on="spec",
        how="left",
        validate="many_to_one",
        suffixes=("", "_meta"),
    )

    if "truth_formula_meta" in selected.columns:
        if not (
            selected["truth_formula"].astype(str)
            == selected["truth_formula_meta"].astype(str)
        ).all():
            raise RuntimeError(
                "Formula mismatch while building pruned MIST candidates"
            )
        selected = selected.drop(columns=["truth_formula_meta"])

    union_rows = (
        selected[
            [
                "truth_formula",
                "candidate_inchikey14",
                "candidate_smiles",
            ]
        ]
        .drop_duplicates(
            subset=[
                "truth_formula",
                "candidate_inchikey14",
            ]
        )
        .copy()
    )

    per_spec_counts = (
        selected.groupby("spec")
        .size()
        .rename("selected_per_spec")
        .reset_index()
        .merge(
            spec_formula,
            on="spec",
            how="left",
            validate="one_to_one",
        )
    )

    union_counts = (
        union_rows.groupby("truth_formula")
        .size()
        .rename("formula_union_count")
        .reset_index()
    )

    formula_spec_counts = (
        spec_formula.groupby("truth_formula")["spec"]
        .nunique()
        .rename("spec_count")
        .reset_index()
    )

    formula_summary = (
        formula_spec_counts
        .merge(
            union_counts,
            on="truth_formula",
            how="left",
            validate="one_to_one",
        )
    )

    formula_summary.to_parquet(
        FORMULA_SUMMARY_PATH,
        index=False,
    )

    selected.to_parquet(
        SELECTED_PATH,
        index=False,
    )

    form_to_smi = {}

    for formula, group in union_rows.groupby(
        "truth_formula",
        sort=False,
    ):
        form_to_smi[str(formula)] = [
            {
                "smi": str(row.candidate_smiles),
                "ikey": str(row.candidate_inchikey14),
            }
            for row in group.itertuples(index=False)
        ]

    with FORM_TO_SMI_PATH.open("wb") as handle:
        pickle.dump(form_to_smi, handle)

    duplicate_formula_rows = formula_summary.loc[
        formula_summary["spec_count"] > 1
    ]

    print()
    print("Pruned PubChem/MIST candidate map")
    print("=================================")
    print("Benchmark molecules:", meta["spec"].nunique())
    print("Unique formulas:", spec_formula["truth_formula"].nunique())
    print("Selected spectrum-candidate rows:", f"{len(selected):,}")
    print("Formula-union candidate rows:", f"{len(union_rows):,}")
    print(
        "Formulas shared by multiple benchmark spectra:",
        len(duplicate_formula_rows),
    )

    if not duplicate_formula_rows.empty:
        print()
        print("Shared-formula union details:")
        print(
            duplicate_formula_rows.sort_values(
                ["spec_count", "formula_union_count"],
                ascending=[False, False],
            ).to_string(index=False)
        )

    print()
    print("Per-bucket selected candidates:")
    bucket_counts = (
        selected.groupby("burden_bucket")
        .size()
    )
    print(bucket_counts.to_string())

    print()
    print(f"Wrote {FORM_TO_SMI_PATH}")
    print(f"Wrote {SELECTED_PATH}")
    print(f"Wrote {FORMULA_SUMMARY_PATH}")


if __name__ == "__main__":
    main()
