from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from casmi26.chemistry.formula_enumeration import (
    build_signature_rules,
    enumerate_for_mass,
    neutral_mass_from_adduct,
    select_query_spectra,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ROOT = PROJECT_ROOT / "data/processed/blind_pseudotest"

TEST_PATH = ROOT / "test.parquet"
TRAINING_STRUCTURES_PATH = (
    PROJECT_ROOT
    / "data/processed/structure_index/training_structures.parquet"
)
PUBCHEM_FORMULAS_PATH = (
    PROJECT_ROOT
    / "data/processed/pubchem_formula_index/formula_buckets.parquet"
)

OUTPUT_PATH = ROOT / "generated_formula_universe.parquet"
SUMMARY_PATH = ROOT / "generated_formula_universe_summary.csv"


def main() -> None:
    for path in [
        TEST_PATH,
        TRAINING_STRUCTURES_PATH,
        PUBCHEM_FORMULAS_PATH,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"Missing required artifact: {path}"
            )

    test = pd.read_parquet(TEST_PATH).copy()
    test["molecule_id"] = test["molecule_id"].astype(str)

    rules = build_signature_rules(
        TRAINING_STRUCTURES_PATH
    )

    current_formulas = set(
        pd.read_parquet(
            PUBCHEM_FORMULAS_PATH,
            columns=["molecular_formula"],
        )["molecular_formula"]
        .dropna()
        .astype(str)
    )

    per_query_rows = []
    union_formulas: set[str] = set()

    for number, (molecule_id, group) in enumerate(
        test.groupby(
            "molecule_id",
            sort=False,
        ),
        start=1,
    ):
        selected = select_query_spectra(group)

        formulas: set[str] = set()

        for spectrum in selected.itertuples(
            index=False
        ):
            neutral_mass = neutral_mass_from_adduct(
                spectrum.precursor_mz,
                spectrum.adduct,
            )

            if not np.isfinite(neutral_mass):
                continue

            for (
                formula,
                _,
                _,
                _,
            ) in enumerate_for_mass(
                neutral_mass,
                rules,
            ):
                formulas.add(formula)

        existing = formulas & current_formulas
        missing = formulas - current_formulas

        union_formulas.update(formulas)

        per_query_rows.append(
            {
                "molecule_id": str(molecule_id),
                "generated_formula_count": len(formulas),
                "already_in_pubchem_asset": len(existing),
                "missing_from_pubchem_asset": len(missing),
                "asset_coverage_fraction": (
                    len(existing) / len(formulas)
                    if formulas
                    else np.nan
                ),
            }
        )

        if number % 50 == 0:
            print(
                f"  processed {number:,} molecules | "
                f"union formulas {len(union_formulas):,}"
            )

    per_query = pd.DataFrame(per_query_rows)

    universe = pd.DataFrame(
        {
            "molecular_formula":
                sorted(union_formulas),
        }
    )

    universe[
        "already_in_pubchem_asset"
    ] = universe[
        "molecular_formula"
    ].isin(
        current_formulas
    )

    universe.to_parquet(
        OUTPUT_PATH,
        index=False,
        compression="zstd",
    )

    summary = pd.DataFrame(
        [
            {
                "queries":
                    len(per_query),
                "unique_generated_formulas":
                    len(union_formulas),
                "already_in_pubchem_asset":
                    int(
                        universe[
                            "already_in_pubchem_asset"
                        ].sum()
                    ),
                "missing_from_pubchem_asset":
                    int(
                        (
                            ~universe[
                                "already_in_pubchem_asset"
                            ]
                        ).sum()
                    ),
                "formula_asset_coverage_fraction":
                    universe[
                        "already_in_pubchem_asset"
                    ].mean(),
                "median_generated_per_query":
                    per_query[
                        "generated_formula_count"
                    ].median(),
                "p90_generated_per_query":
                    per_query[
                        "generated_formula_count"
                    ].quantile(
                        0.90
                    ),
                "median_missing_per_query":
                    per_query[
                        "missing_from_pubchem_asset"
                    ].median(),
                "p90_missing_per_query":
                    per_query[
                        "missing_from_pubchem_asset"
                    ].quantile(
                        0.90
                    ),
                "median_query_asset_coverage":
                    per_query[
                        "asset_coverage_fraction"
                    ].median(),
            }
        ]
    )

    summary.to_csv(
        SUMMARY_PATH,
        index=False,
    )

    row = summary.iloc[0]

    print()
    print(
        "Generated formula universe audit"
    )
    print("=" * 92)
    print(
        "Queries:",
        int(row["queries"]),
    )
    print(
        "Unique generated formulas:",
        f"{int(row['unique_generated_formulas']):,}",
    )
    print(
        "Already in PubChem asset:",
        f"{int(row['already_in_pubchem_asset']):,}",
    )
    print(
        "Missing from PubChem asset:",
        f"{int(row['missing_from_pubchem_asset']):,}",
    )
    print(
        "Formula asset coverage:",
        f"{row['formula_asset_coverage_fraction']:.2%}",
    )
    print(
        "Generated formulas/query median:",
        f"{row['median_generated_per_query']:.1f}",
    )
    print(
        "Generated formulas/query p90:",
        f"{row['p90_generated_per_query']:.1f}",
    )
    print(
        "Missing formulas/query median:",
        f"{row['median_missing_per_query']:.1f}",
    )
    print(
        "Missing formulas/query p90:",
        f"{row['p90_missing_per_query']:.1f}",
    )
    print(
        "Median per-query asset coverage:",
        f"{row['median_query_asset_coverage']:.2%}",
    )
    print()
    print("Wrote:", OUTPUT_PATH)
    print("Wrote:", SUMMARY_PATH)


if __name__ == "__main__":
    main()
