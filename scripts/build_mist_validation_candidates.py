from pathlib import Path

import pandas as pd


ROOT = Path("data/processed/mist/validation_500")

MAPPING_PATH = ROOT / "mist_multiformula_mapping_processable.parquet"

TRAINING_STRUCTURES_PATH = Path(
    "data/processed/structure_index/training_structures.parquet"
)

COCONUT_PATH = Path(
    "data/external/coconut/coconut_approved.parquet"
)

OUTPUT_PATH = ROOT / "mist_multiformula_candidates.parquet"


def build_structure_catalog() -> pd.DataFrame:
    training = pd.read_parquet(TRAINING_STRUCTURES_PATH)[
        [
            "inchikey14",
            "normalized_smiles",
            "molecular_formula",
            "monoisotopic_mass",
        ]
    ].copy()

    training = training.rename(
        columns={"normalized_smiles": "smiles"}
    )

    training = (
        training
        .drop_duplicates("inchikey14")
        .assign(in_training=True)
    )

    coconut = pd.read_parquet(COCONUT_PATH)[
        [
            "inchikey14",
            "smiles",
            "molecular_formula",
            "mass",
            "np_likeness",
            "approved_collections",
        ]
    ].copy()

    coconut = coconut.rename(
        columns={"mass": "coconut_mass"}
    )

    coconut = coconut.drop_duplicates("inchikey14")

    catalog = training.merge(
        coconut,
        on="inchikey14",
        how="outer",
        suffixes=("_training", "_coconut"),
        indicator=True,
    )

    conflict = (
        catalog["molecular_formula_training"].notna()
        & catalog["molecular_formula_coconut"].notna()
        & (
            catalog["molecular_formula_training"]
            != catalog["molecular_formula_coconut"]
        )
    )

    print("Excluding formula conflicts:", int(conflict.sum()))

    catalog = catalog.loc[~conflict].copy()

    catalog["in_training"] = (
        catalog["_merge"].isin(["left_only", "both"])
    )
    catalog["in_coconut"] = (
        catalog["_merge"].isin(["right_only", "both"])
    )

    catalog["smiles"] = (
        catalog["smiles_training"]
        .combine_first(catalog["smiles_coconut"])
    )

    catalog["molecular_formula"] = (
        catalog["molecular_formula_training"]
        .combine_first(catalog["molecular_formula_coconut"])
    )

    catalog["monoisotopic_mass"] = (
        catalog["monoisotopic_mass"]
        .combine_first(catalog["coconut_mass"])
    )

    return catalog[
        [
            "inchikey14",
            "smiles",
            "molecular_formula",
            "monoisotopic_mass",
            "in_training",
            "in_coconut",
            "np_likeness",
            "approved_collections",
        ]
    ].reset_index(drop=True)


def main():
    mapping = pd.read_parquet(MAPPING_PATH)
    structures = build_structure_catalog()

    candidates = mapping.merge(
        structures,
        left_on="candidate_formula",
        right_on="molecular_formula",
        how="left",
    )

    candidates = candidates.rename(
        columns={"inchikey14": "candidate_inchikey14"}
    )

    candidates = candidates[
        candidates["candidate_inchikey14"].notna()
    ].copy()

    candidates["is_truth"] = (
        candidates["candidate_inchikey14"]
        == candidates["query_inchikey14"]
    )

    candidates = candidates.sort_values(
        [
            "query_inchikey14",
            "hypothesis_spec",
            "candidate_inchikey14",
        ],
        kind="stable",
    ).reset_index(drop=True)

    candidates.to_parquet(
        OUTPUT_PATH,
        index=False,
    )

    per_query = (
        candidates.groupby("query_inchikey14")
        ["candidate_inchikey14"]
        .nunique()
    )

    print("\nCandidate table")
    print("---------------")
    print("Rows:", len(candidates))
    print(
        "Formula hypotheses with >=1 structure:",
        candidates["hypothesis_spec"].nunique(),
        "/",
        mapping["hypothesis_spec"].nunique(),
    )
    print(
        "Queries with >=1 structure candidate:",
        candidates["query_inchikey14"].nunique(),
    )
    print(
        "Queries whose truth structure is present:",
        candidates.loc[
            candidates["is_truth"],
            "query_inchikey14",
        ].nunique(),
        "/ 500",
    )

    print("\nStructures/query:")
    print(
        per_query.describe(
            percentiles=[0.5, 0.75, 0.9, 0.95, 0.99]
        ).to_string()
    )

    print(f"\nWrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()