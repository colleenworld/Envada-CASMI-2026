from pathlib import Path

import pandas as pd


ROOT = Path("data/processed/mist/validation_500_v2")

HYPOTHESES_PATH = (
    ROOT / "mist_multispectrum_hypotheses.parquet"
)

MANIFEST_PATH = ROOT / "manifest.parquet"

TRAINING_STRUCTURES_PATH = Path(
    "data/processed/structure_index/training_structures.parquet"
)

COCONUT_PATH = Path(
    "data/external/coconut/coconut_approved.parquet"
)

OUTPUT_PATH = (
    ROOT / "mist_structure_candidates.parquet"
)


def main():
    hypotheses = pd.read_parquet(
        HYPOTHESES_PATH
    )

    manifest = pd.read_parquet(
        MANIFEST_PATH
    )

    training = pd.read_parquet(
        TRAINING_STRUCTURES_PATH
    )

    coconut = pd.read_parquet(
        COCONUT_PATH
    )

    print("Hypotheses:", len(hypotheses))
    print("Queries:", len(manifest))

    #
    # Normalize the two structure catalogs into the same schema.
    #
    training = training[
        [
            "inchikey14",
            "normalized_smiles",
            "molecular_formula",
            "monoisotopic_mass",
        ]
    ].copy()

    training["in_training"] = True
    training["in_coconut"] = False

    coconut = coconut[
        [
            "inchikey14",
            "smiles",
            "molecular_formula",
            "mass",
        ]
    ].copy()

    coconut = coconut.rename(
        columns={
            "smiles": "normalized_smiles",
            "mass": "monoisotopic_mass",
        }
    )

    coconut["in_training"] = False
    coconut["in_coconut"] = True

    #
    # Identify structures appearing in both catalogs with
    # conflicting formulas.
    #
    overlap = training[
        [
            "inchikey14",
            "molecular_formula",
        ]
    ].merge(
        coconut[
            [
                "inchikey14",
                "molecular_formula",
            ]
        ],
        on="inchikey14",
        how="inner",
        suffixes=(
            "_training",
            "_coconut",
        ),
    )

    conflicts = overlap[
        overlap["molecular_formula_training"]
        != overlap["molecular_formula_coconut"]
    ]["inchikey14"].unique()

    conflict_set = set(conflicts)

    print(
        "Cross-catalog formula conflicts:",
        len(conflict_set),
    )

    training = training[
        ~training["inchikey14"].isin(
            conflict_set
        )
    ].copy()

    coconut = coconut[
        ~coconut["inchikey14"].isin(
            conflict_set
        )
    ].copy()

    #
    # Combine catalogs.
    #
    # Prefer the training representation when a structure exists
    # in both catalogs.
    #
    catalog = pd.concat(
        [
            training,
            coconut,
        ],
        ignore_index=True,
    )

    catalog = catalog.sort_values(
        by=[
            "inchikey14",
            "in_training",
        ],
        ascending=[
            True,
            False,
        ],
    )

    #
    # Preserve source membership before deduplication.
    #
    membership = (
        catalog.groupby(
            "inchikey14",
            as_index=False,
        )
        .agg(
            in_training=(
                "in_training",
                "max",
            ),
            in_coconut=(
                "in_coconut",
                "max",
            ),
        )
    )

    catalog = (
        catalog.drop_duplicates(
            subset=["inchikey14"],
            keep="first",
        )
        .drop(
            columns=[
                "in_training",
                "in_coconut",
            ]
        )
        .merge(
            membership,
            on="inchikey14",
            how="left",
            validate="one_to_one",
        )
    )

    print(
        "Combined unique structures:",
        len(catalog),
    )

    print(
        "Training structures:",
        int(
            catalog[
                "in_training"
            ].sum()
        ),
    )

    print(
        "COCONUT structures:",
        int(
            catalog[
                "in_coconut"
            ].sum()
        ),
    )

    #
    # Join each formula hypothesis to every structure having
    # that formula.
    #
    candidates = hypotheses[
        [
            "hypothesis_spec",
            "query_inchikey14",
            "candidate_formula",
            "is_truth_formula",
            "selected_mass_error_ppm",
            "mass_error_best_abs_ppm",
            "formula_support_count",
            "formula_support_fraction",
        ]
    ].merge(
        catalog,
        left_on="candidate_formula",
        right_on="molecular_formula",
        how="inner",
        validate="many_to_many",
    )

    candidates = candidates.rename(
        columns={
            "inchikey14":
                "candidate_inchikey14",
            "normalized_smiles":
                "candidate_smiles",
            "monoisotopic_mass":
                "candidate_monoisotopic_mass",
        }
    )

    candidates["is_truth_structure"] = (
        candidates[
            "candidate_inchikey14"
        ]
        == candidates[
            "query_inchikey14"
        ]
    )

    #
    # No duplicate structure within a hypothesis.
    #
    duplicate_mask = candidates.duplicated(
        subset=[
            "hypothesis_spec",
            "candidate_inchikey14",
        ],
        keep=False,
    )

    if duplicate_mask.any():
        example = candidates.loc[
            duplicate_mask,
            [
                "hypothesis_spec",
                "candidate_inchikey14",
            ],
        ].head(20)

        raise ValueError(
            "Duplicate structures within formula "
            f"hypotheses:\n{example}"
        )

    candidates.to_parquet(
        OUTPUT_PATH,
        index=False,
    )

    #
    # Diagnostics.
    #
    query_candidate_counts = (
        candidates.groupby(
            "query_inchikey14"
        )[
            "candidate_inchikey14"
        ]
        .nunique()
    )

    hypothesis_candidate_counts = (
        candidates.groupby(
            "hypothesis_spec"
        )[
            "candidate_inchikey14"
        ]
        .nunique()
    )

    truth_queries = set(
        candidates.loc[
            candidates[
                "is_truth_structure"
            ],
            "query_inchikey14",
        ]
    )

    hypothesis_ids = set(
        hypotheses[
            "hypothesis_spec"
        ]
    )

    candidate_hypothesis_ids = set(
        candidates[
            "hypothesis_spec"
        ]
    )

    print("\nStructure candidates")
    print("--------------------")

    print(
        "Candidate rows:",
        len(candidates),
    )

    print(
        "Hypotheses with >=1 structure:",
        len(
            candidate_hypothesis_ids
        ),
        "/",
        len(hypothesis_ids),
    )

    print(
        "Hypotheses with no structures:",
        len(
            hypothesis_ids
            - candidate_hypothesis_ids
        ),
    )

    print(
        "Queries with >=1 structure:",
        candidates[
            "query_inchikey14"
        ].nunique(),
        "/ 500",
    )

    print(
        "Queries with truth structure:",
        len(truth_queries),
        "/ 500",
    )

    print(
        "\nStructures per hypothesis:"
    )

    print(
        hypothesis_candidate_counts
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
        "\nUnique structures per query:"
    )

    print(
        query_candidate_counts
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
        "\nTruth candidate rows:",
        int(
            candidates[
                "is_truth_structure"
            ].sum()
        ),
    )

    print(
        f"\nWrote {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()