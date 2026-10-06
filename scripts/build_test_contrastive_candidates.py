from pathlib import Path
import pickle
import sys

import pandas as pd


PROJECT_ROOT = Path(
    "/home/colleen/PycharmProjects/Envada-CASMI-2026"
)

MIST_ROOT = Path(
    "/home/colleen/PycharmProjects/MIST"
)

TEST_ROOT = (
    PROJECT_ROOT
    / "data/processed/test"
)

HYPOTHESES_PATH = (
    TEST_ROOT
    / "multispectrum_formula_hypotheses.parquet"
)

LABELS_PATH = (
    TEST_ROOT
    / "mist/test_multiformula_labels.tsv"
)

TRAINING_STRUCTURES_PATH = (
    PROJECT_ROOT
    / "data/processed/structure_index/structures.parquet"
)

COCONUT_PATH = (
    PROJECT_ROOT
    / "data/external/coconut/coconut_approved.parquet"
)

CANDIDATES_PATH = (
    TEST_ROOT
    / "mist_structure_candidates.parquet"
)

FORM_TO_SMI_PATH = (
    TEST_ROOT
    / "contrastive_form_to_smi.pkl"
)

HDF_OUTPUT_DIR = (
    TEST_ROOT
    / "contrastive_retrieval"
)

DATABASE_NAME = (
    "casmi_test_morgan4096"
)


def build_structure_catalog():
    """
    Reconstruct the same structure universe used during validation:

      * leakage-safe training structures
      * approved COCONUT structures
      * exclude cross-catalog formula conflicts
      * prefer training representation where a structure occurs
        in both catalogs
    """

    training = pd.read_parquet(
        TRAINING_STRUCTURES_PATH
    )[
        [
            "inchikey14",
            "normalized_smiles",
            "molecular_formula",
        ]
    ].copy()

    training["in_training"] = True
    training["in_coconut"] = False

    coconut = pd.read_parquet(
        COCONUT_PATH
    )[
        [
            "inchikey14",
            "smiles",
            "molecular_formula",
        ]
    ].copy()

    coconut = coconut.rename(
        columns={
            "smiles": "normalized_smiles",
        }
    )

    coconut["in_training"] = False
    coconut["in_coconut"] = True

    #
    # Identify structures for which training and COCONUT disagree
    # about the molecular formula.
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

    conflicts = set(
        overlap.loc[
            overlap[
                "molecular_formula_training"
            ]
            != overlap[
                "molecular_formula_coconut"
            ],
            "inchikey14",
        ]
    )

    print(
        "Cross-catalog formula conflicts:",
        len(conflicts),
    )

    training = training[
        ~training["inchikey14"].isin(
            conflicts
        )
    ].copy()

    coconut = coconut[
        ~coconut["inchikey14"].isin(
            conflicts
        )
    ].copy()

    #
    # Combine.
    #
    combined = pd.concat(
        [
            training,
            coconut,
        ],
        ignore_index=True,
    )

    #
    # Prefer training representation on overlap.
    #
    combined = combined.sort_values(
        [
            "inchikey14",
            "in_training",
        ],
        ascending=[
            True,
            False,
        ],
    )

    catalog = (
        combined
        .drop_duplicates(
            subset=[
                "inchikey14",
            ],
            keep="first",
        )
        .reset_index(
            drop=True
        )
    )

    return catalog


def main():
    HDF_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    hypotheses = pd.read_parquet(
        HYPOTHESES_PATH
    )

    labels = pd.read_csv(
        LABELS_PATH,
        sep="\t",
    )

    print(
        "Formula hypotheses:",
        len(hypotheses),
    )

    print(
        "Molecules:",
        hypotheses[
            "molecule_id"
        ].nunique(),
    )

    print(
        "Unique formulas:",
        hypotheses[
            "candidate_formula"
        ].nunique(),
    )

    if hypotheses[
        "hypothesis_spec"
    ].duplicated().any():
        raise ValueError(
            "Duplicate hypothesis_spec values."
        )

    #
    # Confirm labels and hypotheses describe exactly the same
    # MIST spectrum/formula hypotheses.
    #
    expected_specs = set(
        hypotheses[
            "hypothesis_spec"
        ].astype(str)
    )

    label_specs = set(
        labels[
            "spec"
        ].astype(str)
    )

    if expected_specs != label_specs:
        raise ValueError(
            "Hypothesis and label spec sets differ. "
            f"Missing from labels: "
            f"{len(expected_specs - label_specs)}, "
            f"unexpected labels: "
            f"{len(label_specs - expected_specs)}"
        )

    #
    # Frozen training + approved COCONUT catalog.
    #
    catalog = build_structure_catalog()

    print(
        "Catalog structures:",
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
    # Generate exact formula-matched structure candidates.
    #
    candidate_catalog = catalog[
        [
            "inchikey14",
            "normalized_smiles",
            "molecular_formula",
            "in_training",
            "in_coconut",
        ]
    ].rename(
        columns={
            "inchikey14":
                "candidate_inchikey14",

            "normalized_smiles":
                "candidate_smiles",

            "molecular_formula":
                "candidate_formula",
        }
    )

    candidates = hypotheses.merge(
        candidate_catalog,
        on="candidate_formula",
        how="left",
        validate="many_to_many",
    )

    #
    # Identify formula hypotheses for which our structure catalog
    # contains no structures.
    #
    missing_structure_mask = (
        candidates[
            "candidate_inchikey14"
        ].isna()
    )

    missing_hypotheses = candidates.loc[
        missing_structure_mask,
        [
            "molecule_id",
            "hypothesis_spec",
            "candidate_formula",
        ],
    ].drop_duplicates()

    print()
    print(
        "Formula hypotheses without structure candidates:",
        len(
            missing_hypotheses
        ),
    )

    if len(
        missing_hypotheses
    ):
        print(
            missing_hypotheses
            .head(20)
            .to_string(
                index=False
            )
        )

    #
    # Those empty rows are not candidate structures.
    #
    candidates = candidates[
        ~missing_structure_mask
    ].copy()

    #
    # There should be no duplicate structure within a hypothesis.
    #
    duplicate_mask = candidates.duplicated(
        subset=[
            "hypothesis_spec",
            "candidate_inchikey14",
        ],
        keep=False,
    )

    if duplicate_mask.any():
        duplicates = candidates.loc[
            duplicate_mask,
            [
                "hypothesis_spec",
                "candidate_inchikey14",
            ],
        ]

        raise ValueError(
            "Duplicate hypothesis/structure candidate pairs. "
            f"Examples:\n{duplicates.head(20)}"
        )

    candidates = (
        candidates
        .sort_values(
            [
                "molecule_id",
                "hypothesis_spec",
                "candidate_inchikey14",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    candidates.to_parquet(
        CANDIDATES_PATH,
        index=False,
    )

    print()
    print(
        "Test structure candidates"
    )
    print(
        "-------------------------"
    )

    print(
        "Candidate rows:",
        len(candidates),
    )

    print(
        "Hypotheses with candidates:",
        candidates[
            "hypothesis_spec"
        ].nunique(),
    )

    print(
        "Molecules with candidates:",
        candidates[
            "molecule_id"
        ].nunique(),
    )

    print(
        "Unique candidate structures:",
        candidates[
            "candidate_inchikey14"
        ].nunique(),
    )

    #
    # Candidate sizes per formula hypothesis.
    #
    per_hypothesis = (
        candidates.groupby(
            "hypothesis_spec"
        )
        .size()
    )

    print()
    print(
        "Structures/hypothesis mean:",
        float(
            per_hypothesis.mean()
        ),
    )

    print(
        "Structures/hypothesis median:",
        float(
            per_hypothesis.median()
        ),
    )

    print(
        "Structures/hypothesis max:",
        int(
            per_hypothesis.max()
        ),
    )

    #
    # Candidate sizes across the union of formula hypotheses for
    # each test molecule.
    #
    per_molecule = (
        candidates[
            [
                "molecule_id",
                "candidate_inchikey14",
            ]
        ]
        .drop_duplicates()
        .groupby(
            "molecule_id"
        )
        .size()
    )

    print()
    print(
        "Unique structures/molecule mean:",
        float(
            per_molecule.mean()
        ),
    )

    print(
        "Unique structures/molecule median:",
        float(
            per_molecule.median()
        ),
    )

    print(
        "Unique structures/molecule max:",
        int(
            per_molecule.max()
        ),
    )

    print()
    print(
        f"Wrote {CANDIDATES_PATH}"
    )

    #
    # Build the exact formula -> structures mapping required by
    # MIST's HDF builder.
    #
    formula_structures = (
        candidates[
            [
                "candidate_formula",
                "candidate_inchikey14",
                "candidate_smiles",
            ]
        ]
        .drop_duplicates(
            subset=[
                "candidate_formula",
                "candidate_inchikey14",
            ]
        )
    )

    form_to_smi = {}

    for formula, group in (
        formula_structures.groupby(
            "candidate_formula",
            sort=False,
        )
    ):
        form_to_smi[
            formula
        ] = [
            {
                "smi":
                    row.candidate_smiles,

                "ikey":
                    row.candidate_inchikey14,
            }
            for row in group.itertuples(
                index=False
            )
        ]

    with FORM_TO_SMI_PATH.open(
        "wb"
    ) as f:
        pickle.dump(
            form_to_smi,
            f,
        )

    print()
    print(
        "Contrastive HDF mapping"
    )
    print(
        "-----------------------"
    )

    print(
        "Formulas:",
        len(
            form_to_smi
        ),
    )

    print(
        "Formula/structure entries:",
        sum(
            len(entries)
            for entries
            in form_to_smi.values()
        ),
    )

    print(
        f"Wrote {FORM_TO_SMI_PATH}"
    )

    #
    # Build Morgan-4096 retrieval database using MIST itself.
    #
    sys.path.insert(
        0,
        str(
            MIST_ROOT
            / "src"
        ),
    )

    from mist.retrieval_lib.make_hdf5 import (
        make_retrieval_hdf5,
    )

    print()
    print(
        "Building Morgan-4096 MIST retrieval HDF..."
    )

    make_retrieval_hdf5(
        labels_file=str(
            LABELS_PATH
        ),
        form_to_smi=form_to_smi,
        output_dir=str(
            HDF_OUTPUT_DIR
        ),
        database_name=DATABASE_NAME,
        fp_names=[
            "morgan4096",
        ],
    )

    print()
    print(
        "Finished."
    )

    print(
        "HDF output directory:",
        HDF_OUTPUT_DIR,
    )


if __name__ == "__main__":
    main()