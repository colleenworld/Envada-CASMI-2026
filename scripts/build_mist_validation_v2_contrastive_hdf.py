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

ROOT = (
    PROJECT_ROOT
    / "data/processed/mist/validation_500_v2"
)

CANDIDATES_PATH = (
    ROOT
    / "mist_structure_candidates.parquet"
)

LABELS_PATH = (
    ROOT
    / "mist_multispectrum_labels.tsv"
)

TRAINING_STRUCTURES_PATH = (
    PROJECT_ROOT
    / "data/processed/structure_index/training_structures.parquet"
)

COCONUT_PATH = (
    PROJECT_ROOT
    / "data/external/coconut/coconut_approved.parquet"
)

FORM_TO_SMI_PATH = (
    ROOT
    / "contrastive_form_to_smi.pkl"
)

OUTPUT_DIR = (
    ROOT
    / "contrastive_retrieval"
)


def build_structure_catalog():
    """
    Rebuild the same leakage-safe structure catalog used by the
    v2 candidate generator:

      - training_structures.parquet
      - approved COCONUT
      - exclude cross-catalog formula conflicts
      - prefer training SMILES when a structure exists in both
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

    #
    # Detect the same cross-catalog formula conflicts excluded
    # during v2 structure candidate generation.
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
        ~training[
            "inchikey14"
        ].isin(
            conflicts
        )
    ].copy()

    coconut = coconut[
        ~coconut[
            "inchikey14"
        ].isin(
            conflicts
        )
    ].copy()

    combined = pd.concat(
        [
            training,
            coconut,
        ],
        ignore_index=True,
    )

    #
    # Prefer the training representation on overlap.
    #
    combined = combined.sort_values(
        by=[
            "inchikey14",
            "in_training",
        ],
        ascending=[
            True,
            False,
        ],
    )

    catalog = combined.drop_duplicates(
        subset=[
            "inchikey14"
        ],
        keep="first",
    )

    return catalog[
        [
            "inchikey14",
            "normalized_smiles",
            "molecular_formula",
        ]
    ].copy()


def main():
    candidates = pd.read_parquet(
        CANDIDATES_PATH
    )

    labels = pd.read_csv(
        LABELS_PATH,
        sep="\t",
    ).astype(str)

    print(
        "Candidate rows:",
        len(candidates),
    )

    print(
        "Hypotheses in candidates:",
        candidates[
            "hypothesis_spec"
        ].nunique(),
    )

    print(
        "Hypotheses in labels:",
        labels[
            "spec"
        ].nunique(),
    )

    print(
        "Unique candidate structures:",
        candidates[
            "candidate_inchikey14"
        ].nunique(),
    )

    required_candidate_columns = {
        "hypothesis_spec",
        "candidate_formula",
        "candidate_inchikey14",
    }

    missing = (
        required_candidate_columns
        - set(
            candidates.columns
        )
    )

    if missing:
        raise ValueError(
            "Candidate file missing columns: "
            f"{sorted(missing)}"
        )

    #
    # Rebuild the frozen leakage-safe structure catalog so we can
    # recover SMILES for every candidate structure.
    #
    catalog = build_structure_catalog()

    candidate_ids = set(
        candidates[
            "candidate_inchikey14"
        ]
    )

    catalog = catalog[
        catalog[
            "inchikey14"
        ].isin(
            candidate_ids
        )
    ].copy()

    recovered_ids = set(
        catalog[
            "inchikey14"
        ]
    )

    missing_structures = (
        candidate_ids
        - recovered_ids
    )

    print(
        "Catalog structures recovered:",
        len(
            recovered_ids
        ),
    )

    print(
        "Missing structures:",
        len(
            missing_structures
        ),
    )

    if missing_structures:
        raise ValueError(
            "Could not recover all candidate structures. "
            f"Examples: "
            f"{sorted(missing_structures)[:10]}"
        )

    #
    # Preserve exactly the formula/structure combinations in the
    # already-frozen v2 candidate universe.
    #
    formula_structures = candidates[
        [
            "candidate_formula",
            "candidate_inchikey14",
        ]
    ].drop_duplicates()

    formula_structures = (
        formula_structures.merge(
            catalog[
                [
                    "inchikey14",
                    "normalized_smiles",
                ]
            ],
            left_on=(
                "candidate_inchikey14"
            ),
            right_on="inchikey14",
            how="left",
            validate="many_to_one",
        )
    )

    if formula_structures[
        "normalized_smiles"
    ].isna().any():
        missing_rows = (
            formula_structures[
                formula_structures[
                    "normalized_smiles"
                ].isna()
            ]
        )

        raise ValueError(
            "Some candidate structures have no SMILES. "
            f"Examples:\n"
            f"{missing_rows.head(10)}"
        )

    #
    # Build the dictionary expected by MIST:
    #
    # {
    #   formula: [
    #       {"smi": "...", "ikey": "..."},
    #       ...
    #   ]
    # }
    #
    form_to_smi = {}

    for formula, group in (
        formula_structures.groupby(
            "candidate_formula",
            sort=False,
        )
    ):
        entries = []

        for row in group.itertuples(
            index=False
        ):
            entries.append(
                {
                    "smi":
                        row.normalized_smiles,

                    "ikey":
                        row.candidate_inchikey14,
                }
            )

        form_to_smi[
            formula
        ] = entries

    #
    # Save the mapping for reproducibility/debugging.
    #
    with FORM_TO_SMI_PATH.open(
        "wb"
    ) as f:
        pickle.dump(
            form_to_smi,
            f,
        )

    print()
    print(
        "Contrastive candidate mapping"
    )
    print(
        "-----------------------------"
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
    # Check how many label formulas are represented in the HDF
    # candidate mapping.
    #
    label_formulas = set(
        labels[
            "formula"
        ]
    )

    mapped_formulas = set(
        form_to_smi
    )

    missing_label_formulas = (
        label_formulas
        - mapped_formulas
    )

    print(
        "Label formulas:",
        len(
            label_formulas
        ),
    )

    print(
        "Label formulas with no candidate structures:",
        len(
            missing_label_formulas
        ),
    )

    if missing_label_formulas:
        print(
            "Examples:",
            sorted(
                missing_label_formulas
            )[:20],
        )

    #
    # Import MIST's own retrieval HDF builder.
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

    #
    # IMPORTANT:
    #
    # The contrastive model's actual molecule encoder has input
    # dimension 4096, so this MUST be Morgan-4096.
    #
    make_retrieval_hdf5(
        labels_file=str(
            LABELS_PATH
        ),
        form_to_smi=form_to_smi,
        output_dir=str(
            OUTPUT_DIR
        ),
        database_name=(
            "casmi_validation_500_v2_morgan4096"
        ),
        fp_names=[
            "morgan4096"
        ],
    )

    print()
    print(
        "HDF output directory:",
        OUTPUT_DIR,
    )


if __name__ == "__main__":
    main()