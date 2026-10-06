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
    / "data/processed/mist/validation_500"
)

RANKED_PATH = (
    ROOT
    / "mist_validation_ranked_candidates.parquet"
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

LABELS_PATH = (
    ROOT / "mist_multiformula_labels_processable.tsv"
)


def build_structure_catalog():
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
    # Apply the same cross-catalog conflict exclusion used
    # throughout the leakage-safe structure pipeline.
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
        ].isin(conflicts)
    ].copy()

    coconut = coconut[
        ~coconut[
            "inchikey14"
        ].isin(conflicts)
    ].copy()

    combined = pd.concat(
        [
            training,
            coconut,
        ],
        ignore_index=True,
    )

    #
    # Prefer the training representation for overlap.
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
        subset=["inchikey14"],
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
    if LABELS_PATH is None:
        raise ValueError(
            "Set LABELS_PATH to the v1 MIST labels TSV first."
        )

    ranked = pd.read_parquet(
        RANKED_PATH
    )

    print(
        "Ranked candidate rows:",
        len(ranked),
    )

    print(
        "Formula hypotheses:",
        ranked[
            "hypothesis_spec"
        ].nunique(),
    )

    print(
        "Unique candidate structures:",
        ranked[
            "candidate_inchikey14"
        ].nunique(),
    )

    catalog = build_structure_catalog()

    candidate_ids = set(
        ranked[
            "candidate_inchikey14"
        ]
    )

    catalog = catalog[
        catalog[
            "inchikey14"
        ].isin(candidate_ids)
    ].copy()

    found_ids = set(
        catalog[
            "inchikey14"
        ]
    )

    missing = (
        candidate_ids
        - found_ids
    )

    print(
        "Catalog structures recovered:",
        len(found_ids),
    )

    print(
        "Missing structures:",
        len(missing),
    )

    if missing:
        raise ValueError(
            "Could not recover all ranked structures. "
            f"Examples: {sorted(missing)[:10]}"
        )

    #
    # Preserve the exact formula/structure combinations from
    # the existing ranked candidate universe.
    #
    candidates = ranked[
        [
            "candidate_formula",
            "candidate_inchikey14",
        ]
    ].drop_duplicates()

    candidates = candidates.merge(
        catalog[
            [
                "inchikey14",
                "normalized_smiles",
            ]
        ],
        left_on="candidate_inchikey14",
        right_on="inchikey14",
        how="left",
        validate="many_to_one",
    )

    if candidates[
        "normalized_smiles"
    ].isna().any():
        raise ValueError(
            "Some candidates have no SMILES."
        )

    form_to_smi = {}

    for formula, group in candidates.groupby(
        "candidate_formula",
        sort=False,
    ):
        form_to_smi[
            formula
        ] = [
            {
                "smi":
                    row.normalized_smiles,

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

    print(
        "\nFormulas:",
        len(form_to_smi),
    )

    print(
        "Formula/structure entries:",
        sum(
            len(v)
            for v in form_to_smi.values()
        ),
    )

    print(
        f"Wrote {FORM_TO_SMI_PATH}"
    )

    #
    # Import MIST's own HDF builder.
    #
    sys.path.insert(
        0,
        str(MIST_ROOT / "src"),
    )

    from mist.retrieval_lib.make_hdf5 import (
        make_retrieval_hdf5,
    )

    make_retrieval_hdf5(
        labels_file=str(
            LABELS_PATH
        ),
        form_to_smi=form_to_smi,
        output_dir=str(
            OUTPUT_DIR
        ),
        database_name=(
            "casmi_validation_500_morgan4096"
        ),

        fp_names=[
            "morgan4096"
        ],
    )

    print(
        f"\nHDF output directory: {OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()
