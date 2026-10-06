from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem


ROOT = Path("data/processed/mist/validation_500")

RANKED_PATH = (
    ROOT / "mist_validation_ranked_candidates.parquet"
)

TRAINING_STRUCTURES_PATH = Path(
    "data/processed/structure_index/training_structures.parquet"
)

COCONUT_PATH = Path(
    "data/external/coconut/coconut_approved.parquet"
)

OUTPUT_FP_PATH = (
    ROOT / "mist_validation_candidate_fingerprints.npz"
)

OUTPUT_META_PATH = (
    ROOT / "mist_validation_candidate_fingerprint_metadata.parquet"
)

RADIUS = 2
N_BITS = 4096


def make_fp(smiles: str):
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None

    fp = AllChem.GetMorganFingerprintAsBitVect(
        mol,
        radius=RADIUS,
        nBits=N_BITS,
    )

    bit_string = fp.ToBitString()

    return np.fromiter(
        (1.0 if c == "1" else 0.0 for c in bit_string),
        dtype=np.float32,
        count=N_BITS,
    )


def build_catalog():
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

    catalog = catalog.drop_duplicates(
        subset=["inchikey14"],
        keep="first",
    )

    return catalog[
        [
            "inchikey14",
            "normalized_smiles",
        ]
    ]


def main():
    ranked = pd.read_parquet(
        RANKED_PATH
    )

    print(
        "Ranked candidate rows:",
        len(ranked),
    )

    catalog = build_catalog()

    ranked = ranked.merge(
        catalog.rename(
            columns={
                "inchikey14":
                    "candidate_inchikey14",
                "normalized_smiles":
                    "candidate_smiles",
            }
        ),
        on="candidate_inchikey14",
        how="left",
        validate="many_to_one",
    )

    if ranked[
        "candidate_smiles"
    ].isna().any():
        raise ValueError(
            "Some ranked candidates are missing SMILES."
        )

    unique = (
        ranked[
            [
                "candidate_inchikey14",
                "candidate_smiles",
            ]
        ]
        .drop_duplicates(
            "candidate_inchikey14"
        )
        .reset_index(drop=True)
    )

    print(
        "Unique structures:",
        len(unique),
    )

    fp_by_ikey = {}

    invalid = []

    for i, row in enumerate(
        unique.itertuples(
            index=False
        ),
        start=1,
    ):
        fp = make_fp(
            row.candidate_smiles
        )

        if fp is None:
            invalid.append(
                row.candidate_inchikey14
            )
        else:
            fp_by_ikey[
                row.candidate_inchikey14
            ] = fp

        if (
            i % 10000 == 0
            or i == len(unique)
        ):
            print(
                f"Fingerprint progress: "
                f"{i}/{len(unique)}"
            )

    print(
        "Invalid structures:",
        len(invalid),
    )

    if invalid:
        raise ValueError(
            f"Invalid structures found: "
            f"{invalid[:10]}"
        )

    fingerprints = np.stack(
        [
            fp_by_ikey[
                ikey
            ]
            for ikey in ranked[
                "candidate_inchikey14"
            ]
        ],
        axis=0,
    ).astype(
        np.float32,
        copy=False,
    )

    if fingerprints.shape != (
        len(ranked),
        N_BITS,
    ):
        raise ValueError(
            "Unexpected fingerprint matrix shape: "
            f"{fingerprints.shape}"
        )

    np.savez_compressed(
        OUTPUT_FP_PATH,
        fingerprints=fingerprints,
    )

    ranked.to_parquet(
        OUTPUT_META_PATH,
        index=False,
    )

    print(
        "\nFingerprint shape:",
        fingerprints.shape,
    )

    print(
        "Fingerprint dtype:",
        fingerprints.dtype,
    )

    print(
        f"\nWrote {OUTPUT_FP_PATH}"
    )

    print(
        f"Wrote {OUTPUT_META_PATH}"
    )


if __name__ == "__main__":
    main()