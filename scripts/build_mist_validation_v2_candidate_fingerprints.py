from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem


ROOT = Path(
    "/home/colleen/PycharmProjects/Envada-CASMI-2026/"
    "data/processed/mist/validation_500_v2"
)

CANDIDATES_PATH = (
    ROOT / "mist_structure_candidates.parquet"
)

OUTPUT_PATH = (
    ROOT / "mist_structure_candidate_fingerprints.npz"
)

METADATA_PATH = (
    ROOT / "mist_structure_candidate_fingerprint_metadata.parquet"
)

RADIUS = 2
N_BITS = 4096


def fingerprint_smiles(smiles: str):
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None

    fp = AllChem.GetMorganFingerprintAsBitVect(
        mol,
        radius=RADIUS,
        nBits=N_BITS,
    )

    arr = np.zeros(
        (N_BITS,),
        dtype=np.float32,
    )

    # RDKit ExplicitBitVect supports direct conversion through
    # ToBitString without requiring DataStructs.ConvertToNumpyArray.
    bit_string = fp.ToBitString()

    arr[:] = np.fromiter(
        (1.0 if c == "1" else 0.0 for c in bit_string),
        dtype=np.float32,
        count=N_BITS,
    )

    return arr


def main():
    candidates = pd.read_parquet(
        CANDIDATES_PATH
    )

    print(
        "Candidate rows:",
        len(candidates),
    )

    required = {
        "hypothesis_spec",
        "query_inchikey14",
        "candidate_inchikey14",
        "candidate_smiles",
        "candidate_formula",
        "is_truth_structure",
    }

    missing = (
        required
        - set(candidates.columns)
    )

    if missing:
        raise ValueError(
            "Missing required columns: "
            f"{sorted(missing)}"
        )

    #
    # Fingerprint unique structures only once.
    #
    unique_structures = (
        candidates[
            [
                "candidate_inchikey14",
                "candidate_smiles",
            ]
        ]
        .drop_duplicates(
            subset=[
                "candidate_inchikey14"
            ]
        )
        .reset_index(drop=True)
    )

    print(
        "Unique structures to fingerprint:",
        len(unique_structures),
    )

    fingerprint_by_structure = {}

    invalid = []

    for i, row in enumerate(
        unique_structures.itertuples(
            index=False
        ),
        start=1,
    ):
        fp = fingerprint_smiles(
            row.candidate_smiles
        )

        if fp is None:
            invalid.append(
                (
                    row.candidate_inchikey14,
                    row.candidate_smiles,
                )
            )
            continue

        fingerprint_by_structure[
            row.candidate_inchikey14
        ] = fp

        if (
            i % 10_000 == 0
            or i == len(unique_structures)
        ):
            print(
                f"Fingerprint progress: "
                f"{i}/{len(unique_structures)}"
            )

    print(
        "Valid unique fingerprints:",
        len(fingerprint_by_structure),
    )

    print(
        "Invalid unique structures:",
        len(invalid),
    )

    if invalid:
        print(
            "\nExample invalid structures:"
        )

        for item in invalid[:20]:
            print(item)

    #
    # Drop only rows whose structure could not be fingerprinted.
    #
    valid_mask = candidates[
        "candidate_inchikey14"
    ].isin(
        fingerprint_by_structure
    )

    removed = candidates[
        ~valid_mask
    ].copy()

    candidates = candidates[
        valid_mask
    ].copy().reset_index(
        drop=True
    )

    #
    # Build fingerprint matrix in exactly the same row order
    # as the metadata table.
    #
    fingerprints = np.stack(
        [
            fingerprint_by_structure[
                inchikey
            ]
            for inchikey in candidates[
                "candidate_inchikey14"
            ]
        ],
        axis=0,
    ).astype(
        np.float32,
        copy=False,
    )

    if fingerprints.shape != (
        len(candidates),
        N_BITS,
    ):
        raise ValueError(
            "Unexpected fingerprint matrix shape: "
            f"{fingerprints.shape}"
        )

    #
    # Save matrix and metadata separately.
    #
    np.savez_compressed(
        OUTPUT_PATH,
        fingerprints=fingerprints,
    )

    candidates.to_parquet(
        METADATA_PATH,
        index=False,
    )

    #
    # Diagnostics.
    #
    truth_before = (
        pd.read_parquet(
            CANDIDATES_PATH
        )
        .loc[
            lambda x: x[
                "is_truth_structure"
            ],
            "query_inchikey14",
        ]
        .nunique()
    )

    truth_after = (
        candidates.loc[
            candidates[
                "is_truth_structure"
            ],
            "query_inchikey14",
        ]
        .nunique()
    )

    print(
        "\nCandidate fingerprint matrix"
    )
    print(
        "----------------------------"
    )

    print(
        "Rows before fingerprinting:",
        len(
            pd.read_parquet(
                CANDIDATES_PATH
            )
        ),
    )

    print(
        "Rows after fingerprinting:",
        len(candidates),
    )

    print(
        "Fingerprint shape:",
        fingerprints.shape,
    )

    print(
        "Fingerprint dtype:",
        fingerprints.dtype,
    )

    print(
        "Truth-containing queries before:",
        truth_before,
        "/ 500",
    )

    print(
        "Truth-containing queries after:",
        truth_after,
        "/ 500",
    )

    print(
        "Removed candidate rows:",
        len(removed),
    )

    print(
        f"\nWrote {OUTPUT_PATH}"
    )

    print(
        f"Wrote {METADATA_PATH}"
    )


if __name__ == "__main__":
    main()