from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem
from scipy import sparse

from casmi26.retrieval.index import load_retrieval_index


STRICT_INDEX_DIR = Path(
    "data/processed/retrieval_index_strict_validation"
)

TRAIN_STRUCTURES_PATH = Path(
    "data/processed/splits/train_structures.parquet"
)

STRUCTURE_INDEX_PATH = Path(
    "data/processed/structure_index/training_structures.parquet"
)

OUTPUT_ROOT = Path(
    "data/processed/spectrum_fingerprint"
)

SEED = 42
FP_BITS = 2048
FP_RADIUS = 2

RDLogger.DisableLog("rdApp.*")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train-structures",
        type=int,
        default=150_000,
    )

    parser.add_argument(
        "--dev-structures",
        type=int,
        default=10_000,
    )

    parser.add_argument(
        "--name",
        default="scaled150k",
    )

    args = parser.parse_args()

    if args.train_structures <= 0:
        parser.error(
            "--train-structures must be positive"
        )

    if args.dev_structures <= 0:
        parser.error(
            "--dev-structures must be positive"
        )

    return args


def morgan_array(
    smiles: str,
) -> np.ndarray | None:
    mol = Chem.MolFromSmiles(
        str(smiles)
    )

    if mol is None:
        return None

    fp = AllChem.GetMorganFingerprintAsBitVect(
        mol,
        radius=FP_RADIUS,
        nBits=FP_BITS,
    )

    result = np.zeros(
        FP_BITS,
        dtype=np.uint8,
    )

    DataStructs.ConvertToNumpyArray(
        fp,
        result,
    )

    return result


def representative_rows(
    metadata: pd.DataFrame,
    vectors: sparse.csr_matrix,
    selected_keys: set[str],
) -> dict[str, int]:
    keys = (
        metadata[
            "inchikey14"
        ]
        .astype(str)
        .to_numpy()
    )

    precursor = pd.to_numeric(
        metadata[
            "precursor_mz"
        ],
        errors="coerce",
    ).to_numpy(
        dtype=np.float64
    )

    row_nnz = np.diff(
        vectors.indptr
    )

    best: dict[
        str,
        tuple[
            int,
            float,
            int,
        ],
    ] = {}

    for row_index, key in enumerate(
        keys
    ):
        if key not in selected_keys:
            continue

        mz = float(
            precursor[
                row_index
            ]
        )

        if not np.isfinite(
            mz
        ):
            mz = np.inf

        candidate = (
            int(
                row_nnz[
                    row_index
                ]
            ),
            mz,
            row_index,
        )

        existing = best.get(
            key
        )

        if (
            existing is None
            or candidate[
                0
            ]
            > existing[
                0
            ]
            or (
                candidate[
                    0
                ]
                == existing[
                    0
                ]
                and candidate[
                    1
                ]
                < existing[
                    1
                ]
            )
        ):
            best[
                key
            ] = candidate

    return {
        key:
            row[
                2
            ]
        for key, row
        in best.items()
    }


def write_split(
    *,
    label: str,
    requested_keys: set[str],
    row_by_key: dict[str, int],
    vectors: sparse.csr_matrix,
    metadata: pd.DataFrame,
    smiles_by_key: dict[str, str],
    output_dir: Path,
) -> None:
    usable_keys = []
    row_indices = []
    targets = []

    missing_spectrum = 0
    missing_smiles = 0
    invalid_smiles = 0

    for key in sorted(
        requested_keys
    ):
        row_index = row_by_key.get(
            key
        )

        if row_index is None:
            missing_spectrum += 1
            continue

        smiles = smiles_by_key.get(
            key
        )

        if smiles is None:
            missing_smiles += 1
            continue

        fp = morgan_array(
            smiles
        )

        if fp is None:
            invalid_smiles += 1
            continue

        usable_keys.append(
            key
        )
        row_indices.append(
            row_index
        )
        targets.append(
            fp
        )

    if not usable_keys:
        raise RuntimeError(
            f"No usable {label} structures"
        )

    split_vectors = vectors[
        np.asarray(
            row_indices,
            dtype=np.int64,
        )
    ].tocsr()

    target_array = np.asarray(
        targets,
        dtype=np.uint8,
    )

    split_metadata = (
        metadata.iloc[
            row_indices
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    split_metadata[
        "inchikey14"
    ] = usable_keys

    sparse.save_npz(
        output_dir
        / f"{label}_vectors.npz",
        split_vectors,
        compressed=True,
    )

    np.save(
        output_dir
        / f"{label}_targets.npy",
        target_array,
        allow_pickle=False,
    )

    split_metadata.to_parquet(
        output_dir
        / f"{label}_metadata.parquet",
        index=False,
        compression="zstd",
    )

    print()
    print(
        f"{label.title()} split"
    )
    print(
        "-" * 72
    )
    print(
        "Requested structures:",
        f"{len(requested_keys):,}",
    )
    print(
        "Usable structures:",
        f"{len(usable_keys):,}",
    )
    print(
        "Missing spectrum:",
        f"{missing_spectrum:,}",
    )
    print(
        "Missing SMILES:",
        f"{missing_smiles:,}",
    )
    print(
        "Invalid SMILES:",
        f"{invalid_smiles:,}",
    )
    print(
        "Vector shape:",
        split_vectors.shape,
    )
    print(
        "Target shape:",
        target_array.shape,
    )
    print(
        "Mean fingerprint density:",
        f"{target_array.mean():.4%}",
    )


def main() -> None:
    args = parse_args()

    output_dir = (
        OUTPUT_ROOT
        / args.name
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    for path in [
        STRICT_INDEX_DIR
        / "metadata.parquet",
        STRICT_INDEX_DIR
        / "vectors.npz",
        TRAIN_STRUCTURES_PATH,
        STRUCTURE_INDEX_PATH,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"Missing required artifact: {path}"
            )

    training_keys = (
        pd.read_parquet(
            TRAIN_STRUCTURES_PATH,
            columns=[
                "inchikey14",
            ],
        )[
            "inchikey14"
        ]
        .dropna()
        .astype(str)
        .drop_duplicates()
        .sort_values()
        .to_numpy(
            dtype=object
        )
    )

    required_count = (
        args.train_structures
        + args.dev_structures
    )

    if len(
        training_keys
    ) < required_count:
        raise RuntimeError(
            "Requested more structures than the "
            "training split contains: "
            f"{required_count:,} requested, "
            f"{len(training_keys):,} available"
        )

    rng = np.random.default_rng(
        SEED
    )

    shuffled = training_keys.copy()

    rng.shuffle(
        shuffled
    )

    dev_keys = set(
        shuffled[
            :args.dev_structures
        ]
    )

    fit_keys = set(
        shuffled[
            args.dev_structures:
            args.dev_structures
            + args.train_structures
        ]
    )

    if fit_keys & dev_keys:
        raise RuntimeError(
            "Train/dev structure overlap"
        )

    print(
        "Loading strict training retrieval index..."
    )

    metadata, vectors = (
        load_retrieval_index(
            STRICT_INDEX_DIR
        )
    )

    vectors = vectors.tocsr()

    selected_keys = (
        fit_keys
        | dev_keys
    )

    print(
        "Selecting representative spectra from "
        "the strict training index..."
    )

    row_by_key = representative_rows(
        metadata,
        vectors,
        selected_keys,
    )

    structures = pd.read_parquet(
        STRUCTURE_INDEX_PATH,
        columns=[
            "inchikey14",
            "normalized_smiles",
        ],
    )

    structures[
        "inchikey14"
    ] = (
        structures[
            "inchikey14"
        ]
        .astype(str)
    )

    smiles_by_key = dict(
        zip(
            structures[
                "inchikey14"
            ],
            structures[
                "normalized_smiles"
            ].astype(str),
            strict=True,
        )
    )

    write_split(
        label="train",
        requested_keys=fit_keys,
        row_by_key=row_by_key,
        vectors=vectors,
        metadata=metadata,
        smiles_by_key=smiles_by_key,
        output_dir=output_dir,
    )

    write_split(
        label="dev",
        requested_keys=dev_keys,
        row_by_key=row_by_key,
        vectors=vectors,
        metadata=metadata,
        smiles_by_key=smiles_by_key,
        output_dir=output_dir,
    )

    config = pd.DataFrame(
        [
            {
                "name":
                    args.name,
                "seed":
                    SEED,
                "requested_train_structures":
                    args.train_structures,
                "requested_dev_structures":
                    args.dev_structures,
                "fingerprint_radius":
                    FP_RADIUS,
                "fingerprint_bits":
                    FP_BITS,
            }
        ]
    )

    config.to_json(
        output_dir
        / "config.json",
        orient="records",
        indent=2,
    )

    print()
    print(
        "Scaled spectrum-fingerprint dataset complete"
    )
    print(
        "=" * 72
    )
    print(
        "Output:",
        output_dir,
    )


if __name__ == "__main__":
    main()
