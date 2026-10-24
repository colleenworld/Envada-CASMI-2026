from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem
from scipy import sparse

from casmi26.spectra.binning import bin_spectrum


TRAIN_PATH = Path(
    "data/raw/train.parquet"
)

TRAIN_STRUCTURES_PATH = Path(
    "data/processed/splits/train_structures.parquet"
)

STRUCTURE_INDEX_PATH = Path(
    "data/processed/structure_index/structures.parquet"
)

OUTPUT_DIR = Path(
    "data/processed/spectrum_fingerprint/prototype"
)

TRAIN_VECTORS_PATH = (
    OUTPUT_DIR
    / "train_vectors.npz"
)

TRAIN_TARGETS_PATH = (
    OUTPUT_DIR
    / "train_targets.npy"
)

TRAIN_METADATA_PATH = (
    OUTPUT_DIR
    / "train_metadata.parquet"
)

DEV_VECTORS_PATH = (
    OUTPUT_DIR
    / "dev_vectors.npz"
)

DEV_TARGETS_PATH = (
    OUTPUT_DIR
    / "dev_targets.npy"
)

DEV_METADATA_PATH = (
    OUTPUT_DIR
    / "dev_metadata.parquet"
)

SEED = 42
TRAIN_STRUCTURES = 50_000
DEV_STRUCTURES = 5_000

MAX_MZ = 1500.0
BIN_WIDTH = 1.0
FP_BITS = 2048
FP_RADIUS = 2

RDLogger.DisableLog("rdApp.*")


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


def main() -> None:
    for path in [
        TRAIN_PATH,
        TRAIN_STRUCTURES_PATH,
        STRUCTURE_INDEX_PATH,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"Missing required artifact: {path}"
            )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    train_split = pd.read_parquet(
        TRAIN_STRUCTURES_PATH,
        columns=[
            "inchikey14",
        ],
    )

    training_keys = (
        train_split[
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

    if len(
        training_keys
    ) < (
        TRAIN_STRUCTURES
        + DEV_STRUCTURES
    ):
        raise RuntimeError(
            "Not enough training structures for prototype split"
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
            :DEV_STRUCTURES
        ]
    )

    fit_keys = set(
        shuffled[
            DEV_STRUCTURES:
            DEV_STRUCTURES
            + TRAIN_STRUCTURES
        ]
    )

    selected_keys = (
        fit_keys
        | dev_keys
    )

    structure_index = pd.read_parquet(
        STRUCTURE_INDEX_PATH,
        columns=[
            "inchikey14",
            "normalized_smiles",
        ],
    )

    structure_index[
        "inchikey14"
    ] = (
        structure_index[
            "inchikey14"
        ]
        .astype(str)
    )

    smiles_by_key = dict(
        zip(
            structure_index[
                "inchikey14"
            ],
            structure_index[
                "normalized_smiles"
            ].astype(str),
            strict=True,
        )
    )

    #
    # Select one deterministic representative spectrum per structure:
    # most peaks, then lowest precursor m/z, then first row encountered.
    #
    best: dict[
        str,
        tuple[
            int,
            float,
            object,
            object,
            str,
            str,
            float,
        ],
    ] = {}

    parquet = pq.ParquetFile(
        TRAIN_PATH
    )

    total_rows = (
        parquet.metadata.num_rows
    )

    scanned = 0

    for batch in parquet.iter_batches(
        batch_size=100_000,
        columns=[
            "inchikey14",
            "ms2_mzs",
            "ms2_normalized_intensities",
            "num_peaks",
            "ionization_mode",
            "adduct",
            "precursor_mz",
        ],
    ):
        frame = batch.to_pandas()

        scanned += len(
            frame
        )

        frame = frame.loc[
            frame[
                "inchikey14"
            ]
            .astype(str)
            .isin(
                selected_keys
            )
        ]

        for row in frame.itertuples(
            index=False
        ):
            key = str(
                row.inchikey14
            )

            try:
                num_peaks = int(
                    row.num_peaks
                )
            except (
                TypeError,
                ValueError,
            ):
                num_peaks = len(
                    row.ms2_mzs
                )

            try:
                precursor_mz = float(
                    row.precursor_mz
                )
            except (
                TypeError,
                ValueError,
            ):
                precursor_mz = np.inf

            if not np.isfinite(
                precursor_mz
            ):
                precursor_mz = np.inf

            candidate = (
                num_peaks,
                precursor_mz,
                row.ms2_mzs,
                row.ms2_normalized_intensities,
                str(
                    row.ionization_mode
                ),
                str(
                    row.adduct
                ),
                precursor_mz,
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

        print(
            f"\rRows scanned: {scanned:,}/{total_rows:,} | "
            f"selected structures: {len(best):,}/"
            f"{len(selected_keys):,}",
            end="",
            flush=True,
        )

    print()

    def build_split(
        keys: set[str],
        *,
        vectors_path: Path,
        targets_path: Path,
        metadata_path: Path,
        label: str,
    ) -> None:
        vectors = []
        targets = []
        rows = []

        missing_spectrum = 0
        missing_smiles = 0
        invalid_smiles = 0

        for key in sorted(
            keys
        ):
            spectrum = best.get(
                key
            )

            if spectrum is None:
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

            (
                num_peaks,
                _,
                mzs,
                intensities,
                ionization_mode,
                adduct,
                precursor_mz,
            ) = spectrum

            vector = bin_spectrum(
                mzs,
                intensities,
                max_mz=MAX_MZ,
                bin_width=BIN_WIDTH,
            )

            if not np.any(
                vector
            ):
                continue

            vectors.append(
                vector
            )
            targets.append(
                fp
            )
            rows.append(
                {
                    "inchikey14":
                        key,
                    "ionization_mode":
                        ionization_mode,
                    "adduct":
                        adduct,
                    "precursor_mz":
                        precursor_mz,
                    "num_peaks":
                        num_peaks,
                }
            )

        if not rows:
            raise RuntimeError(
                f"No usable rows for {label}"
            )

        matrix = sparse.csr_matrix(
            np.asarray(
                vectors,
                dtype=np.float32,
            )
        )

        target_array = np.asarray(
            targets,
            dtype=np.uint8,
        )

        sparse.save_npz(
            vectors_path,
            matrix,
            compressed=True,
        )

        np.save(
            targets_path,
            target_array,
            allow_pickle=False,
        )

        pd.DataFrame(
            rows
        ).to_parquet(
            metadata_path,
            index=False,
            compression="zstd",
        )

        print()
        print(
            f"{label} prototype"
        )
        print(
            "-" * 72
        )
        print(
            "Requested structures:",
            f"{len(keys):,}",
        )
        print(
            "Usable structures:",
            f"{len(rows):,}",
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
            matrix.shape,
        )
        print(
            "Target shape:",
            target_array.shape,
        )
        print(
            "Mean fingerprint density:",
            f"{target_array.mean():.4%}",
        )

    build_split(
        fit_keys,
        vectors_path=TRAIN_VECTORS_PATH,
        targets_path=TRAIN_TARGETS_PATH,
        metadata_path=TRAIN_METADATA_PATH,
        label="Training",
    )

    build_split(
        dev_keys,
        vectors_path=DEV_VECTORS_PATH,
        targets_path=DEV_TARGETS_PATH,
        metadata_path=DEV_METADATA_PATH,
        label="Development",
    )

    print()
    print(
        "Spectrum-to-fingerprint prototype dataset complete"
    )
    print(
        "=" * 72
    )
    print(
        "Training structures are drawn only from the existing "
        "training split; validation structures are excluded."
    )
    print(
        "Output:",
        OUTPUT_DIR,
    )


if __name__ == "__main__":
    main()
