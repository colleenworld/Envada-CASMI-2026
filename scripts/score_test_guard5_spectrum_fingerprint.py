from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem
import torch
from torch import nn

from casmi26.spectra.binning import bin_spectrum


PROJECT_ROOT = Path(
    os.environ.get(
        "CASMI26_PROJECT_ROOT",
        Path(__file__).resolve().parent.parent,
    )
)

TEST_PATH = PROJECT_ROOT / "data/raw/test.parquet"
TEST_ROOT = PROJECT_ROOT / "data/processed/test"
CANDIDATES_PATH = TEST_ROOT / "guard5_neighbor_top500.parquet"
MODEL_PATH = (
    PROJECT_ROOT
    / "data/processed/spectrum_fingerprint/prototype/spectrum_to_morgan2048.pt"
)
OUTPUT_PATH = TEST_ROOT / "guard5_spectrum_fingerprint.parquet"

MAX_MZ = 1500.0
BIN_WIDTH = 1.0
FP_RADIUS = 2
FP_BITS = 2048
EPS = 1e-6

RDLogger.DisableLog("rdApp.*")


class SpectrumFingerprintMLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(1024, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def morgan_array(smiles: str) -> np.ndarray | None:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None

    fp = AllChem.GetMorganFingerprintAsBitVect(
        mol,
        radius=FP_RADIUS,
        nBits=FP_BITS,
    )

    result = np.zeros(FP_BITS, dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(fp, result)
    return result


def main() -> None:
    for path in [TEST_PATH, CANDIDATES_PATH, MODEL_PATH]:
        if not path.exists():
            raise FileNotFoundError(f"Missing required artifact: {path}")

    test = pd.read_parquet(TEST_PATH).reset_index(drop=True)
    candidates = pd.read_parquet(CANDIDATES_PATH).copy()

    required = {
        "hypothesis_spec",
        "source_test_row_id",
        "candidate_formula",
        "candidate_inchikey14",
        "candidate_smiles",
    }
    missing = required - set(candidates.columns)
    if missing:
        raise ValueError(
            "Guard-5 candidate artifact missing columns: "
            f"{sorted(missing)}"
        )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=True,
    )

    model = SpectrumFingerprintMLP(
        input_dim=int(checkpoint["input_dim"]),
        output_dim=int(checkpoint["output_dim"]),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    source_ids = sorted(
        candidates["source_test_row_id"]
        .astype(int)
        .unique()
    )

    predicted_by_source: dict[int, np.ndarray] = {}

    vectors = []
    vector_ids = []

    for source_id in source_ids:
        row = test.iloc[source_id]
        vector = bin_spectrum(
            row.ms2_mzs,
            row.ms2_normalized_intensities,
            max_mz=MAX_MZ,
            bin_width=BIN_WIDTH,
        )
        if np.any(vector):
            vectors.append(vector)
            vector_ids.append(source_id)

    batch_size = 256
    with torch.no_grad():
        for start in range(0, len(vectors), batch_size):
            batch = np.asarray(
                vectors[start:start + batch_size],
                dtype=np.float32,
            )
            probabilities = (
                torch.sigmoid(model(torch.from_numpy(batch)))
                .cpu()
                .numpy()
                .astype(np.float32, copy=False)
            )
            for source_id, probability in zip(
                vector_ids[start:start + batch_size],
                probabilities,
                strict=True,
            ):
                predicted_by_source[source_id] = probability

    fp_cache: dict[str, np.ndarray | None] = {}
    output_frames = []

    groups = candidates.groupby("hypothesis_spec", sort=False)
    group_count = candidates["hypothesis_spec"].nunique()

    for number, (spec, group) in enumerate(groups, start=1):
        source_ids_for_group = (
            group["source_test_row_id"]
            .astype(int)
            .unique()
        )
        if len(source_ids_for_group) != 1:
            raise RuntimeError(
                f"{spec} maps to multiple source_test_row_id values"
            )

        source_id = int(source_ids_for_group[0])
        predicted = predicted_by_source.get(source_id)

        result = group[
            [
                "hypothesis_spec",
                "candidate_formula",
                "candidate_inchikey14",
            ]
        ].copy()

        n = len(group)
        cosine = np.full(n, np.nan, dtype=np.float32)
        positive_mean = np.full(n, np.nan, dtype=np.float32)
        log_likelihood = np.full(n, np.nan, dtype=np.float32)

        if predicted is not None:
            positions = []
            candidate_arrays = []

            for position, row in enumerate(group.itertuples(index=False)):
                smiles = str(row.candidate_smiles)

                if smiles not in fp_cache:
                    fp_cache[smiles] = morgan_array(smiles)

                fp = fp_cache[smiles]
                if fp is None:
                    continue

                positions.append(position)
                candidate_arrays.append(fp)

            if candidate_arrays:
                candidate_matrix = np.asarray(
                    candidate_arrays,
                    dtype=np.float32,
                )

                dot = candidate_matrix @ predicted
                candidate_sum = candidate_matrix.sum(axis=1)

                predicted_norm = float(np.linalg.norm(predicted))
                candidate_norm = np.linalg.norm(
                    candidate_matrix,
                    axis=1,
                )

                cos = dot / np.maximum(
                    candidate_norm * predicted_norm,
                    EPS,
                )

                pos_mean = dot / np.maximum(candidate_sum, 1.0)

                clipped = np.clip(
                    predicted,
                    EPS,
                    1.0 - EPS,
                )
                log_p = np.log(clipped)
                log_not_p = np.log(1.0 - clipped)

                ll = (
                    candidate_matrix @ log_p
                    + (1.0 - candidate_matrix) @ log_not_p
                ) / FP_BITS

                for local, position in enumerate(positions):
                    cosine[position] = float(cos[local])
                    positive_mean[position] = float(pos_mean[local])
                    log_likelihood[position] = float(ll[local])

        result["predicted_fp_cosine"] = cosine
        result["predicted_fp_positive_mean"] = positive_mean
        result["predicted_fp_log_likelihood"] = log_likelihood

        result = result.sort_values(
            [
                "predicted_fp_log_likelihood",
                "predicted_fp_cosine",
                "candidate_inchikey14",
            ],
            ascending=[False, False, True],
            na_position="last",
            kind="stable",
        ).reset_index(drop=True)

        result["predicted_fp_rank"] = (
            np.arange(len(result), dtype=np.int64) + 1
        )

        output_frames.append(result)

        if number % 100 == 0 or number == group_count:
            print(
                "Fingerprint-scored hypotheses:",
                f"{number:,}/{group_count:,}",
            )

    output = pd.concat(output_frames, ignore_index=True)

    if len(output) != len(candidates):
        raise RuntimeError(
            "Fingerprint score row count does not match guard-5 candidates"
        )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(
        OUTPUT_PATH,
        index=False,
        compression="zstd",
    )

    print()
    print("Guard-5 spectrum fingerprint scoring")
    print("=" * 72)
    print("Candidates:", f"{len(output):,}")
    print(
        "Rows with fingerprint scores:",
        f"{output['predicted_fp_log_likelihood'].notna().sum():,}",
        f"({output['predicted_fp_log_likelihood'].notna().mean():.2%})",
    )
    print("Wrote:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
