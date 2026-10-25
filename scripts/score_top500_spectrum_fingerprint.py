from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem
import torch
from torch import nn

from casmi26.spectra.binning import bin_spectrum


ROOT = Path(
    "data/processed/diagnostics/large_validation_audit/"
    "true_formula_neighbor_benchmark"
)

CANDIDATES_PATH = (
    ROOT
    / "top3_neighbor_collapse/top3_neighbor_top500.parquet"
)

SAMPLE_PATH = (
    ROOT
    / "mist_full/sample.parquet"
)

TRAIN_PATH = Path(
    "data/raw/train.parquet"
)

PUBCHEM_ROOT = Path(
    "data/processed/pubchem_formula_index"
)


MAX_MZ = 1500.0
BIN_WIDTH = 1.0
FP_RADIUS = 2
FP_BITS = 2048
EPS = 1e-6

RDLogger.DisableLog("rdApp.*")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-name",
        default="prototype",
        help="Subdirectory under data/processed/spectrum_fingerprint.",
    )
    parser.add_argument(
        "--output-name",
        default=None,
        help="Output enrichment stem. Defaults to the model name.",
    )
    return parser.parse_args()


class SpectrumFingerprintMLP(
    nn.Module
):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
    ) -> None:
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(
                input_dim,
                1024,
            ),
            nn.ReLU(),
            nn.Dropout(
                0.15
            ),
            nn.Linear(
                1024,
                1024,
            ),
            nn.ReLU(),
            nn.Dropout(
                0.15
            ),
            nn.Linear(
                1024,
                output_dim,
            ),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return self.network(
            x
        )


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


def select_query_vectors(
    keys: set[str],
) -> dict[str, np.ndarray]:
    parquet = pq.ParquetFile(
        TRAIN_PATH
    )

    best: dict[
        str,
        tuple[
            int,
            float,
            object,
            object,
        ],
    ] = {}

    scanned = 0
    total_rows = (
        parquet.metadata.num_rows
    )

    for batch in parquet.iter_batches(
        batch_size=100_000,
        columns=[
            "inchikey14",
            "ms2_mzs",
            "ms2_normalized_intensities",
            "num_peaks",
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
                keys
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
            f"query spectra: {len(best):,}/{len(keys):,}",
            end="",
            flush=True,
        )

    print()

    result = {}

    for key, (
        _,
        _,
        mzs,
        intensities,
    ) in best.items():
        vector = bin_spectrum(
            mzs,
            intensities,
            max_mz=MAX_MZ,
            bin_width=BIN_WIDTH,
        )

        if np.any(
            vector
        ):
            result[
                key
            ] = vector

    return result


def load_candidate_smiles(
    candidates: pd.DataFrame,
) -> dict[
    tuple[str, str],
    str,
]:
    formulas = set(
        candidates[
            "candidate_formula"
        ]
        .astype(str)
    )

    keys_by_formula = {
        formula:
            set(
                group[
                    "candidate_inchikey14"
                ]
                .astype(str)
            )
        for formula, group
        in candidates.groupby(
            "candidate_formula",
            sort=False,
        )
    }

    formula_buckets = pd.read_parquet(
        PUBCHEM_ROOT
        / "formula_buckets.parquet"
    )[
        [
            "molecular_formula",
            "bucket",
        ]
    ]

    formula_buckets[
        "molecular_formula"
    ] = (
        formula_buckets[
            "molecular_formula"
        ]
        .astype(str)
    )

    selected = formula_buckets.loc[
        formula_buckets[
            "molecular_formula"
        ]
        .isin(
            formulas
        )
    ]

    result: dict[
        tuple[str, str],
        str,
    ] = {}

    for bucket in sorted(
        selected[
            "bucket"
        ]
        .astype(int)
        .unique()
    ):
        formulas_in_bucket = set(
            selected.loc[
                selected[
                    "bucket"
                ].astype(int)
                == bucket,
                "molecular_formula",
            ]
            .astype(str)
        )

        frame = pd.read_parquet(
            PUBCHEM_ROOT
            / "candidates"
            / f"bucket-{bucket:03d}.parquet",
            columns=[
                "molecular_formula",
                "inchikey14",
                "smiles",
            ],
        )

        frame[
            "molecular_formula"
        ] = (
            frame[
                "molecular_formula"
            ]
            .astype(str)
        )

        frame[
            "inchikey14"
        ] = (
            frame[
                "inchikey14"
            ]
            .astype(str)
        )

        frame = frame.loc[
            frame[
                "molecular_formula"
            ]
            .isin(
                formulas_in_bucket
            )
        ]

        for row in frame.itertuples(
            index=False
        ):
            formula = str(
                row.molecular_formula
            )
            key = str(
                row.inchikey14
            )

            wanted = (
                keys_by_formula.get(
                    formula
                )
            )

            if (
                wanted is not None
                and key in wanted
            ):
                pair = (
                    formula,
                    key,
                )

                if pair not in result:
                    result[
                        pair
                    ] = str(
                        row.smiles
                    )

        print(
            "Loaded PubChem bucket:",
            f"{bucket:03d}",
        )

    return result


def main() -> None:
    args = parse_args()

    model_path = (
        Path("data/processed/spectrum_fingerprint")
        / args.model_name
        / "spectrum_to_morgan2048.pt"
    )

    output_name = (
        args.output_name
        if args.output_name is not None
        else args.model_name
    )

    output_path = (
        ROOT
        / "learned_ranker_top500"
        / f"spectrum_fingerprint_enrichment_{output_name}.parquet"
    )

    for path in [
        CANDIDATES_PATH,
        SAMPLE_PATH,
        TRAIN_PATH,
        PUBCHEM_ROOT
        / "formula_buckets.parquet",
        model_path,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"Missing required artifact: {path}"
            )

    candidates = pd.read_parquet(
        CANDIDATES_PATH
    ).copy()

    for column in [
        "inchikey14",
        "candidate_formula",
        "candidate_inchikey14",
    ]:
        candidates[
            column
        ] = (
            candidates[
                column
            ]
            .astype(str)
        )

    query_keys = set(
        candidates[
            "inchikey14"
        ]
    )

    query_vectors = (
        select_query_vectors(
            query_keys
        )
    )

    if len(
        query_vectors
    ) != len(
        query_keys
    ):
        print(
            "WARNING: usable query vectors:",
            f"{len(query_vectors):,}/"
            f"{len(query_keys):,}",
        )

    checkpoint = torch.load(
        model_path,
        map_location="cpu",
        weights_only=True,
    )

    model = SpectrumFingerprintMLP(
        input_dim=int(
            checkpoint[
                "input_dim"
            ]
        ),
        output_dim=int(
            checkpoint[
                "output_dim"
            ]
        ),
    )

    model.load_state_dict(
        checkpoint[
            "state_dict"
        ]
    )

    model.eval()

    predicted_by_query: dict[
        str,
        np.ndarray,
    ] = {}

    ordered_keys = sorted(
        query_vectors
    )

    batch_size = 256

    with torch.no_grad():
        for start in range(
            0,
            len(
                ordered_keys
            ),
            batch_size,
        ):
            batch_keys = ordered_keys[
                start:
                start
                + batch_size
            ]

            batch = np.asarray(
                [
                    query_vectors[
                        key
                    ]
                    for key in batch_keys
                ],
                dtype=np.float32,
            )

            probabilities = (
                torch.sigmoid(
                    model(
                        torch.from_numpy(
                            batch
                        )
                    )
                )
                .cpu()
                .numpy()
                .astype(
                    np.float32,
                    copy=False,
                )
            )

            for key, vector in zip(
                batch_keys,
                probabilities,
                strict=True,
            ):
                predicted_by_query[
                    key
                ] = vector

    print(
        "Predicted query fingerprints:",
        f"{len(predicted_by_query):,}",
    )

    smiles_by_pair = (
        load_candidate_smiles(
            candidates
        )
    )

    output_frames = []

    query_count = candidates[
        "inchikey14"
    ].nunique()

    for query_number, (
        query_key,
        group,
    ) in enumerate(
        candidates.groupby(
            "inchikey14",
            sort=False,
        ),
        start=1,
    ):
        predicted = (
            predicted_by_query.get(
                str(
                    query_key
                )
            )
        )

        result = group[
            [
                "inchikey14",
                "candidate_formula",
                "candidate_inchikey14",
            ]
        ].copy()

        n = len(
            result
        )

        soft_jaccard = np.full(
            n,
            np.nan,
            dtype=np.float32,
        )
        cosine = np.full(
            n,
            np.nan,
            dtype=np.float32,
        )
        positive_mean = np.full(
            n,
            np.nan,
            dtype=np.float32,
        )
        normalized_log_likelihood = (
            np.full(
                n,
                np.nan,
                dtype=np.float32,
            )
        )

        if predicted is not None:
            candidate_arrays = []
            positions = []

            for position, row in enumerate(
                group.itertuples(
                    index=False
                )
            ):
                pair = (
                    str(
                        row.candidate_formula
                    ),
                    str(
                        row.candidate_inchikey14
                    ),
                )

                smiles = (
                    smiles_by_pair.get(
                        pair
                    )
                )

                if smiles is None:
                    continue

                fp = morgan_array(
                    smiles
                )

                if fp is None:
                    continue

                positions.append(
                    position
                )
                candidate_arrays.append(
                    fp
                )

            if candidate_arrays:
                candidate_matrix = np.asarray(
                    candidate_arrays,
                    dtype=np.float32,
                )

                dot = (
                    candidate_matrix
                    @ predicted
                )

                candidate_sum = (
                    candidate_matrix.sum(
                        axis=1
                    )
                )

                predicted_sum = float(
                    predicted.sum()
                )

                union = (
                    candidate_sum
                    + predicted_sum
                    - dot
                )

                soft = (
                    dot
                    / np.maximum(
                        union,
                        EPS,
                    )
                )

                predicted_norm = float(
                    np.linalg.norm(
                        predicted
                    )
                )

                candidate_norm = (
                    np.linalg.norm(
                        candidate_matrix,
                        axis=1,
                    )
                )

                cos = (
                    dot
                    / np.maximum(
                        candidate_norm
                        * predicted_norm,
                        EPS,
                    )
                )

                pos_mean = (
                    dot
                    / np.maximum(
                        candidate_sum,
                        1.0,
                    )
                )

                clipped = np.clip(
                    predicted,
                    EPS,
                    1.0
                    - EPS,
                )

                log_p = np.log(
                    clipped
                )

                log_not_p = np.log(
                    1.0
                    - clipped
                )

                ll = (
                    candidate_matrix
                    @ log_p
                    + (
                        1.0
                        - candidate_matrix
                    )
                    @ log_not_p
                ) / FP_BITS

                for local, position in enumerate(
                    positions
                ):
                    soft_jaccard[
                        position
                    ] = float(
                        soft[
                            local
                        ]
                    )
                    cosine[
                        position
                    ] = float(
                        cos[
                            local
                        ]
                    )
                    positive_mean[
                        position
                    ] = float(
                        pos_mean[
                            local
                        ]
                    )
                    normalized_log_likelihood[
                        position
                    ] = float(
                        ll[
                            local
                        ]
                    )

        result[
            "predicted_fp_soft_jaccard"
        ] = soft_jaccard
        result[
            "predicted_fp_cosine"
        ] = cosine
        result[
            "predicted_fp_positive_mean"
        ] = positive_mean
        result[
            "predicted_fp_log_likelihood"
        ] = normalized_log_likelihood

        output_frames.append(
            result
        )

        if (
            query_number % 100 == 0
            or query_number
            == query_count
        ):
            print(
                "Scored queries:",
                f"{query_number:,}/"
                f"{query_count:,}",
            )

    result = pd.concat(
        output_frames,
        ignore_index=True,
    )

    if len(
        result
    ) != len(
        candidates
    ):
        raise RuntimeError(
            "Fingerprint enrichment row count mismatch"
        )

    duplicates = int(
        result.duplicated(
            subset=[
                "inchikey14",
                "candidate_formula",
                "candidate_inchikey14",
            ]
        ).sum()
    )

    if duplicates:
        raise RuntimeError(
            f"Fingerprint enrichment contains "
            f"{duplicates:,} duplicate rows"
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_parquet(
        output_path,
        index=False,
        compression="zstd",
    )

    print()
    print(
        "Spectrum-fingerprint candidate enrichment"
    )
    print(
        "=" * 76
    )
    print(
        "Rows:",
        f"{len(result):,}",
    )
    print(
        "Queries:",
        f"{result['inchikey14'].nunique():,}",
    )
    print(
        "Rows with scores:",
        f"{result['predicted_fp_soft_jaccard'].notna().sum():,}",
        f"({result['predicted_fp_soft_jaccard'].notna().mean():.2%})",
    )
    print()
    print(
        result[
            [
                "predicted_fp_soft_jaccard",
                "predicted_fp_cosine",
                "predicted_fp_positive_mean",
                "predicted_fp_log_likelihood",
            ]
        ]
        .describe(
            percentiles=[
                0.5,
                0.9,
                0.95,
                0.99,
            ]
        )
        .to_string()
    )

    truth = result.loc[
        result[
            "candidate_inchikey14"
        ]
        == result[
            "inchikey14"
        ]
    ]

    nontruth = result.loc[
        result[
            "candidate_inchikey14"
        ]
        != result[
            "inchikey14"
        ]
    ]

    print()
    print(
        "Truth vs non-truth predicted-fingerprint scores"
    )
    print(
        "-" * 76
    )

    for column in [
        "predicted_fp_soft_jaccard",
        "predicted_fp_cosine",
        "predicted_fp_positive_mean",
        "predicted_fp_log_likelihood",
    ]:
        truth_values = truth[
            column
        ].dropna()
        false_values = nontruth[
            column
        ].dropna()

        print(
            column
        )
        print(
            "  truth median:",
            (
                f"{truth_values.median():.6f}"
                if not truth_values.empty
                else "n/a"
            ),
        )
        print(
            "  nontruth median:",
            (
                f"{false_values.median():.6f}"
                if not false_values.empty
                else "n/a"
            ),
        )

    print()
    print(
        "Wrote:",
        output_path,
    )


if __name__ == "__main__":
    main()
