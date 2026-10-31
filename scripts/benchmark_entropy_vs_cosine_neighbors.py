from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from ms_entropy import FlashEntropySearch
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem
from scipy import sparse

from casmi26.chemistry.formula_enumeration import (
    select_query_spectra,
)
from casmi26.spectra.binning import (
    bin_spectrum,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent

TRAIN_PATH = PROJECT_ROOT / "data/raw/train.parquet"

TRAIN_STRUCTURES_PATH = (
    PROJECT_ROOT
    / "data/processed/splits/train_structures.parquet"
)

BLIND_TEST_PATH = (
    PROJECT_ROOT
    / "data/processed/blind_pseudotest/test.parquet"
)

BLIND_TRUTH_PATH = (
    PROJECT_ROOT
    / "data/processed/blind_pseudotest/truth.parquet"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data/processed/blind_pseudotest/"
      "entropy_vs_cosine_neighbors.parquet"
)

SUMMARY_PATH = (
    PROJECT_ROOT
    / "data/processed/blind_pseudotest/"
      "entropy_vs_cosine_neighbors_summary.csv"
)

MAX_MZ = 1500.0
BIN_WIDTH = 1.0

RDLogger.DisableLog("rdApp.*")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare current 1-Da cosine retrieval with Flash Entropy "
            "Search on the same training-only spectral library."
        )
    )

    parser.add_argument(
        "--max-library-structures",
        type=int,
        default=50_000,
        help=(
            "Maximum unique training structures retained in the benchmark "
            "library. One spectrum per structure/polarity is kept."
        ),
    )

    parser.add_argument(
        "--max-queries",
        type=int,
        default=250,
        help="Maximum blind molecules to evaluate.",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=25,
    )

    parser.add_argument(
        "--ms2-tolerance-da",
        type=float,
        default=0.02,
    )

    args = parser.parse_args()

    if args.max_library_structures <= 0:
        parser.error("--max-library-structures must be positive")

    if args.max_queries <= 0:
        parser.error("--max-queries must be positive")

    if args.top_k <= 0:
        parser.error("--top-k must be positive")

    if args.ms2_tolerance_da <= 0:
        parser.error("--ms2-tolerance-da must be positive")

    return args


def morgan_fp(smiles: str):
    mol = Chem.MolFromSmiles(str(smiles))

    if mol is None:
        return None

    return AllChem.GetMorganFingerprintAsBitVect(
        mol,
        radius=2,
        nBits=2048,
    )


def tanimoto(
    truth_fp,
    candidate_fp,
) -> float:
    if truth_fp is None or candidate_fp is None:
        return np.nan

    return float(
        DataStructs.TanimotoSimilarity(
            truth_fp,
            candidate_fp,
        )
    )


def make_peaks(
    mzs,
    intensities,
) -> np.ndarray:
    mz = np.asarray(
        mzs,
        dtype=np.float32,
    )

    intensity = np.asarray(
        intensities,
        dtype=np.float32,
    )

    finite = (
        np.isfinite(mz)
        & np.isfinite(intensity)
        & (mz > 0)
        & (intensity > 0)
    )

    if not finite.any():
        return np.empty(
            (0, 2),
            dtype=np.float32,
        )

    return np.column_stack(
        [
            mz[finite],
            intensity[finite],
        ]
    ).astype(
        np.float32,
        copy=False,
    )


def load_training_keys() -> set[str]:
    frame = pd.read_parquet(
        TRAIN_STRUCTURES_PATH,
        columns=["inchikey14"],
    )

    return set(
        frame["inchikey14"]
        .dropna()
        .astype(str)
    )


def build_reference_library(
    training_keys: set[str],
    max_structures: int,
) -> pd.DataFrame:
    parquet = pq.ParquetFile(
        TRAIN_PATH
    )

    columns = [
        "inchikey14",
        "normalized_smiles",
        "ionization_mode",
        "precursor_mz",
        "ms2_mzs",
        "ms2_normalized_intensities",
    ]

    seen = set()
    unique_structures = set()
    rows = []

    total_rows = parquet.metadata.num_rows
    scanned = 0

    print()
    print("Building benchmark reference library...")

    for batch in parquet.iter_batches(
        batch_size=50_000,
        columns=columns,
    ):
        frame = batch.to_pandas()
        scanned += len(frame)

        frame["inchikey14"] = (
            frame["inchikey14"]
            .astype(str)
        )

        frame = frame.loc[
            frame["inchikey14"].isin(
                training_keys
            )
            & frame["ionization_mode"].isin(
                [
                    "positive",
                    "negative",
                ]
            )
        ]

        for row in frame.itertuples(
            index=False
        ):
            key = (
                str(row.inchikey14),
                str(row.ionization_mode),
            )

            if key in seen:
                continue

            peaks = make_peaks(
                row.ms2_mzs,
                row.ms2_normalized_intensities,
            )

            if len(peaks) < 2:
                continue

            seen.add(key)
            unique_structures.add(
                str(row.inchikey14)
            )

            rows.append(
                {
                    "inchikey14":
                        str(row.inchikey14),
                    "normalized_smiles":
                        str(row.normalized_smiles),
                    "ionization_mode":
                        str(row.ionization_mode),
                    "precursor_mz":
                        float(row.precursor_mz),
                    "peaks":
                        peaks,
                }
            )

            if len(unique_structures) >= max_structures:
                break

        print(
            f"\rScanned {scanned:,}/{total_rows:,} spectra | "
            f"reference rows {len(rows):,}",
            end="",
            flush=True,
        )

        if len(unique_structures) >= max_structures:
            break

    print()

    library = pd.DataFrame(
        rows
    )

    if library.empty:
        raise RuntimeError(
            "Reference library is empty"
        )

    return library


def build_mode_indexes(
    library: pd.DataFrame,
):
    result = {}

    for mode, group in library.groupby(
        "ionization_mode",
        sort=True,
    ):
        group = group.reset_index(
            drop=True
        ).copy()

        entropy_library = [
            {
                "id":
                    int(index),
                "inchikey14":
                    str(row.inchikey14),
                "normalized_smiles":
                    str(row.normalized_smiles),
                "precursor_mz":
                    float(row.precursor_mz),
                "peaks":
                    row.peaks,
            }
            for index, row
            in enumerate(
                group.itertuples(
                    index=False
                )
            )
        ]

        entropy = FlashEntropySearch()

        indexed_library = entropy.build_index(
            entropy_library
        )

        indexed = pd.DataFrame(
            [
                {
                    "library_row_id":
                        int(item["id"]),
                    "inchikey14":
                        str(item["inchikey14"]),
                    "normalized_smiles":
                        str(item["normalized_smiles"]),
                    "precursor_mz":
                        float(item["precursor_mz"]),
                }
                for item
                in indexed_library
            ]
        )

        raw_peaks_by_id = {
            int(index):
                row.peaks
            for index, row
            in enumerate(
                group.itertuples(
                    index=False
                )
            )
        }

        cosine_vectors = sparse.csr_matrix(
            np.asarray(
                [
                    bin_spectrum(
                        raw_peaks_by_id[
                            int(item["id"])
                        ][:, 0],
                        raw_peaks_by_id[
                            int(item["id"])
                        ][:, 1],
                        max_mz=MAX_MZ,
                        bin_width=BIN_WIDTH,
                    )
                    for item
                    in indexed_library
                ],
                dtype=np.float32,
            )
        )

        result[
            str(mode)
        ] = {
            "entropy":
                entropy,
            "library":
                indexed,
            "cosine_vectors":
                cosine_vectors,
        }

        print(
            f"{mode}: indexed {len(indexed):,} reference spectra"
        )

    return result


def aggregate_structure_scores(
    keys: np.ndarray,
    scores: np.ndarray,
) -> dict[str, float]:
    best = {}

    for key, score in zip(
        keys,
        scores,
        strict=True,
    ):
        score = float(score)

        previous = best.get(
            str(key)
        )

        if (
            previous is None
            or score > previous
        ):
            best[
                str(key)
            ] = score

    return best


def merge_max(
    destination: dict[str, float],
    incoming: dict[str, float],
) -> None:
    for key, score in incoming.items():
        previous = destination.get(
            key
        )

        if (
            previous is None
            or score > previous
        ):
            destination[
                key
            ] = score


def top_keys(
    scores: dict[str, float],
    k: int,
) -> list[str]:
    return [
        key
        for key, _
        in sorted(
            scores.items(),
            key=lambda item: (
                -item[1],
                item[0],
            ),
        )[:k]
    ]


def summarize_similarity(
    method: str,
    molecule_id: str,
    top_structure_keys: list[str],
    truth_fp,
    fp_lookup: dict[str, object],
) -> list[dict]:
    rows = []

    similarities = [
        tanimoto(
            truth_fp,
            fp_lookup.get(key),
        )
        for key
        in top_structure_keys
    ]

    finite = [
        value
        for value in similarities
        if np.isfinite(value)
    ]

    for cutoff in [
        1,
        5,
        10,
        25,
    ]:
        subset = [
            value
            for value
            in similarities[:cutoff]
            if np.isfinite(value)
        ]

        rows.append(
            {
                "molecule_id":
                    molecule_id,
                "method":
                    method,
                "cutoff":
                    cutoff,
                "max_truth_tanimoto":
                    (
                        max(subset)
                        if subset
                        else np.nan
                    ),
                "mean_truth_tanimoto":
                    (
                        float(
                            np.mean(
                                subset
                            )
                        )
                        if subset
                        else np.nan
                    ),
                "candidate_count":
                    len(subset),
            }
        )

    return rows


def main() -> None:
    args = parse_args()

    for path in [
        TRAIN_PATH,
        TRAIN_STRUCTURES_PATH,
        BLIND_TEST_PATH,
        BLIND_TRUTH_PATH,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"Missing required artifact: {path}"
            )

    training_keys = load_training_keys()

    library = build_reference_library(
        training_keys,
        args.max_library_structures,
    )

    mode_indexes = build_mode_indexes(
        library
    )

    structure_rows = (
        library[
            [
                "inchikey14",
                "normalized_smiles",
            ]
        ]
        .drop_duplicates(
            subset=[
                "inchikey14",
            ]
        )
    )

    fp_lookup = {}

    for row in structure_rows.itertuples(
        index=False
    ):
        fp_lookup[
            str(row.inchikey14)
        ] = morgan_fp(
            row.normalized_smiles
        )

    blind = pd.read_parquet(
        BLIND_TEST_PATH
    ).copy()

    blind["molecule_id"] = (
        blind["molecule_id"]
        .astype(str)
    )

    selected = select_query_spectra(
        blind
    )

    truth = pd.read_parquet(
        BLIND_TRUTH_PATH,
        columns=[
            "molecule_id",
            "normalized_smiles",
        ],
    ).copy()

    truth["molecule_id"] = (
        truth["molecule_id"]
        .astype(str)
    )

    truth_smiles = dict(
        zip(
            truth["molecule_id"],
            truth["normalized_smiles"],
            strict=True,
        )
    )

    molecule_ids = (
        selected["molecule_id"]
        .drop_duplicates()
        .astype(str)
        .head(
            args.max_queries
        )
        .tolist()
    )

    result_rows = []

    print()
    print(
        "Comparing cosine and entropy neighbours..."
    )

    for number, molecule_id in enumerate(
        molecule_ids,
        start=1,
    ):
        spectra = selected.loc[
            selected["molecule_id"]
            == molecule_id
        ]

        cosine_scores = {}
        entropy_scores = {}

        for query in spectra.itertuples(
            index=False
        ):
            mode = str(
                query.ionization_mode
            )

            index = mode_indexes.get(
                mode
            )

            if index is None:
                continue

            query_peaks = make_peaks(
                query.ms2_mzs,
                query.ms2_normalized_intensities,
            )

            if len(query_peaks) < 2:
                continue

            query_vector = bin_spectrum(
                query_peaks[:, 0],
                query_peaks[:, 1],
                max_mz=MAX_MZ,
                bin_width=BIN_WIDTH,
            )

            cosine = (
                index[
                    "cosine_vectors"
                ]
                .dot(
                    query_vector
                )
            )

            cosine = np.asarray(
                cosine
            ).reshape(-1)

            entropy_result = (
                index[
                    "entropy"
                ]
                .search(
                    precursor_mz=float(
                        query.precursor_mz
                    ),
                    peaks=query_peaks,
                    method="open",
                    ms2_tolerance_in_da=(
                        args.ms2_tolerance_da
                    ),
                )
            )

            entropy_values = np.asarray(
                entropy_result[
                    "open_search"
                ]
            )

            keys = (
                index[
                    "library"
                ][
                    "inchikey14"
                ]
                .astype(str)
                .to_numpy()
            )

            merge_max(
                cosine_scores,
                aggregate_structure_scores(
                    keys,
                    cosine,
                ),
            )

            merge_max(
                entropy_scores,
                aggregate_structure_scores(
                    keys,
                    entropy_values,
                ),
            )

        truth_fp = morgan_fp(
            truth_smiles[
                molecule_id
            ]
        )

        result_rows.extend(
            summarize_similarity(
                "cosine_1da",
                molecule_id,
                top_keys(
                    cosine_scores,
                    args.top_k,
                ),
                truth_fp,
                fp_lookup,
            )
        )

        result_rows.extend(
            summarize_similarity(
                "flash_entropy_open",
                molecule_id,
                top_keys(
                    entropy_scores,
                    args.top_k,
                ),
                truth_fp,
                fp_lookup,
            )
        )

        if (
            number % 25 == 0
            or number
            == len(molecule_ids)
        ):
            print(
                f"  queries {number:,}/{len(molecule_ids):,}"
            )

    results = pd.DataFrame(
        result_rows
    )

    if results.empty:
        raise RuntimeError(
            "No benchmark results were produced"
        )

    summary = (
        results.groupby(
            [
                "method",
                "cutoff",
            ],
            as_index=False,
        )
        .agg(
            queries=(
                "molecule_id",
                "nunique",
            ),
            mean_max_truth_tanimoto=(
                "max_truth_tanimoto",
                "mean",
            ),
            median_max_truth_tanimoto=(
                "max_truth_tanimoto",
                "median",
            ),
            mean_truth_tanimoto=(
                "mean_truth_tanimoto",
                "mean",
            ),
        )
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results.to_parquet(
        OUTPUT_PATH,
        index=False,
        compression="zstd",
    )

    summary.to_csv(
        SUMMARY_PATH,
        index=False,
    )

    print()
    print(
        "Entropy vs cosine neighbour benchmark"
    )
    print("=" * 96)
    print(
        summary.to_string(
            index=False,
            formatters={
                "mean_max_truth_tanimoto":
                    lambda value:
                        f"{value:.4f}",
                "median_max_truth_tanimoto":
                    lambda value:
                        f"{value:.4f}",
                "mean_truth_tanimoto":
                    lambda value:
                        f"{value:.4f}",
            },
        )
    )
    print()
    print("Wrote:", OUTPUT_PATH)
    print("Wrote:", SUMMARY_PATH)


if __name__ == "__main__":
    main()
