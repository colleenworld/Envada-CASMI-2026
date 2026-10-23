from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem
from scipy import sparse

from casmi26.retrieval.index import load_retrieval_index
from casmi26.retrieval.indexed import search_structures_indexed_blockwise
from casmi26.spectra.binning import bin_spectrum


SPECTRAL_NEIGHBORS = 25
BATCH_STRUCTURES = 16
MAX_MZ = 1500.0
BIN_WIDTH = 1.0

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

STRICT_INDEX_DIR = Path(
    "data/processed/retrieval_index_strict_validation"
)

TRAINING_STRUCTURES_PATH = Path(
    "data/processed/structure_index/training_structures.parquet"
)

PUBCHEM_ROOT = Path(
    "data/processed/pubchem_formula_index"
)

OUTPUT_PATH = (
    ROOT
    / "learned_ranker_top500/neighbor_enrichment.parquet"
)

RDLogger.DisableLog("rdApp.*")


def morgan_fp(smiles: str):
    mol = Chem.MolFromSmiles(
        str(smiles)
    )

    if mol is None:
        return None

    return AllChem.GetMorganFingerprintAsBitVect(
        mol,
        radius=2,
        nBits=4096,
    )


def top_neighbors_from_scores(
    scores: np.ndarray,
    candidate_keys: np.ndarray,
    count: int,
) -> list[tuple[str, float]]:
    finite = np.flatnonzero(
        np.isfinite(
            scores
        )
    )

    if not len(
        finite
    ):
        return []

    finite_scores = scores[
        finite
    ]

    order = np.lexsort(
        (
            finite,
            -finite_scores,
        )
    )

    selected = finite[
        order[
            :count
        ]
    ]

    return [
        (
            str(
                candidate_keys[
                    candidate_id
                ]
            ),
            float(
                scores[
                    candidate_id
                ]
            ),
        )
        for candidate_id
        in selected
    ]


def main() -> None:
    required = [
        CANDIDATES_PATH,
        SAMPLE_PATH,
        TRAIN_PATH,
        STRICT_INDEX_DIR
        / "metadata.parquet",
        STRICT_INDEX_DIR
        / "vectors.npz",
        TRAINING_STRUCTURES_PATH,
        PUBCHEM_ROOT
        / "formula_buckets.parquet",
    ]

    missing = [
        path
        for path in required
        if not path.exists()
    ]

    if missing:
        raise FileNotFoundError(
            "Missing required artifacts: "
            + ", ".join(
                str(path)
                for path in missing
            )
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

    sample = (
        pd.read_parquet(
            SAMPLE_PATH
        )[
            [
                "inchikey14",
            ]
        ]
        .drop_duplicates()
        .reset_index(
            drop=True
        )
    )

    sample[
        "inchikey14"
    ] = (
        sample[
            "inchikey14"
        ]
        .astype(str)
    )

    sample_keys = set(
        sample[
            "inchikey14"
        ]
    )

    sample[
        "benchmark_id"
    ] = np.arange(
        len(sample),
        dtype=np.int64,
    )

    benchmark_id_by_key = dict(
        zip(
            sample[
                "inchikey14"
            ],
            sample[
                "benchmark_id"
            ].astype(int),
            strict=True,
        )
    )

    print(
        "Candidate rows:",
        f"{len(candidates):,}",
    )
    print(
        "Queries:",
        f"{candidates['inchikey14'].nunique():,}",
    )

    #
    # Rebuild the exact leakage-safe spectral-neighbor lists.
    #
    parquet = pq.ParquetFile(
        TRAIN_PATH
    )

    spectra_parts = []
    scanned = 0

    for batch in parquet.iter_batches(
        batch_size=50_000,
        columns=[
            "inchikey14",
            "ionization_mode",
            "ms2_mzs",
            "ms2_normalized_intensities",
        ],
    ):
        frame = batch.to_pandas()
        scanned += len(
            frame
        )

        matched = frame.loc[
            frame[
                "inchikey14"
            ]
            .astype(str)
            .isin(
                sample_keys
            )
        ].copy()

        if not matched.empty:
            spectra_parts.append(
                matched
            )

        print(
            f"\rRows scanned: {scanned:,}",
            end="",
            flush=True,
        )

    print()

    if not spectra_parts:
        raise RuntimeError(
            "No benchmark spectra found"
        )

    spectra = pd.concat(
        spectra_parts,
        ignore_index=True,
    )

    query_vectors = []
    query_structure_ids = []
    query_modes = []

    for row in spectra.itertuples(
        index=False
    ):
        query_vectors.append(
            bin_spectrum(
                row.ms2_mzs,
                row.ms2_normalized_intensities,
                max_mz=MAX_MZ,
                bin_width=BIN_WIDTH,
            )
        )

        query_structure_ids.append(
            benchmark_id_by_key[
                str(
                    row.inchikey14
                )
            ]
        )

        query_modes.append(
            str(
                row.ionization_mode
            )
        )

    query_vectors = sparse.csr_matrix(
        np.asarray(
            query_vectors,
            dtype=np.float32,
        )
    )

    query_structure_ids = np.asarray(
        query_structure_ids,
        dtype=np.int64,
    )

    query_modes = np.asarray(
        query_modes,
        dtype=object,
    )

    metadata, reference_vectors = (
        load_retrieval_index(
            STRICT_INDEX_DIR
        )
    )

    reference_keys = (
        metadata[
            "inchikey14"
        ]
        .astype(str)
        .to_numpy()
    )

    (
        reference_candidate_keys,
        reference_candidate_ids,
    ) = np.unique(
        reference_keys,
        return_inverse=True,
    )

    reference_candidate_ids = (
        reference_candidate_ids
        .astype(
            np.int64,
            copy=False,
        )
    )

    reference_modes = (
        metadata[
            "ionization_mode"
        ]
        .astype(str)
        .to_numpy()
    )

    neighbors_by_id: dict[
        int,
        list[tuple[str, float]],
    ] = {}

    benchmark_ids = (
        sample[
            "benchmark_id"
        ]
        .astype(int)
        .to_numpy()
    )

    for start in range(
        0,
        len(benchmark_ids),
        BATCH_STRUCTURES,
    ):
        batch_ids = benchmark_ids[
            start:
            start
            + BATCH_STRUCTURES
        ]

        query_mask = np.isin(
            query_structure_ids,
            batch_ids,
        )

        original_ids = (
            query_structure_ids[
                query_mask
            ]
        )

        local_id_by_original = {
            int(original):
                local
            for local, original
            in enumerate(
                batch_ids
            )
        }

        local_structure_ids = np.asarray(
            [
                local_id_by_original[
                    int(original)
                ]
                for original
                in original_ids
            ],
            dtype=np.int64,
        )

        score_matrix = (
            search_structures_indexed_blockwise(
                query_vectors=query_vectors[
                    query_mask
                ],
                query_structure_ids=local_structure_ids,
                query_modes=query_modes[
                    query_mask
                ],
                reference_vectors=reference_vectors,
                reference_candidate_ids=reference_candidate_ids,
                reference_modes=reference_modes,
                query_structure_count=len(
                    batch_ids
                ),
                candidate_count=len(
                    reference_candidate_keys
                ),
                block_size=25_000,
            )
        )

        for local_id, benchmark_id in enumerate(
            batch_ids
        ):
            neighbors_by_id[
                int(benchmark_id)
            ] = top_neighbors_from_scores(
                score_matrix[
                    local_id
                ],
                reference_candidate_keys,
                SPECTRAL_NEIGHBORS,
            )

        completed = min(
            start
            + len(batch_ids),
            len(benchmark_ids),
        )

        print(
            "Retrieved neighbors:",
            f"{completed:,}/{len(benchmark_ids):,}",
        )

        del score_matrix

    neighbor_keys = {
        key
        for neighbors
        in neighbors_by_id.values()
        for key, _
        in neighbors
    }

    training_structures = pd.read_parquet(
        TRAINING_STRUCTURES_PATH,
        columns=[
            "inchikey14",
            "normalized_smiles",
        ],
    )

    training_structures[
        "inchikey14"
    ] = (
        training_structures[
            "inchikey14"
        ]
        .astype(str)
    )

    training_fps = {}

    for row in (
        training_structures.loc[
            training_structures[
                "inchikey14"
            ]
            .isin(
                neighbor_keys
            )
        ]
        .itertuples(
            index=False
        )
    ):
        fp = morgan_fp(
            row.normalized_smiles
        )

        if fp is not None:
            training_fps[
                str(
                    row.inchikey14
                )
            ] = fp

    print(
        "Usable neighbor fingerprints:",
        f"{len(training_fps):,}",
    )

    #
    # Load smiles only for the already-retained top-500 candidates.
    #
    formulas = set(
        candidates[
            "candidate_formula"
        ]
    )

    keys_by_formula = {
        formula:
            set(
                group[
                    "candidate_inchikey14"
                ]
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

    smiles_by_key: dict[
        str,
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
            key = str(
                row.inchikey14
            )

            wanted = (
                keys_by_formula.get(
                    str(
                        row.molecular_formula
                    )
                )
            )

            if (
                wanted is not None
                and key in wanted
                and key not in smiles_by_key
            ):
                smiles_by_key[
                    key
                ] = str(
                    row.smiles
                )

        print(
            "Loaded PubChem bucket:",
            f"{bucket:03d}",
        )

    candidate_fps = {}

    for key in sorted(
        set(
            candidates[
                "candidate_inchikey14"
            ]
        )
    ):
        smiles = smiles_by_key.get(
            key
        )

        if smiles is None:
            continue

        fp = morgan_fp(
            smiles
        )

        if fp is not None:
            candidate_fps[
                key
            ] = fp

    print(
        "Usable retained candidate fingerprints:",
        f"{len(candidate_fps):,}",
    )

    #
    # Enrich each retained query/candidate row with the full distribution
    # of structural similarity to the 25 spectral neighbors.
    #
    output_frames = []

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
        benchmark_id = (
            benchmark_id_by_key[
                str(query_key)
            ]
        )

        raw_neighbors = (
            neighbors_by_id[
                benchmark_id
            ]
        )

        usable = [
            (
                key,
                max(
                    float(
                        spectral_score
                    ),
                    0.0,
                ),
                training_fps[
                    key
                ],
            )
            for key, spectral_score
            in raw_neighbors
            if key in training_fps
        ]

        enriched = group[
            [
                "inchikey14",
                "candidate_formula",
                "candidate_inchikey14",
            ]
        ].copy()

        n = len(
            enriched
        )

        tanimoto_max = np.full(
            n,
            np.nan,
            dtype=np.float32,
        )
        tanimoto_mean = np.full(
            n,
            np.nan,
            dtype=np.float32,
        )
        tanimoto_top3_mean = np.full(
            n,
            np.nan,
            dtype=np.float32,
        )
        tanimoto_top5_mean = np.full(
            n,
            np.nan,
            dtype=np.float32,
        )
        weighted_mean = np.full(
            n,
            np.nan,
            dtype=np.float32,
        )
        weighted_top3_mean = np.full(
            n,
            np.nan,
            dtype=np.float32,
        )
        count_ge_03 = np.zeros(
            n,
            dtype=np.int16,
        )
        count_ge_05 = np.zeros(
            n,
            dtype=np.int16,
        )
        count_ge_07 = np.zeros(
            n,
            dtype=np.int16,
        )
        best_neighbor_spectral_score = np.full(
            n,
            np.nan,
            dtype=np.float32,
        )

        valid_positions = []
        valid_fps = []

        for position, key in enumerate(
            enriched[
                "candidate_inchikey14"
            ]
        ):
            fp = candidate_fps.get(
                str(key)
            )

            if fp is not None:
                valid_positions.append(
                    position
                )
                valid_fps.append(
                    fp
                )

        if usable and valid_fps:
            similarities = np.empty(
                (
                    len(usable),
                    len(valid_fps),
                ),
                dtype=np.float32,
            )

            weights = np.asarray(
                [
                    spectral_score
                    for _,
                    spectral_score,
                    _
                    in usable
                ],
                dtype=np.float32,
            )

            for neighbor_index, (
                _,
                _,
                neighbor_fp,
            ) in enumerate(
                usable
            ):
                similarities[
                    neighbor_index
                ] = np.asarray(
                    DataStructs.BulkTanimotoSimilarity(
                        neighbor_fp,
                        valid_fps,
                    ),
                    dtype=np.float32,
                )

            sorted_sim = np.sort(
                similarities,
                axis=0,
            )[
                ::-1
            ]

            weighted = (
                similarities
                * weights[
                    :,
                    None
                ]
            )

            sorted_weighted = np.sort(
                weighted,
                axis=0,
            )[
                ::-1
            ]

            max_neighbor_idx = np.argmax(
                similarities,
                axis=0,
            )

            weight_sum = float(
                weights.sum()
            )

            for local_column, position in enumerate(
                valid_positions
            ):
                values = similarities[
                    :,
                    local_column
                ]

                tanimoto_max[
                    position
                ] = float(
                    values.max()
                )

                tanimoto_mean[
                    position
                ] = float(
                    values.mean()
                )

                tanimoto_top3_mean[
                    position
                ] = float(
                    sorted_sim[
                        :min(
                            3,
                            len(usable),
                        ),
                        local_column,
                    ].mean()
                )

                tanimoto_top5_mean[
                    position
                ] = float(
                    sorted_sim[
                        :min(
                            5,
                            len(usable),
                        ),
                        local_column,
                    ].mean()
                )

                if weight_sum > 0:
                    weighted_mean[
                        position
                    ] = float(
                        weighted[
                            :,
                            local_column
                        ].sum()
                        / weight_sum
                    )

                weighted_top3_mean[
                    position
                ] = float(
                    sorted_weighted[
                        :min(
                            3,
                            len(usable),
                        ),
                        local_column,
                    ].mean()
                )

                count_ge_03[
                    position
                ] = int(
                    np.sum(
                        values
                        >= 0.3
                    )
                )

                count_ge_05[
                    position
                ] = int(
                    np.sum(
                        values
                        >= 0.5
                    )
                )

                count_ge_07[
                    position
                ] = int(
                    np.sum(
                        values
                        >= 0.7
                    )
                )

                best_neighbor_spectral_score[
                    position
                ] = float(
                    weights[
                        max_neighbor_idx[
                            local_column
                        ]
                    ]
                )

        enriched[
            "neighbor_tanimoto_max"
        ] = tanimoto_max
        enriched[
            "neighbor_tanimoto_mean"
        ] = tanimoto_mean
        enriched[
            "neighbor_tanimoto_top3_mean"
        ] = tanimoto_top3_mean
        enriched[
            "neighbor_tanimoto_top5_mean"
        ] = tanimoto_top5_mean
        enriched[
            "neighbor_weighted_mean"
        ] = weighted_mean
        enriched[
            "neighbor_weighted_top3_mean"
        ] = weighted_top3_mean
        enriched[
            "neighbor_count_ge_03"
        ] = count_ge_03
        enriched[
            "neighbor_count_ge_05"
        ] = count_ge_05
        enriched[
            "neighbor_count_ge_07"
        ] = count_ge_07
        enriched[
            "best_structural_neighbor_spectral_score"
        ] = best_neighbor_spectral_score
        enriched[
            "neighbor_feature_count"
        ] = len(
            usable
        )

        output_frames.append(
            enriched
        )

        if (
            query_number % 100 == 0
            or query_number
            == candidates[
                "inchikey14"
            ].nunique()
        ):
            print(
                "Enriched queries:",
                f"{query_number:,}/"
                f"{candidates['inchikey14'].nunique():,}",
            )

    result = pd.concat(
        output_frames,
        ignore_index=True,
    )

    if len(result) != len(
        candidates
    ):
        raise RuntimeError(
            "Enrichment row count does not match candidate artifact"
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
            f"Enrichment contains {duplicates:,} duplicate rows"
        )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_parquet(
        OUTPUT_PATH,
        index=False,
        compression="zstd",
    )

    print()
    print(
        "Top-500 neighbor enrichment"
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
    print()
    print(
        result[
            [
                "neighbor_tanimoto_max",
                "neighbor_tanimoto_mean",
                "neighbor_tanimoto_top3_mean",
                "neighbor_tanimoto_top5_mean",
                "neighbor_weighted_mean",
                "neighbor_weighted_top3_mean",
                "neighbor_count_ge_03",
                "neighbor_count_ge_05",
                "neighbor_count_ge_07",
                "best_structural_neighbor_spectral_score",
            ]
        ]
        .describe()
        .to_string()
    )
    print()
    print(
        "Wrote:",
        OUTPUT_PATH,
    )


if __name__ == "__main__":
    main()
