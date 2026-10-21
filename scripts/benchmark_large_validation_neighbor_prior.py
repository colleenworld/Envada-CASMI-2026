from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

from casmi26.pubchem.neighbor_pruning import score_candidates
from casmi26.retrieval.index import load_retrieval_index
from casmi26.retrieval.indexed import search_structures_indexed_blockwise
from casmi26.spectra.binning import bin_spectrum


SEED = 42
SPECTRAL_NEIGHBORS = 25
SAVE_TOP_K = 500
BATCH_STRUCTURES = 16
MAX_MZ = 1500.0
BIN_WIDTH = 1.0

BURDEN_BUCKETS = [
    ("le100", 1, 100),
    ("101to500", 101, 500),
    ("501to1000", 501, 1000),
    ("1001to5000", 1001, 5000),
    ("gt5000", 5001, None),
]

RDLogger.DisableLog("rdApp.*")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--per-bucket",
        type=int,
        default=500,
        help="Maximum structures sampled from each true-formula burden bucket.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
    )
    return parser.parse_args()


def morgan_fp(smiles: str):
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None

    return AllChem.GetMorganFingerprintAsBitVect(
        mol,
        radius=2,
        nBits=4096,
    )


def rr25(rank) -> float:
    if rank is None or pd.isna(rank):
        return 0.0

    rank = int(rank)
    if rank < 1 or rank > 25:
        return 0.0

    return 1.0 / rank


def top_neighbors_from_scores(
    scores: np.ndarray,
    candidate_keys: np.ndarray,
    count: int,
) -> list[tuple[str, float]]:
    finite = np.flatnonzero(
        np.isfinite(scores)
    )

    if not len(finite):
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
        order[:count]
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
        for candidate_id in selected
    ]


def main() -> None:
    args = parse_args()

    if args.per_bucket <= 0:
        raise ValueError(
            "--per-bucket must be positive"
        )

    project_root = Path(
        os.environ.get(
            "CASMI26_PROJECT_ROOT",
            Path(__file__).resolve().parent.parent,
        )
    )

    audit_dir = (
        project_root
        / "data/processed/diagnostics/large_validation_audit"
    )

    coverage_path = (
        audit_dir
        / "pubchem_coverage.parquet"
    )

    train_path = (
        project_root
        / "data/raw/train.parquet"
    )

    strict_index_dir = (
        project_root
        / "data/processed/retrieval_index_strict_validation"
    )

    training_structures_path = (
        project_root
        / "data/processed/structure_index/training_structures.parquet"
    )

    pubchem_root = (
        project_root
        / "data/processed/pubchem_formula_index"
    )

    output_dir = (
        audit_dir
        / "true_formula_neighbor_benchmark"
    )
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    sample_path = (
        output_dir
        / "sample.parquet"
    )
    ranking_path = (
        output_dir
        / "neighbor_prior_truth_ranks.parquet"
    )
    candidate_ranking_path = (
        output_dir
        / "neighbor_prior_top500.parquet"
    )

    for required in [
        coverage_path,
        train_path,
        strict_index_dir / "metadata.parquet",
        strict_index_dir / "vectors.npz",
        training_structures_path,
        pubchem_root / "formula_buckets.parquet",
    ]:
        if not required.exists():
            raise FileNotFoundError(
                f"Missing required artifact: {required}"
            )

    coverage = pd.read_parquet(
        coverage_path
    )

    eligible = coverage.loc[
        coverage[
            "truth_formula_recovered"
        ].fillna(False)
        & coverage[
            "truth_structure_in_pubchem_hypotheses"
        ].fillna(False)
        & (
            coverage[
                "truth_formula_pubchem_candidate_count"
            ]
            > 0
        )
    ].copy()

    eligible[
        "burden_bucket"
    ] = None

    for name, low, high in BURDEN_BUCKETS:
        mask = (
            eligible[
                "truth_formula_pubchem_candidate_count"
            ]
            >= low
        )

        if high is not None:
            mask &= (
                eligible[
                    "truth_formula_pubchem_candidate_count"
                ]
                <= high
            )

        eligible.loc[
            mask,
            "burden_bucket",
        ] = name

    eligible = eligible.loc[
        eligible[
            "burden_bucket"
        ].notna()
    ].copy()

    rng = np.random.default_rng(
        args.seed
    )

    sample_parts = []

    print(
        "Eligible truth-formula/PubChem structures:",
        f"{len(eligible):,}",
    )

    for name, _, _ in BURDEN_BUCKETS:
        bucket = eligible.loc[
            eligible[
                "burden_bucket"
            ]
            == name
        ].copy()

        take = min(
            args.per_bucket,
            len(bucket),
        )

        if take:
            chosen = rng.choice(
                bucket.index.to_numpy(),
                size=take,
                replace=False,
            )

            sample_parts.append(
                bucket.loc[
                    chosen
                ].copy()
            )

        print(
            f"{name:>10}: "
            f"eligible={len(bucket):,} "
            f"selected={take:,}"
        )

    if not sample_parts:
        raise RuntimeError(
            "No benchmark structures selected"
        )

    sample = (
        pd.concat(
            sample_parts,
            ignore_index=True,
        )
        .sort_values(
            [
                "burden_bucket",
                "inchikey14",
            ]
        )
        .reset_index(drop=True)
    )

    sample[
        "benchmark_id"
    ] = np.arange(
        len(sample),
        dtype=np.int64,
    )

    sample.to_parquet(
        sample_path,
        index=False,
    )

    print()
    print(
        "Benchmark structures:",
        f"{len(sample):,}",
    )

    print(
        "Dominant libraries:"
    )
    print(
        sample[
            "dominant_library"
        ]
        .value_counts()
        .to_string()
    )

    #
    # Collect all spectra for benchmark structures and bin them once.
    #
    selected_keys = set(
        sample[
            "inchikey14"
        ]
        .astype(str)
    )

    parquet = pq.ParquetFile(
        train_path
    )

    spectra_parts = []

    columns = [
        "inchikey14",
        "ionization_mode",
        "ms2_mzs",
        "ms2_normalized_intensities",
    ]

    scanned = 0

    print()
    print(
        "Collecting benchmark spectra..."
    )

    for batch in parquet.iter_batches(
        batch_size=50_000,
        columns=columns,
    ):
        frame = batch.to_pandas()
        scanned += len(frame)

        matched = frame.loc[
            frame[
                "inchikey14"
            ]
            .astype(str)
            .isin(
                selected_keys
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

    spectra = pd.concat(
        spectra_parts,
        ignore_index=True,
    )

    print(
        "Benchmark spectra:",
        f"{len(spectra):,}",
    )

    query_vectors = []
    query_structure_ids = []
    query_modes = []

    benchmark_id_by_key = dict(
        zip(
            sample[
                "inchikey14"
            ].astype(str),
            sample[
                "benchmark_id"
            ].astype(int),
            strict=True,
        )
    )

    for row in spectra.itertuples(
        index=False
    ):
        vector = bin_spectrum(
            row.ms2_mzs,
            row.ms2_normalized_intensities,
            max_mz=MAX_MZ,
            bin_width=BIN_WIDTH,
        )

        query_vectors.append(
            vector
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

    from scipy import sparse

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

    #
    # Strict training-only retrieval index.
    #
    print()
    print(
        "Loading strict validation retrieval index..."
    )

    metadata, reference_vectors = (
        load_retrieval_index(
            strict_index_dir
        )
    )

    reference_keys = (
        metadata[
            "inchikey14"
        ]
        .astype(str)
        .to_numpy()
    )

    candidate_keys, reference_candidate_ids = (
        np.unique(
            reference_keys,
            return_inverse=True,
        )
    )

    reference_candidate_ids = (
        reference_candidate_ids.astype(
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

    print(
        "Reference spectra:",
        f"{len(metadata):,}",
    )
    print(
        "Reference structures:",
        f"{len(candidate_keys):,}",
    )

    #
    # Retrieve top training-structure neighbors in small batches.
    #
    neighbors_by_id = {}

    benchmark_ids = sample[
        "benchmark_id"
    ].astype(int).to_numpy()

    for start in range(
        0,
        len(benchmark_ids),
        BATCH_STRUCTURES,
    ):
        batch_ids = benchmark_ids[
            start:
            start + BATCH_STRUCTURES
        ]

        query_mask = np.isin(
            query_structure_ids,
            batch_ids,
        )

        batch_vectors = query_vectors[
            query_mask
        ]
        batch_modes = query_modes[
            query_mask
        ]
        original_ids = query_structure_ids[
            query_mask
        ]

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
                for original in original_ids
            ],
            dtype=np.int64,
        )

        score_matrix = (
            search_structures_indexed_blockwise(
                query_vectors=batch_vectors,
                query_structure_ids=(
                    local_structure_ids
                ),
                query_modes=batch_modes,
                reference_vectors=(
                    reference_vectors
                ),
                reference_candidate_ids=(
                    reference_candidate_ids
                ),
                reference_modes=(
                    reference_modes
                ),
                query_structure_count=(
                    len(batch_ids)
                ),
                candidate_count=(
                    len(candidate_keys)
                ),
                block_size=25_000,
            )
        )

        for local_id, benchmark_id in enumerate(
            batch_ids
        ):
            neighbors_by_id[
                int(
                    benchmark_id
                )
            ] = (
                top_neighbors_from_scores(
                    score_matrix[
                        local_id
                    ],
                    candidate_keys,
                    SPECTRAL_NEIGHBORS,
                )
            )

        completed = min(
            start + len(batch_ids),
            len(benchmark_ids),
        )

        print(
            "Retrieved neighbors:",
            f"{completed:,}/"
            f"{len(benchmark_ids):,}",
        )

        del score_matrix

    #
    # Fingerprint only training structures that actually appear as
    # spectral neighbors.
    #
    neighbor_keys = {
        key
        for neighbors
        in neighbors_by_id.values()
        for key, _
        in neighbors
    }

    training_structures = pd.read_parquet(
        training_structures_path,
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

    anchor_rows = training_structures.loc[
        training_structures[
            "inchikey14"
        ]
        .isin(
            neighbor_keys
        )
    ]

    training_fps = {}

    for row in anchor_rows.itertuples(
        index=False
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

    print()
    print(
        "Unique spectral-neighbor structures:",
        f"{len(neighbor_keys):,}",
    )
    print(
        "Usable neighbor fingerprints:",
        f"{len(training_fps):,}",
    )

    #
    # Score PubChem candidates bucket-by-bucket so high-burden formulas
    # do not need to be loaded simultaneously.
    #
    formula_buckets = pd.read_parquet(
        pubchem_root
        / "formula_buckets.parquet"
    )

    formula_buckets[
        "molecular_formula"
    ] = (
        formula_buckets[
            "molecular_formula"
        ]
        .astype(str)
    )

    selected_formulas = set(
        sample[
            "truth_formula"
        ]
        .astype(str)
    )

    selected_formula_buckets = (
        formula_buckets.loc[
            formula_buckets[
                "molecular_formula"
            ]
            .isin(
                selected_formulas
            ),
            [
                "molecular_formula",
                "bucket",
            ],
        ]
    )

    samples_by_formula = {
        formula: group.copy()
        for formula, group
        in sample.groupby(
            "truth_formula",
            sort=False,
        )
    }

    result_rows = []
    candidate_ranking_rows = []
    processed = 0

    for bucket in sorted(
        selected_formula_buckets[
            "bucket"
        ]
        .astype(int)
        .unique()
    ):
        formulas_in_bucket = set(
            selected_formula_buckets.loc[
                selected_formula_buckets[
                    "bucket"
                ].astype(int)
                == bucket,
                "molecular_formula",
            ]
            .astype(str)
        )

        candidates = pd.read_parquet(
            pubchem_root
            / "candidates"
            / f"bucket-{bucket:03d}.parquet",
            columns=[
                "molecular_formula",
                "inchikey14",
                "smiles",
            ],
        )

        candidates[
            "molecular_formula"
        ] = (
            candidates[
                "molecular_formula"
            ]
            .astype(str)
        )

        candidates = candidates.loc[
            candidates[
                "molecular_formula"
            ]
            .isin(
                formulas_in_bucket
            )
        ].drop_duplicates(
            subset=[
                "molecular_formula",
                "inchikey14",
            ]
        )

        for formula, formula_samples in (
            samples_by_formula.items()
        ):
            if formula not in formulas_in_bucket:
                continue

            formula_candidates = (
                candidates.loc[
                    candidates[
                        "molecular_formula"
                    ]
                    == formula
                ]
                .reset_index(
                    drop=True
                )
            )

            valid_keys = []
            valid_fps = []

            for row in formula_candidates.itertuples(
                index=False
            ):
                fp = morgan_fp(
                    row.smiles
                )

                if fp is None:
                    continue

                valid_keys.append(
                    str(
                        row.inchikey14
                    )
                )
                valid_fps.append(
                    fp
                )

            if not valid_fps:
                for query in formula_samples.itertuples(
                    index=False
                ):
                    result_rows.append(
                        {
                            "benchmark_id":
                                int(
                                    query.benchmark_id
                                ),
                            "inchikey14":
                                str(
                                    query.inchikey14
                                ),
                            "truth_formula":
                                formula,
                            "burden_bucket":
                                str(
                                    query.burden_bucket
                                ),
                            "dominant_library":
                                str(
                                    query.dominant_library
                                ),
                            "candidate_count":
                                0,
                            "usable_neighbor_fingerprints":
                                0,
                            "truth_rank":
                                np.nan,
                        }
                    )
                continue

            valid_keys_array = np.asarray(
                valid_keys,
                dtype=object,
            )

            for query in formula_samples.itertuples(
                index=False
            ):
                neighbors = neighbors_by_id[
                    int(
                        query.benchmark_id
                    )
                ]

                scores, usable_neighbors = (
                    score_candidates(
                        valid_fps,
                        neighbors,
                        training_fps,
                    )
                )

                order = np.lexsort(
                    (
                        valid_keys_array,
                        -scores,
                    )
                )

                ordered_keys = (
                    valid_keys_array[
                        order
                    ]
                )

                top_count = min(
                    SAVE_TOP_K,
                    len(order),
                )

                for prior_rank, candidate_index in enumerate(
                    order[:top_count],
                    start=1,
                ):
                    candidate_ranking_rows.append(
                        {
                            "benchmark_id":
                                int(
                                    query.benchmark_id
                                ),
                            "inchikey14":
                                str(
                                    query.inchikey14
                                ),
                            "candidate_inchikey14":
                                str(
                                    valid_keys_array[
                                        candidate_index
                                    ]
                                ),
                            "prior_rank":
                                prior_rank,
                            "neighbor_prior_score":
                                float(
                                    scores[
                                        candidate_index
                                    ]
                                ),
                            "burden_bucket":
                                str(
                                    query.burden_bucket
                                ),
                            "dominant_library":
                                str(
                                    query.dominant_library
                                ),
                            "candidate_count":
                                len(
                                    valid_keys_array
                                ),
                        }
                    )

                truth_positions = np.flatnonzero(
                    ordered_keys
                    == str(
                        query.inchikey14
                    )
                )

                truth_rank = (
                    int(
                        truth_positions[0]
                    )
                    + 1
                    if len(
                        truth_positions
                    )
                    else np.nan
                )

                result_rows.append(
                    {
                        "benchmark_id":
                            int(
                                query.benchmark_id
                            ),
                        "inchikey14":
                            str(
                                query.inchikey14
                            ),
                        "truth_formula":
                            formula,
                        "burden_bucket":
                            str(
                                query.burden_bucket
                            ),
                        "dominant_library":
                            str(
                                query.dominant_library
                            ),
                        "candidate_count":
                            len(
                                valid_keys_array
                            ),
                        "usable_neighbor_fingerprints":
                            usable_neighbors,
                        "truth_rank":
                            truth_rank,
                    }
                )

                processed += 1

                if (
                    processed % 100 == 0
                    or processed
                    == len(sample)
                ):
                    print(
                        "Ranked structures:",
                        f"{processed:,}/"
                        f"{len(sample):,}",
                    )

    results = pd.DataFrame(
        result_rows
    )

    if len(results) != len(sample):
        raise RuntimeError(
            "Ranking results do not cover the complete sample: "
            f"{len(results):,} / {len(sample):,}"
        )

    results[
        "rr25"
    ] = results[
        "truth_rank"
    ].map(
        rr25
    )

    results.to_parquet(
        ranking_path,
        index=False,
    )

    candidate_rankings = pd.DataFrame(
        candidate_ranking_rows
    )

    candidate_rankings.to_parquet(
        candidate_ranking_path,
        index=False,
    )

    def report(
        label: str,
        frame: pd.DataFrame,
    ) -> None:
        ranks = frame[
            "truth_rank"
        ]

        print()
        print(label)
        print("-" * len(label))
        print(
            "Structures:",
            f"{len(frame):,}",
        )
        print(
            "MRR@25:",
            f"{frame['rr25'].mean():.4f}",
        )
        print(
            "Top-1:",
            f"{(ranks == 1).mean():.2%}",
        )
        print(
            "Top-5:",
            f"{ranks.between(1, 5).mean():.2%}",
        )
        print(
            "Top-10:",
            f"{ranks.between(1, 10).mean():.2%}",
        )
        print(
            "Top-25:",
            f"{ranks.between(1, 25).mean():.2%}",
        )
        print(
            "Top-100:",
            f"{ranks.between(1, 100).mean():.2%}",
        )
        print(
            "Truth returned:",
            f"{ranks.notna().mean():.2%}",
        )
        print(
            "Median candidates:",
            f"{frame['candidate_count'].median():,.0f}",
        )

    print()
    print(
        "True-formula neighbor-prior benchmark"
    )
    print(
        "====================================="
    )

    report(
        "Overall",
        results,
    )

    for name, _, _ in BURDEN_BUCKETS:
        frame = results.loc[
            results[
                "burden_bucket"
            ]
            == name
        ]

        if not frame.empty:
            report(
                name,
                frame,
            )

    print()
    print(
        "By dominant library"
    )
    print(
        "-------------------"
    )

    library_rows = []

    for library, frame in results.groupby(
        "dominant_library",
        dropna=False,
    ):
        if len(frame) < 25:
            continue

        ranks = frame[
            "truth_rank"
        ]

        library_rows.append(
            {
                "dominant_library":
                    library,
                "structures":
                    len(frame),
                "mrr25":
                    frame[
                        "rr25"
                    ].mean(),
                "top1":
                    (
                        ranks
                        == 1
                    ).mean(),
                "top25":
                    ranks.between(
                        1,
                        25,
                    ).mean(),
                "top100":
                    ranks.between(
                        1,
                        100,
                    ).mean(),
                "median_candidates":
                    frame[
                        "candidate_count"
                    ].median(),
            }
        )

    by_library = (
        pd.DataFrame(
            library_rows
        )
        .sort_values(
            "structures",
            ascending=False,
        )
    )

    print(
        by_library.to_string(
            index=False,
            formatters={
                "mrr25":
                    lambda value:
                    f"{value:.4f}",
                "top1":
                    lambda value:
                    f"{value:.2%}",
                "top25":
                    lambda value:
                    f"{value:.2%}",
                "top100":
                    lambda value:
                    f"{value:.2%}",
                "median_candidates":
                    lambda value:
                    f"{value:,.0f}",
            },
        )
    )

    by_library.to_csv(
        output_dir
        / "by_library.csv",
        index=False,
    )

    print()
    print(
        "Wrote:"
    )
    print(
        sample_path
    )
    print(
        ranking_path
    )
    print(
        candidate_ranking_path
    )
    print(
        output_dir
        / "by_library.csv"
    )


if __name__ == "__main__":
    main()
