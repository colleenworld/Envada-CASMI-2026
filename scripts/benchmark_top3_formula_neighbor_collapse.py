from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from scipy import sparse

from casmi26.pubchem.neighbor_pruning import score_candidates
from casmi26.retrieval.index import load_retrieval_index
from casmi26.retrieval.indexed import search_structures_indexed_blockwise
from casmi26.spectra.binning import bin_spectrum


SPECTRAL_NEIGHBORS = 25
BATCH_STRUCTURES = 16
SAVE_TOP_K_PER_FORMULA = 100
MAX_MZ = 1500.0
BIN_WIDTH = 1.0

FORMULA_PENALTIES = [1, 3, 5, 10, 25, 50]

RDLogger.DisableLog("rdApp.*")


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


def summarize(
    label: str,
    ranks: pd.Series,
) -> dict:
    return {
        "policy":
            label,
        "mrr25":
            ranks.map(
                rr25
            ).mean(),
        "top1":
            ranks.eq(
                1
            ).mean(),
        "top5":
            ranks.between(
                1,
                5,
            ).mean(),
        "top10":
            ranks.between(
                1,
                10,
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
    }


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


def collapse(
    candidates: pd.DataFrame,
    policy: str,
    penalty: int | None = None,
) -> pd.DataFrame:
    frame = candidates.copy()

    if policy == "formula_first":
        sort_columns = [
            "inchikey14",
            "formula_rank",
            "prior_rank",
            "candidate_inchikey14",
        ]
        ascending = [
            True,
            True,
            True,
            True,
        ]

    elif policy == "round_robin":
        sort_columns = [
            "inchikey14",
            "prior_rank",
            "formula_rank",
            "candidate_inchikey14",
        ]
        ascending = [
            True,
            True,
            True,
            True,
        ]

    elif policy == "rank_penalty":
        if penalty is None:
            raise ValueError(
                "rank_penalty requires penalty"
            )

        frame[
            "_collapse_score"
        ] = (
            frame[
                "prior_rank"
            ].astype(float)
            + float(
                penalty
            )
            * (
                frame[
                    "formula_rank"
                ].astype(float)
                - 1.0
            )
        )

        sort_columns = [
            "inchikey14",
            "_collapse_score",
            "formula_rank",
            "prior_rank",
            "candidate_inchikey14",
        ]
        ascending = [
            True,
            True,
            True,
            True,
            True,
        ]

    else:
        raise ValueError(
            f"Unknown collapse policy: {policy}"
        )

    ranked = (
        frame
        .sort_values(
            sort_columns,
            ascending=ascending,
            kind="stable",
        )
        .drop_duplicates(
            subset=[
                "inchikey14",
                "candidate_inchikey14",
            ],
            keep="first",
        )
        .reset_index(
            drop=True
        )
    )

    ranked[
        "collapse_rank"
    ] = (
        ranked.groupby(
            "inchikey14"
        )
        .cumcount()
        + 1
    )

    return ranked


def main() -> None:
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

    benchmark_dir = (
        audit_dir
        / "true_formula_neighbor_benchmark"
    )

    sample_path = (
        benchmark_dir
        / "mist_full/sample.parquet"
    )

    hypotheses_path = (
        audit_dir
        / "formula_hypotheses.parquet"
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
        benchmark_dir
        / "top3_neighbor_collapse"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    candidate_path = (
        output_dir
        / "top3_neighbor_top100.parquet"
    )

    summary_path = (
        output_dir
        / "collapse_summary.csv"
    )

    truth_path = (
        output_dir
        / "truth_ranks.parquet"
    )

    required = [
        sample_path,
        hypotheses_path,
        train_path,
        strict_index_dir
        / "metadata.parquet",
        strict_index_dir
        / "vectors.npz",
        training_structures_path,
        pubchem_root
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
                str(
                    path
                )
                for path
                in missing
            )
        )

    sample = (
        pd.read_parquet(
            sample_path
        )
        .copy()
    )

    sample[
        "inchikey14"
    ] = sample[
        "inchikey14"
    ].astype(str)

    sample_keys = set(
        sample[
            "inchikey14"
        ]
    )

    sample = (
        sample
        .drop_duplicates(
            subset=[
                "inchikey14"
            ]
        )
        .reset_index(
            drop=True
        )
    )

    sample[
        "benchmark_id"
    ] = np.arange(
        len(sample),
        dtype=np.int64,
    )

    print(
        "Benchmark structures:",
        f"{len(sample):,}",
    )

    hypotheses = pd.read_parquet(
        hypotheses_path
    )

    required_hypothesis_columns = {
        "inchikey14",
        "candidate_formula",
        "is_truth_formula",
        "formula_support_count",
        "mass_error_best_abs_ppm",
    }

    missing_columns = (
        required_hypothesis_columns
        - set(
            hypotheses.columns
        )
    )

    if missing_columns:
        raise ValueError(
            "Formula hypotheses are missing columns: "
            f"{sorted(missing_columns)}"
        )

    hypotheses[
        "inchikey14"
    ] = hypotheses[
        "inchikey14"
    ].astype(str)

    hypotheses = hypotheses.loc[
        hypotheses[
            "inchikey14"
        ]
        .isin(
            sample_keys
        )
    ].copy()

    hypotheses = (
        hypotheses
        .sort_values(
            [
                "inchikey14",
                "mass_error_best_abs_ppm",
                "formula_support_count",
                "candidate_formula",
            ],
            ascending=[
                True,
                True,
                False,
                True,
            ],
            kind="stable",
        )
        .copy()
    )

    hypotheses[
        "formula_rank"
    ] = (
        hypotheses.groupby(
            "inchikey14"
        )
        .cumcount()
        + 1
    )

    top3 = hypotheses.loc[
        hypotheses[
            "formula_rank"
        ]
        <= 3
    ].copy()

    truth_formula_rank = (
        hypotheses.loc[
            hypotheses[
                "is_truth_formula"
            ].astype(bool),
            [
                "inchikey14",
                "formula_rank",
            ],
        ]
        .drop_duplicates(
            subset=[
                "inchikey14"
            ]
        )
    )

    formula_metrics = (
        sample[
            [
                "inchikey14",
            ]
        ]
        .merge(
            truth_formula_rank,
            on="inchikey14",
            how="left",
            validate="one_to_one",
        )
    )

    print(
        "Truth formula Top-1:",
        f"{formula_metrics['formula_rank'].eq(1).mean():.2%}",
    )
    print(
        "Truth formula Top-3:",
        f"{formula_metrics['formula_rank'].between(1,3).mean():.2%}",
    )

    #
    # Collect spectra for the benchmark structures.
    #
    parquet = pq.ParquetFile(
        train_path
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

    #
    # Retrieve leakage-safe training neighbors once per structure.
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

    candidate_keys, reference_candidate_ids = np.unique(
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

    neighbors_by_id = {}

    benchmark_ids = (
        sample[
            "benchmark_id"
        ]
        .astype(int)
        .to_numpy()
    )

    for start in range(
        0,
        len(
            benchmark_ids
        ),
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

        original_ids = query_structure_ids[
            query_mask
        ]

        local_id_by_original = {
            int(
                original
            ):
                local
            for local, original
            in enumerate(
                batch_ids
            )
        }

        local_structure_ids = np.asarray(
            [
                local_id_by_original[
                    int(
                        original
                    )
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
                query_structure_ids=(
                    local_structure_ids
                ),
                query_modes=query_modes[
                    query_mask
                ],
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
                    len(
                        batch_ids
                    )
                ),
                candidate_count=(
                    len(
                        candidate_keys
                    )
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
            start
            + len(
                batch_ids
            ),
            len(
                benchmark_ids
            ),
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
    # Score every top-3 formula candidate set.
    #
    formula_buckets = pd.read_parquet(
        pubchem_root
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

    selected_formulas = set(
        top3[
            "candidate_formula"
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
            )
        ]
    )

    queries_by_formula = {
        formula: group.copy()
        for formula, group
        in top3.groupby(
            "candidate_formula",
            sort=False,
        )
    }

    candidate_rows = []

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

        frame = pd.read_parquet(
            pubchem_root
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

        frame = (
            frame.loc[
                frame[
                    "molecular_formula"
                ]
                .isin(
                    formulas_in_bucket
                )
            ]
            .drop_duplicates(
                subset=[
                    "molecular_formula",
                    "inchikey14",
                ]
            )
        )

        for formula in sorted(
            formulas_in_bucket
        ):
            formula_queries = (
                queries_by_formula.get(
                    formula
                )
            )

            if (
                formula_queries is None
                or formula_queries.empty
            ):
                continue

            formula_candidates = (
                frame.loc[
                    frame[
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
                continue

            valid_keys_array = np.asarray(
                valid_keys,
                dtype=object,
            )

            for query in formula_queries.itertuples(
                index=False
            ):
                benchmark_id = (
                    benchmark_id_by_key[
                        str(
                            query.inchikey14
                        )
                    ]
                )

                scores, usable_neighbors = (
                    score_candidates(
                        valid_fps,
                        neighbors_by_id[
                            benchmark_id
                        ],
                        training_fps,
                    )
                )

                order = np.lexsort(
                    (
                        valid_keys_array,
                        -scores,
                    )
                )

                top_count = min(
                    SAVE_TOP_K_PER_FORMULA,
                    len(
                        order
                    ),
                )

                for prior_rank, candidate_index in enumerate(
                    order[
                        :top_count
                    ],
                    start=1,
                ):
                    candidate_rows.append(
                        {
                            "inchikey14":
                                str(
                                    query.inchikey14
                                ),
                            "candidate_formula":
                                formula,
                            "formula_rank":
                                int(
                                    query.formula_rank
                                ),
                            "mass_error_best_abs_ppm":
                                float(
                                    query.mass_error_best_abs_ppm
                                ),
                            "candidate_inchikey14":
                                str(
                                    valid_keys_array[
                                        candidate_index
                                    ]
                                ),
                            "prior_rank":
                                int(
                                    prior_rank
                                ),
                            "neighbor_prior_score":
                                float(
                                    scores[
                                        candidate_index
                                    ]
                                ),
                            "usable_neighbor_fingerprints":
                                int(
                                    usable_neighbors
                                ),
                        }
                    )

        print(
            "Processed PubChem bucket:",
            f"{bucket:03d}",
        )

    if not candidate_rows:
        raise RuntimeError(
            "No top-3 neighbor candidate rankings were generated"
        )

    candidates = pd.DataFrame(
        candidate_rows
    )

    candidates.to_parquet(
        candidate_path,
        index=False,
        compression="zstd",
    )

    #
    # Correct-formula neighbor oracle: how well structure ranking works
    # if formula selection is perfect.
    #
    truth_formula_candidates = candidates.merge(
        top3.loc[
            top3[
                "is_truth_formula"
            ].astype(bool),
            [
                "inchikey14",
                "candidate_formula",
            ],
        ].drop_duplicates(),
        on=[
            "inchikey14",
            "candidate_formula",
        ],
        how="inner",
    )

    oracle_truth = (
        truth_formula_candidates.loc[
            truth_formula_candidates[
                "candidate_inchikey14"
            ].astype(str)
            == truth_formula_candidates[
                "inchikey14"
            ].astype(str),
            [
                "inchikey14",
                "prior_rank",
            ],
        ]
        .drop_duplicates(
            subset=[
                "inchikey14"
            ]
        )
        .rename(
            columns={
                "prior_rank":
                    "truth_formula_neighbor_rank",
            }
        )
    )

    oracle_metrics = (
        sample[
            [
                "inchikey14",
            ]
        ]
        .merge(
            oracle_truth,
            on="inchikey14",
            how="left",
            validate="one_to_one",
        )
    )

    summary_rows = [
        summarize(
            "truth_formula_neighbor_oracle",
            oracle_metrics[
                "truth_formula_neighbor_rank"
            ],
        )
    ]

    truth_rank_frames = []

    policies = [
        (
            "formula_first",
            "formula_first",
            None,
        ),
        (
            "round_robin",
            "round_robin",
            None,
        ),
    ]

    for penalty in FORMULA_PENALTIES:
        policies.append(
            (
                f"rank_penalty_{penalty}",
                "rank_penalty",
                penalty,
            )
        )

    for label, policy, penalty in policies:
        ranked = collapse(
            candidates,
            policy,
            penalty,
        )

        truth = (
            ranked.loc[
                ranked[
                    "candidate_inchikey14"
                ].astype(str)
                == ranked[
                    "inchikey14"
                ].astype(str),
                [
                    "inchikey14",
                    "collapse_rank",
                ],
            ]
            .drop_duplicates(
                subset=[
                    "inchikey14"
                ]
            )
        )

        metrics = (
            sample[
                [
                    "inchikey14",
                ]
            ]
            .merge(
                truth,
                on="inchikey14",
                how="left",
                validate="one_to_one",
            )
        )

        summary_rows.append(
            summarize(
                label,
                metrics[
                    "collapse_rank"
                ],
            )
        )

        metrics[
            "policy"
        ] = label

        truth_rank_frames.append(
            metrics
        )

    summary = pd.DataFrame(
        summary_rows
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    pd.concat(
        truth_rank_frames,
        ignore_index=True,
    ).to_parquet(
        truth_path,
        index=False,
    )

    print()
    print(
        "Top-3 formula + neighbor collapse benchmark"
    )
    print(
        "==========================================="
    )
    print(
        "Structures:",
        f"{len(sample):,}",
    )
    print(
        "Per-formula candidates retained:",
        SAVE_TOP_K_PER_FORMULA,
    )
    print()
    print(
        summary.to_string(
            index=False,
            formatters={
                "mrr25":
                    lambda value:
                    f"{value:.4f}",
                "top1":
                    lambda value:
                    f"{value:.2%}",
                "top5":
                    lambda value:
                    f"{value:.2%}",
                "top10":
                    lambda value:
                    f"{value:.2%}",
                "top25":
                    lambda value:
                    f"{value:.2%}",
                "top100":
                    lambda value:
                    f"{value:.2%}",
            },
        )
    )
    print()
    print(
        "Wrote:",
        summary_path,
    )
    print(
        "Wrote:",
        candidate_path,
    )
    print(
        "Wrote:",
        truth_path,
    )


if __name__ == "__main__":
    main()
