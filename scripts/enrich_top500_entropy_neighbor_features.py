from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from ms_entropy import calculate_entropy_similarity
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem


ROOT = Path(
    "data/processed/diagnostics/large_validation_audit/"
    "true_formula_neighbor_benchmark"
)

CANDIDATES_PATH = (
    ROOT
    / "top3_neighbor_collapse/top3_neighbor_top500.parquet"
)

SPECTRAL_NEIGHBORS_PATH = (
    ROOT
    / "learned_ranker_top500/spectral_neighbors.parquet"
)

TRAIN_PATH = Path(
    "data/raw/train.parquet"
)

TRAINING_STRUCTURES_PATH = Path(
    "data/processed/structure_index/training_structures.parquet"
)

PUBCHEM_ROOT = Path(
    "data/processed/pubchem_formula_index"
)

OUTPUT_PATH = (
    ROOT
    / "learned_ranker_top500/entropy_neighbor_enrichment.parquet"
)

MS2_TOLERANCE_DA = 0.02
MAX_SPECTRA_PER_STRUCTURE = 8

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


def peaks_array(
    mzs,
    intensities,
) -> np.ndarray | None:
    mz = np.asarray(
        mzs,
        dtype=np.float32,
    )

    intensity = np.asarray(
        intensities,
        dtype=np.float32,
    )

    if (
        mz.ndim != 1
        or intensity.ndim != 1
        or len(mz) == 0
        or len(mz) != len(intensity)
    ):
        return None

    mask = (
        np.isfinite(mz)
        & np.isfinite(intensity)
        & (mz > 0)
        & (intensity > 0)
    )

    if not np.any(mask):
        return None

    return np.column_stack(
        (
            mz[mask],
            intensity[mask],
        )
    ).astype(
        np.float32,
        copy=False,
    )


def load_relevant_spectra(
    keys: set[str],
) -> dict[str, list[tuple[str, np.ndarray]]]:
    spectra_by_key: dict[
        str,
        list[tuple[str, np.ndarray]],
    ] = defaultdict(list)

    parquet = pq.ParquetFile(
        TRAIN_PATH
    )

    scanned = 0
    retained = 0

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
        scanned += len(frame)

        matched = frame.loc[
            frame[
                "inchikey14"
            ]
            .astype(str)
            .isin(keys)
        ]

        for row in matched.itertuples(
            index=False
        ):
            key = str(
                row.inchikey14
            )

            if (
                len(
                    spectra_by_key[
                        key
                    ]
                )
                >= MAX_SPECTRA_PER_STRUCTURE
            ):
                continue

            peaks = peaks_array(
                row.ms2_mzs,
                row.ms2_normalized_intensities,
            )

            if peaks is None:
                continue

            spectra_by_key[
                key
            ].append(
                (
                    str(
                        row.ionization_mode
                    ),
                    peaks,
                )
            )

            retained += 1

        print(
            f"\rRows scanned: {scanned:,}; spectra retained: {retained:,}",
            end="",
            flush=True,
        )

    print()

    return dict(
        spectra_by_key
    )


def structure_entropy_similarity(
    query_spectra: list[tuple[str, np.ndarray]],
    neighbor_spectra: list[tuple[str, np.ndarray]],
) -> tuple[float, float]:
    if (
        not query_spectra
        or not neighbor_spectra
    ):
        return np.nan, np.nan

    all_scores: list[float] = []
    mode_scores: list[float] = []

    for query_mode, query_peaks in query_spectra:
        for neighbor_mode, neighbor_peaks in neighbor_spectra:
            try:
                score = float(
                    calculate_entropy_similarity(
                        query_peaks,
                        neighbor_peaks,
                        ms2_tolerance_in_da=MS2_TOLERANCE_DA,
                    )
                )
            except (
                ValueError,
                RuntimeError,
                TypeError,
            ):
                continue

            if not np.isfinite(
                score
            ):
                continue

            all_scores.append(
                score
            )

            if query_mode == neighbor_mode:
                mode_scores.append(
                    score
                )

    best_any = (
        max(all_scores)
        if all_scores
        else np.nan
    )

    best_mode = (
        max(mode_scores)
        if mode_scores
        else np.nan
    )

    return (
        float(best_any),
        float(best_mode),
    )


def load_candidate_fingerprints(
    candidates: pd.DataFrame,
) -> dict[str, object]:
    formulas = set(
        candidates[
            "candidate_formula"
        ].astype(str)
    )

    keys_by_formula = {
        str(formula):
            set(
                group[
                    "candidate_inchikey14"
                ].astype(str)
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
        .isin(formulas)
    ]

    smiles_by_key: dict[str, str] = {}

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
            ].astype(str)
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

            if (
                key
                in keys_by_formula.get(
                    formula,
                    set(),
                )
                and key
                not in smiles_by_key
            ):
                smiles_by_key[
                    key
                ] = str(
                    row.smiles
                )

    fingerprints = {}

    for key, smiles in smiles_by_key.items():
        fp = morgan_fp(
            smiles
        )

        if fp is not None:
            fingerprints[
                key
            ] = fp

    return fingerprints


def main() -> None:
    required = [
        CANDIDATES_PATH,
        SPECTRAL_NEIGHBORS_PATH,
        TRAIN_PATH,
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
            + ". Re-run scripts/enrich_top500_neighbor_features.py "
            "to create spectral_neighbors.parquet."
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

    neighbors = pd.read_parquet(
        SPECTRAL_NEIGHBORS_PATH
    ).copy()

    for column in [
        "inchikey14",
        "neighbor_inchikey14",
    ]:
        neighbors[
            column
        ] = (
            neighbors[
                column
            ]
            .astype(str)
        )

    query_keys = set(
        neighbors[
            "inchikey14"
        ]
    )

    neighbor_keys = set(
        neighbors[
            "neighbor_inchikey14"
        ]
    )

    relevant_keys = (
        query_keys
        | neighbor_keys
    )

    print(
        "Queries:",
        f"{len(query_keys):,}",
    )
    print(
        "Leakage-safe neighbor structures:",
        f"{len(neighbor_keys):,}",
    )

    spectra_by_key = load_relevant_spectra(
        relevant_keys
    )

    entropy_rows = []

    for query_number, (
        query_key,
        group,
    ) in enumerate(
        neighbors.groupby(
            "inchikey14",
            sort=False,
        ),
        start=1,
    ):
        query_spectra = spectra_by_key.get(
            str(query_key),
            [],
        )

        for row in group.itertuples(
            index=False
        ):
            best_any, best_mode = (
                structure_entropy_similarity(
                    query_spectra,
                    spectra_by_key.get(
                        str(
                            row.neighbor_inchikey14
                        ),
                        [],
                    ),
                )
            )

            entropy_rows.append(
                {
                    "inchikey14":
                        str(query_key),
                    "neighbor_inchikey14":
                        str(row.neighbor_inchikey14),
                    "entropy_similarity_max":
                        best_any,
                    "entropy_similarity_mode_max":
                        best_mode,
                }
            )

        if (
            query_number % 100 == 0
            or query_number
            == len(query_keys)
        ):
            print(
                "Entropy-rescored queries:",
                f"{query_number:,}/{len(query_keys):,}",
            )

    entropy_neighbors = pd.DataFrame(
        entropy_rows
    )

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

    neighbor_fps = {}

    for row in training_structures.loc[
        training_structures[
            "inchikey14"
        ]
        .isin(neighbor_keys)
    ].itertuples(
        index=False
    ):
        fp = morgan_fp(
            row.normalized_smiles
        )

        if fp is not None:
            neighbor_fps[
                str(
                    row.inchikey14
                )
            ] = fp

    candidate_fps = (
        load_candidate_fingerprints(
            candidates
        )
    )

    entropy_by_query = {
        str(query_key):
            group
        for query_key, group
        in entropy_neighbors.groupby(
            "inchikey14",
            sort=False,
        )
    }

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
        entropy_group = (
            entropy_by_query.get(
                str(query_key)
            )
        )

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

        entropy_weighted_mean = np.full(
            n,
            np.nan,
            dtype=np.float32,
        )
        entropy_weighted_top3_mean = np.full(
            n,
            np.nan,
            dtype=np.float32,
        )
        entropy_x_tanimoto_max = np.full(
            n,
            np.nan,
            dtype=np.float32,
        )
        best_structural_neighbor_entropy = np.full(
            n,
            np.nan,
            dtype=np.float32,
        )
        best_structural_neighbor_mode_entropy = np.full(
            n,
            np.nan,
            dtype=np.float32,
        )

        if entropy_group is not None:
            usable = []

            for row in entropy_group.itertuples(
                index=False
            ):
                neighbor_fp = neighbor_fps.get(
                    str(
                        row.neighbor_inchikey14
                    )
                )

                if neighbor_fp is None:
                    continue

                usable.append(
                    (
                        neighbor_fp,
                        float(
                            row.entropy_similarity_max
                        ),
                        float(
                            row.entropy_similarity_mode_max
                        ),
                    )
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

                entropy = np.asarray(
                    [
                        max(
                            score,
                            0.0,
                        )
                        if np.isfinite(score)
                        else 0.0
                        for _,
                        score,
                        _
                        in usable
                    ],
                    dtype=np.float32,
                )

                mode_entropy = np.asarray(
                    [
                        max(
                            score,
                            0.0,
                        )
                        if np.isfinite(score)
                        else 0.0
                        for _,
                        _,
                        score
                        in usable
                    ],
                    dtype=np.float32,
                )

                for neighbor_index, (
                    neighbor_fp,
                    _,
                    _,
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

                weighted = (
                    similarities
                    * entropy[
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

                max_structural_neighbor = np.argmax(
                    similarities,
                    axis=0,
                )

                entropy_sum = float(
                    entropy.sum()
                )

                for local_column, position in enumerate(
                    valid_positions
                ):
                    if entropy_sum > 0:
                        entropy_weighted_mean[
                            position
                        ] = float(
                            weighted[
                                :,
                                local_column
                            ].sum()
                            / entropy_sum
                        )

                    entropy_weighted_top3_mean[
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

                    entropy_x_tanimoto_max[
                        position
                    ] = float(
                        weighted[
                            :,
                            local_column
                        ].max()
                    )

                    best_index = int(
                        max_structural_neighbor[
                            local_column
                        ]
                    )

                    best_structural_neighbor_entropy[
                        position
                    ] = entropy[
                        best_index
                    ]

                    best_structural_neighbor_mode_entropy[
                        position
                    ] = mode_entropy[
                        best_index
                    ]

        enriched[
            "entropy_neighbor_weighted_mean"
        ] = entropy_weighted_mean
        enriched[
            "entropy_neighbor_weighted_top3_mean"
        ] = entropy_weighted_top3_mean
        enriched[
            "entropy_x_tanimoto_max"
        ] = entropy_x_tanimoto_max
        enriched[
            "best_structural_neighbor_entropy_score"
        ] = best_structural_neighbor_entropy
        enriched[
            "best_structural_neighbor_mode_entropy_score"
        ] = best_structural_neighbor_mode_entropy

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
                "Candidate enrichment queries:",
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
            "Entropy enrichment row count does not match candidates"
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
        "Entropy-neighbor enrichment"
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
                "entropy_neighbor_weighted_mean",
                "entropy_neighbor_weighted_top3_mean",
                "entropy_x_tanimoto_max",
                "best_structural_neighbor_entropy_score",
                "best_structural_neighbor_mode_entropy_score",
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
