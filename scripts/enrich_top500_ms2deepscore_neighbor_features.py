from __future__ import annotations

from collections import defaultdict
import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from matchms import Spectrum
from ms2deepscore import MS2DeepScore
from ms2deepscore.models import load_model
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

MODEL_PATH = Path(
    "data/external/ms2deepscore/ms2deepscore_model.pt"
)

OUTPUT_PATH = (
    ROOT
    / "learned_ranker_top500/ms2deepscore_neighbor_enrichment.parquet"
)

MAX_SPECTRA_PER_STRUCTURE = 8
EMBED_BATCH_SIZE = 1024

RDLogger.DisableLog("rdApp.*")


def normalize_ionmode(value: object) -> str:
    text = str(value).strip().lower()

    if text.startswith("pos"):
        return "positive"

    if text.startswith("neg"):
        return "negative"

    return text


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


def to_matchms_spectrum(
    mzs,
    intensities,
    precursor_mz: object,
    ionization_mode: object,
) -> Spectrum | None:
    mz = np.asarray(
        mzs,
        dtype=np.float64,
    )

    intensity = np.asarray(
        intensities,
        dtype=np.float64,
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

    mz = mz[mask]
    intensity = intensity[mask]

    order = np.argsort(
        mz,
        kind="stable",
    )

    mz = mz[
        order
    ]

    intensity = intensity[
        order
    ]

    maximum = float(
        intensity.max()
    )

    if maximum <= 0:
        return None

    intensity = (
        intensity
        / maximum
    )

    try:
        precursor = float(
            precursor_mz
        )
    except (
        TypeError,
        ValueError,
    ):
        precursor = np.nan

    metadata = {
        "ionmode":
            normalize_ionmode(
                ionization_mode
            ),
    }

    if np.isfinite(
        precursor
    ):
        metadata[
            "precursor_mz"
        ] = precursor

    return Spectrum(
        mz=mz,
        intensities=intensity,
        metadata=metadata,
    )


def load_relevant_spectra(
    keys: set[str],
) -> tuple[
    list[Spectrum],
    list[str],
]:
    spectra: list[Spectrum] = []
    structure_keys: list[str] = []

    counts: dict[str, int] = defaultdict(int)

    parquet = pq.ParquetFile(
        TRAIN_PATH
    )

    scanned = 0

    for batch in parquet.iter_batches(
        batch_size=50_000,
        columns=[
            "inchikey14",
            "ionization_mode",
            "precursor_mz",
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
                keys
            )
        ]

        for row in matched.itertuples(
            index=False
        ):
            key = str(
                row.inchikey14
            )

            if (
                counts[
                    key
                ]
                >= MAX_SPECTRA_PER_STRUCTURE
            ):
                continue

            spectrum = (
                to_matchms_spectrum(
                    row.ms2_mzs,
                    row.ms2_normalized_intensities,
                    row.precursor_mz,
                    row.ionization_mode,
                )
            )

            if spectrum is None:
                continue

            spectra.append(
                spectrum
            )

            structure_keys.append(
                key
            )

            counts[
                key
            ] += 1

        print(
            "\rRows scanned: "
            f"{scanned:,}; spectra retained: {len(spectra):,}",
            end="",
            flush=True,
        )

    print()

    return (
        spectra,
        structure_keys,
    )


def normalize_rows(
    values: np.ndarray,
) -> np.ndarray:
    norms = np.linalg.norm(
        values,
        axis=1,
        keepdims=True,
    )

    norms[
        norms == 0
    ] = 1.0

    return (
        values
        / norms
    )


def structure_embedding_map(
    embeddings: np.ndarray,
    structure_keys: list[str],
) -> dict[str, np.ndarray]:
    grouped: dict[
        str,
        list[np.ndarray],
    ] = defaultdict(list)

    for key, embedding in zip(
        structure_keys,
        embeddings,
        strict=True,
    ):
        grouped[
            str(key)
        ].append(
            embedding
        )

    return {
        key:
            normalize_rows(
                np.asarray(
                    values,
                    dtype=np.float32,
                )
            )
        for key, values
        in grouped.items()
    }


def structure_similarity(
    query_embeddings: np.ndarray | None,
    neighbor_embeddings: np.ndarray | None,
) -> tuple[
    float,
    float,
]:
    if (
        query_embeddings is None
        or neighbor_embeddings is None
        or not len(
            query_embeddings
        )
        or not len(
            neighbor_embeddings
        )
    ):
        return (
            np.nan,
            np.nan,
        )

    similarities = (
        query_embeddings
        @ neighbor_embeddings.T
    )

    return (
        float(
            similarities.max()
        ),
        float(
            np.mean(
                np.max(
                    similarities,
                    axis=1,
                )
            )
        ),
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
        MODEL_PATH,
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

    spectra, structure_keys = (
        load_relevant_spectra(
            relevant_keys
        )
    )

    if not spectra:
        raise RuntimeError(
            "No relevant spectra were loaded"
        )

    print(
        "Loading MS2DeepScore model:",
        MODEL_PATH,
    )

    model = load_model(
        str(
            MODEL_PATH
        ),
        allow_legacy=True,
    )

    scorer = MS2DeepScore(
        model,
        progress_bar=True,
    )

    print(
        "Computing embeddings:",
        f"{len(spectra):,} spectra",
    )

    embedding_signature = inspect.signature(
        scorer.get_embedding_array
    )

    embedding_kwargs = {}

    if (
        "batch_size"
        in embedding_signature.parameters
    ):
        embedding_kwargs[
            "batch_size"
        ] = EMBED_BATCH_SIZE

    if (
        "progress_bar"
        in embedding_signature.parameters
    ):
        embedding_kwargs[
            "progress_bar"
        ] = True

    print(
        "MS2DeepScore get_embedding_array signature:",
        embedding_signature,
    )

    embeddings = (
        scorer.get_embedding_array(
            spectra,
            **embedding_kwargs,
        )
    )

    embeddings = normalize_rows(
        np.asarray(
            embeddings,
            dtype=np.float32,
        )
    )

    embeddings_by_structure = (
        structure_embedding_map(
            embeddings,
            structure_keys,
        )
    )

    print(
        "Structures with embeddings:",
        f"{len(embeddings_by_structure):,}",
    )

    score_rows = []

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
        query_embeddings = (
            embeddings_by_structure.get(
                str(query_key)
            )
        )

        for row in group.itertuples(
            index=False
        ):
            (
                max_score,
                mean_best_score,
            ) = structure_similarity(
                query_embeddings,
                embeddings_by_structure.get(
                    str(
                        row.neighbor_inchikey14
                    )
                ),
            )

            score_rows.append(
                {
                    "inchikey14":
                        str(query_key),
                    "neighbor_inchikey14":
                        str(
                            row.neighbor_inchikey14
                        ),
                    "ms2ds_similarity_max":
                        max_score,
                    "ms2ds_similarity_mean_best":
                        mean_best_score,
                }
            )

        if (
            query_number % 100 == 0
            or query_number
            == len(query_keys)
        ):
            print(
                "MS2DeepScore neighbor queries:",
                f"{query_number:,}/{len(query_keys):,}",
            )

    ms2ds_neighbors = pd.DataFrame(
        score_rows
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
        .isin(
            neighbor_keys
        )
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

    scores_by_query = {
        str(query_key):
            group
        for query_key, group
        in ms2ds_neighbors.groupby(
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
        score_group = (
            scores_by_query.get(
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

        x_tanimoto_max = np.full(
            n,
            np.nan,
            dtype=np.float32,
        )

        best_structural_neighbor_score = (
            np.full(
                n,
                np.nan,
                dtype=np.float32,
            )
        )

        best_structural_neighbor_mean_best = (
            np.full(
                n,
                np.nan,
                dtype=np.float32,
            )
        )

        if score_group is not None:
            usable = []

            for row in score_group.itertuples(
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
                            row.ms2ds_similarity_max
                        ),
                        float(
                            row.ms2ds_similarity_mean_best
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

                ms2ds_max = np.asarray(
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

                ms2ds_mean_best = np.asarray(
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
                    * ms2ds_max[
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

                max_structural_neighbor = (
                    np.argmax(
                        similarities,
                        axis=0,
                    )
                )

                score_sum = float(
                    ms2ds_max.sum()
                )

                for (
                    local_column,
                    position,
                ) in enumerate(
                    valid_positions
                ):
                    if score_sum > 0:
                        weighted_mean[
                            position
                        ] = float(
                            weighted[
                                :,
                                local_column
                            ].sum()
                            / score_sum
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

                    x_tanimoto_max[
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

                    best_structural_neighbor_score[
                        position
                    ] = ms2ds_max[
                        best_index
                    ]

                    best_structural_neighbor_mean_best[
                        position
                    ] = (
                        ms2ds_mean_best[
                            best_index
                        ]
                    )

        enriched[
            "ms2ds_neighbor_weighted_mean"
        ] = weighted_mean

        enriched[
            "ms2ds_neighbor_weighted_top3_mean"
        ] = weighted_top3_mean

        enriched[
            "ms2ds_x_tanimoto_max"
        ] = x_tanimoto_max

        enriched[
            "best_structural_neighbor_ms2ds_score"
        ] = best_structural_neighbor_score

        enriched[
            "best_structural_neighbor_ms2ds_mean_best"
        ] = best_structural_neighbor_mean_best

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
            "MS2DeepScore enrichment row count "
            "does not match candidates"
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
            f"MS2DeepScore enrichment contains "
            f"{duplicates:,} duplicate rows"
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
        "MS2DeepScore neighbor enrichment"
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
                "ms2ds_neighbor_weighted_mean",
                "ms2ds_neighbor_weighted_top3_mean",
                "ms2ds_x_tanimoto_max",
                "best_structural_neighbor_ms2ds_score",
                "best_structural_neighbor_ms2ds_mean_best",
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
