from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy import sparse

from casmi26.chemistry.adducts import neutral_mass
from casmi26.retrieval.index import load_retrieval_index


INDEX_DIR = Path(
    "data/processed/retrieval_index"
)

MANIFEST_PATH = Path(
    "data/processed/splits/retrieval_dev.parquet"
)

CROSS_POLARITY_PATH = Path(
    "data/processed/results/"
    "retrieval_dev_cross_polarity_mass_coverage.parquet"
)

RESULTS_PATH = Path(
    "data/processed/results/"
    "retrieval_dev_cross_polarity_cosine.parquet"
)

TOLERANCE_DA = 0.01


def infer_neutral_masses(
    precursor_mz: np.ndarray,
    adducts: np.ndarray,
) -> np.ndarray:
    masses = np.full(
        len(precursor_mz),
        np.nan,
        dtype=np.float64,
    )

    for index, (mz, adduct) in enumerate(
        zip(
            precursor_mz,
            adducts,
            strict=True,
        )
    ):
        try:
            mz_value = float(mz)
        except (
            TypeError,
            ValueError,
        ):
            continue

        if not np.isfinite(mz_value):
            continue

        value = neutral_mass(
            mz_value,
            str(adduct),
        )

        if (
            value is not None
            and np.isfinite(value)
        ):
            masses[index] = value

    return masses


def build_reference_mask(
    metadata: pd.DataFrame,
    manifest: pd.DataFrame,
) -> np.ndarray:
    """
    Same frozen reference-pool rule used by the previous
    retrieval-dev experiments.
    """
    query_pairs = set(
        zip(
            manifest[
                "inchikey14"
            ].astype(str),
            manifest[
                "query_library"
            ].astype(str),
            strict=True,
        )
    )

    return np.fromiter(
        (
            (
                str(key),
                str(library),
            )
            not in query_pairs
            for key, library in zip(
                metadata["inchikey14"],
                metadata["ingest_lib"],
                strict=True,
            )
        ),
        dtype=bool,
        count=len(metadata),
    )


def aggregate_candidate_scores(
    similarities: np.ndarray,
    candidate_ids: np.ndarray,
    candidate_count: int,
) -> np.ndarray:
    """
    For each candidate structure, use the maximum cosine
    similarity across all query/reference spectrum pairs.
    """
    scores = np.full(
        candidate_count,
        -np.inf,
        dtype=np.float32,
    )

    if similarities.size == 0:
        return scores

    # similarities shape:
    #
    #     query spectra x reference spectra
    #
    # First take the best query-spectrum match for each
    # reference spectrum.
    reference_scores = np.max(
        similarities,
        axis=0,
    )

    np.maximum.at(
        scores,
        candidate_ids,
        reference_scores,
    )

    return scores


def rank_truth(
    scores: np.ndarray,
    truth_candidate_id: int,
) -> int | None:
    truth_score = scores[
        truth_candidate_id
    ]

    if not np.isfinite(
        truth_score
    ):
        return None

    return (
        1
        + int(
            np.count_nonzero(
                scores
                > truth_score
            )
        )
    )


def reciprocal_rank(
    rank: int | None,
    k: int = 25,
) -> float:
    if (
        rank is None
        or rank > k
    ):
        return 0.0

    return 1.0 / rank


def main() -> None:
    started = time.monotonic()

    print(
        "Loading retrieval index..."
    )

    metadata, vectors = (
        load_retrieval_index(
            INDEX_DIR
        )
    )

    manifest = pd.read_parquet(
        MANIFEST_PATH
    )

    cross = pd.read_parquet(
        CROSS_POLARITY_PATH
    )

    print(
        f"Index spectra: "
        f"{len(metadata):,}"
    )

    # ---------------------------------------------------------
    # Identify exactly the 17 structures that had no
    # same-polarity Baseline 1 truth but were recovered by
    # cross-polarity neutral mass.
    # ---------------------------------------------------------

    targets = (
        cross[
            cross[
                "baseline1_rank"
            ].isna()
            & cross[
                "cross_polarity_truth_survived"
            ]
        ]
        .copy()
    )

    print(
        f"Cross-polarity-only targets: "
        f"{len(targets):,}"
    )

    if len(targets) != 17:
        raise RuntimeError(
            "Expected 17 cross-polarity-only "
            f"targets; got {len(targets)}"
        )

    # ---------------------------------------------------------
    # Reconstruct frozen reference pool.
    # ---------------------------------------------------------

    print()
    print(
        "Building reference pool..."
    )

    reference_mask = (
        build_reference_mask(
            metadata=metadata,
            manifest=manifest,
        )
    )

    reference_rows = (
        np.flatnonzero(
            reference_mask
        )
    )

    removed = (
        len(metadata)
        - len(reference_rows)
    )

    print(
        f"Reference spectra: "
        f"{len(reference_rows):,}"
    )

    print(
        f"Query spectra removed: "
        f"{removed:,}"
    )

    if removed != 19_812:
        raise RuntimeError(
            "Expected 19,812 query spectra "
            f"removed; got {removed:,}"
        )

    # ---------------------------------------------------------
    # Arrays for the complete index.
    # ---------------------------------------------------------

    metadata_keys = (
        metadata[
            "inchikey14"
        ]
        .astype(str)
        .to_numpy()
    )

    metadata_libraries = (
        metadata[
            "ingest_lib"
        ]
        .astype(str)
        .to_numpy()
    )

    # ---------------------------------------------------------
    # Reference arrays.
    # ---------------------------------------------------------

    reference_metadata = (
        metadata.iloc[
            reference_rows
        ]
    )

    reference_keys = (
        reference_metadata[
            "inchikey14"
        ]
        .astype(str)
        .to_numpy()
    )

    reference_modes = (
        reference_metadata[
            "ionization_mode"
        ]
        .astype(str)
        .to_numpy()
    )

    print()
    print(
        "Inferring reference neutral masses..."
    )

    reference_masses = (
        infer_neutral_masses(
            reference_metadata[
                "precursor_mz"
            ].to_numpy(),
            reference_metadata[
                "adduct"
            ]
            .astype(str)
            .to_numpy(),
        )
    )

    results = []

    print()
    print(
        "Running cross-polarity cosine diagnostic..."
    )

    # ---------------------------------------------------------
    # Analyze each of the 17 structures independently.
    # ---------------------------------------------------------

    for position, target in enumerate(
        targets.itertuples(
            index=False
        ),
        start=1,
    ):
        structure_started = (
            time.monotonic()
        )

        truth_key = str(
            target.inchikey14
        )

        query_library = str(
            target.query_library
        )

        query_mask = (
            (
                metadata_keys
                == truth_key
            )
            & (
                metadata_libraries
                == query_library
            )
        )

        query_rows = (
            np.flatnonzero(
                query_mask
            )
        )

        if len(query_rows) == 0:
            raise RuntimeError(
                f"No query spectra for "
                f"{truth_key}"
            )

        query_metadata = (
            metadata.iloc[
                query_rows
            ]
        )

        query_masses = (
            infer_neutral_masses(
                query_metadata[
                    "precursor_mz"
                ].to_numpy(),
                query_metadata[
                    "adduct"
                ]
                .astype(str)
                .to_numpy(),
            )
        )

        known_query_masses = (
            query_masses[
                np.isfinite(
                    query_masses
                )
            ]
        )

        if (
            len(known_query_masses)
            == 0
        ):
            raise RuntimeError(
                f"No known neutral mass for "
                f"{truth_key}"
            )

        # -----------------------------------------------------
        # Cross-polarity mass candidate reference spectra.
        #
        # Deliberately ignore reference ionization mode.
        # -----------------------------------------------------

        mass_match = np.zeros(
            len(reference_metadata),
            dtype=bool,
        )

        finite_reference_mass = (
            np.isfinite(
                reference_masses
            )
        )

        for query_mass in (
            known_query_masses
        ):
            mass_match |= (
                finite_reference_mass
                & (
                    np.abs(
                        reference_masses
                        - query_mass
                    )
                    <= TOLERANCE_DA
                )
            )

        candidate_reference_local = (
            np.flatnonzero(
                mass_match
            )
        )

        candidate_reference_rows = (
            reference_rows[
                candidate_reference_local
            ]
        )

        candidate_reference_keys = (
            reference_keys[
                candidate_reference_local
            ]
        )

        (
            candidate_keys,
            candidate_ids,
        ) = np.unique(
            candidate_reference_keys,
            return_inverse=True,
        )

        truth_locations = (
            np.flatnonzero(
                candidate_keys
                == truth_key
            )
        )

        if len(
            truth_locations
        ) != 1:
            raise RuntimeError(
                f"Truth candidate missing for "
                f"{truth_key}"
            )

        truth_candidate_id = int(
            truth_locations[0]
        )

        # -----------------------------------------------------
        # Query/reference mode information.
        # -----------------------------------------------------

        query_modes = set(
            query_metadata[
                "ionization_mode"
            ]
            .astype(str)
        )

        truth_reference_mask = (
            candidate_reference_keys
            == truth_key
        )

        truth_reference_modes = set(
            reference_modes[
                candidate_reference_local[
                    truth_reference_mask
                ]
            ]
        )

        # These targets should have no same-polarity truth
        # spectrum in the Baseline 1 reference pool.
        common_truth_modes = (
            query_modes
            & truth_reference_modes
        )

        if common_truth_modes:
            raise RuntimeError(
                f"{truth_key} unexpectedly has "
                "same-polarity truth reference "
                f"modes: {common_truth_modes}"
            )

        # -----------------------------------------------------
        # Cross-polarity cosine.
        #
        # Compare query spectra directly against every
        # mass-matched reference spectrum, irrespective
        # of polarity.
        # -----------------------------------------------------

        query_vectors = (
            vectors[
                query_rows
            ]
        )

        candidate_vectors = (
            vectors[
                candidate_reference_rows
            ]
        )

        similarities = (
            query_vectors
            @ candidate_vectors.T
        )

        if sparse.issparse(
            similarities
        ):
            similarities = (
                similarities.toarray()
            )

        similarities = np.asarray(
            similarities,
            dtype=np.float32,
        )

        scores = (
            aggregate_candidate_scores(
                similarities=similarities,
                candidate_ids=(
                    candidate_ids
                ),
                candidate_count=(
                    len(candidate_keys)
                ),
            )
        )

        truth_rank = rank_truth(
            scores=scores,
            truth_candidate_id=(
                truth_candidate_id
            ),
        )

        truth_score = float(
            scores[
                truth_candidate_id
            ]
        )

        elapsed = (
            time.monotonic()
            - structure_started
        )

        results.append(
            {
                "inchikey14": (
                    truth_key
                ),
                "query_library": (
                    query_library
                ),
                "query_spectrum_count": (
                    len(query_rows)
                ),
                "candidate_reference_spectrum_count": (
                    len(
                        candidate_reference_rows
                    )
                ),
                "candidate_structure_count": (
                    len(candidate_keys)
                ),
                "truth_reference_spectrum_count": int(
                    truth_reference_mask.sum()
                ),
                "truth_score": (
                    truth_score
                ),
                "rank": (
                    truth_rank
                ),
                "reciprocal_rank": (
                    reciprocal_rank(
                        truth_rank
                    )
                ),
                "elapsed_seconds": (
                    elapsed
                ),
            }
        )

        rank_text = (
            str(truth_rank)
            if truth_rank is not None
            else "None"
        )

        print(
            f"{position:>2}/"
            f"{len(targets)}  "
            f"{truth_key}  "
            f"{query_library:<12} "
            f"candidates="
            f"{len(candidate_keys):>4}  "
            f"rank="
            f"{rank_text:>5}  "
            f"score="
            f"{truth_score:.4f}  "
            f"{elapsed:.2f}s"
        )

    # ---------------------------------------------------------
    # Results.
    # ---------------------------------------------------------

    results_frame = pd.DataFrame(
        results
    )

    RESULTS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_frame.to_parquet(
        RESULTS_PATH,
        index=False,
    )

    ranks = results_frame[
        "rank"
    ]

    top1 = int(
        (ranks == 1).sum()
    )

    top5 = int(
        (
            ranks.notna()
            & (ranks <= 5)
        ).sum()
    )

    top25 = int(
        (
            ranks.notna()
            & (ranks <= 25)
        ).sum()
    )

    mrr25 = float(
        results_frame[
            "reciprocal_rank"
        ].mean()
    )

    print()
    print(
        "Cross-polarity cosine diagnostic"
    )
    print(
        "--------------------------------"
    )

    print(
        f"Structures: "
        f"{len(results_frame)}"
    )

    print(
        f"MRR@25:    "
        f"{mrr25:.4f}"
    )

    print(
        f"Top-1:     "
        f"{top1}/{len(results_frame)} "
        f"({top1 / len(results_frame):.2%})"
    )

    print(
        f"Top-5:     "
        f"{top5}/{len(results_frame)} "
        f"({top5 / len(results_frame):.2%})"
    )

    print(
        f"Top-25:    "
        f"{top25}/{len(results_frame)} "
        f"({top25 / len(results_frame):.2%})"
    )

    print()

    candidate_counts = (
        results_frame[
            "candidate_structure_count"
        ]
    )

    print(
        "Candidate structures"
    )
    print(
        "--------------------"
    )

    print(
        f"Min:    "
        f"{candidate_counts.min():,.0f}"
    )

    print(
        f"Median: "
        f"{candidate_counts.median():,.0f}"
    )

    print(
        f"Mean:   "
        f"{candidate_counts.mean():,.2f}"
    )

    print(
        f"Max:    "
        f"{candidate_counts.max():,.0f}"
    )

    print()
    print(
        "Per-structure results"
    )
    print(
        "---------------------"
    )

    print(
        results_frame[
            [
                "inchikey14",
                "query_library",
                "candidate_structure_count",
                "truth_reference_spectrum_count",
                "truth_score",
                "rank",
            ]
        ]
        .sort_values(
            "rank",
            na_position="last",
        )
        .to_string(
            index=False
        )
    )

    total_elapsed = (
        time.monotonic()
        - started
    )

    print()
    print(
        f"Results written to: "
        f"{RESULTS_PATH}"
    )

    print(
        f"Elapsed: "
        f"{total_elapsed:.2f}s"
    )


if __name__ == "__main__":
    main()