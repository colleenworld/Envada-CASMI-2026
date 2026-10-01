from pathlib import Path
import json
import time

import numpy as np
import pandas as pd

from casmi26.chemistry.adducts import neutral_mass
from casmi26.retrieval.index import load_retrieval_index
from casmi26.retrieval.indexed import (
    search_structures_indexed_blockwise,
)
from casmi26.retrieval.mass_index import (
    build_neutral_mass_index,
    search_neutral_masses,
)


INDEX_DIR = Path(
    "data/processed/retrieval_index"
)

MANIFEST_PATH = Path(
    "data/processed/splits/retrieval_dev.parquet"
)

RESULTS_DIR = Path(
    "data/processed/results"
)

CHECKPOINT_PATH = (
    RESULTS_DIR
    / "retrieval_dev_baseline2_checkpoint.json"
)

RESULTS_PATH = (
    RESULTS_DIR
    / "retrieval_dev_baseline2.parquet"
)

NEUTRAL_MASS_TOLERANCE_DA = 0.01

BLOCK_SIZE = 25_000

# None runs the complete frozen 2,000-structure retrieval-dev benchmark.
MAX_STRUCTURES: int | None = None


def build_reference_pool(
    metadata: pd.DataFrame,
    vectors,
    manifest: pd.DataFrame,
):
    """
    Build exactly the same retrieval-dev reference pool used by
    Baseline 1.

    For each benchmark structure, spectra from its assigned query
    library are removed. Spectra for that structure from all other
    libraries remain valid references.
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

    reference_mask = np.fromiter(
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

    reference_indices = np.flatnonzero(
        reference_mask
    )

    reference_metadata = (
        metadata.iloc[
            reference_indices
        ]
        .reset_index(drop=True)
    )

    reference_vectors = vectors[
        reference_indices
    ]

    return (
        reference_metadata,
        reference_vectors,
    )


def infer_neutral_masses(
    precursor_mz: np.ndarray,
    adducts: np.ndarray,
) -> np.ndarray:
    """
    Infer neutral masses for spectra whose precursor m/z and
    adduct are understood.

    Unsupported or invalid observations receive NaN.
    """
    masses = np.full(
        len(precursor_mz),
        np.nan,
        dtype=np.float64,
    )

    for index, (
        mz,
        adduct,
    ) in enumerate(
        zip(
            precursor_mz,
            adducts,
            strict=True,
        )
    ):
        try:
            mz_value = float(
                mz
            )
        except (
            TypeError,
            ValueError,
        ):
            continue

        if not np.isfinite(
            mz_value
        ):
            continue

        value = neutral_mass(
            mz_value,
            str(adduct),
        )

        if (
            value is not None
            and np.isfinite(
                value
            )
        ):
            masses[
                index
            ] = value

    return masses


def find_indexed_rank(
    truth_candidate_id: int,
    scores: np.ndarray,
) -> int | None:
    """
    Return the 1-based truth rank from a candidate score vector.
    """
    truth_score = scores[
        truth_candidate_id
    ]

    if np.isneginf(
        truth_score
    ):
        return None

    return (
        1
        + int(
            np.count_nonzero(
                scores > truth_score
            )
        )
    )


def reciprocal_rank_at_25(
    rank: int | None,
) -> float:
    """
    Reciprocal rank under the competition's @25 cutoff.
    """
    if (
        rank is None
        or rank > 25
    ):
        return 0.0

    return (
        1.0
        / rank
    )


def load_checkpoint() -> list[dict]:
    if not CHECKPOINT_PATH.exists():
        return []

    with CHECKPOINT_PATH.open(
        "r"
    ) as handle:
        return json.load(
            handle
        )


def save_checkpoint(
    results: list[dict],
) -> None:
    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = (
        CHECKPOINT_PATH.with_suffix(
            ".tmp"
        )
    )

    with temporary_path.open(
        "w"
    ) as handle:
        json.dump(
            results,
            handle,
            indent=2,
        )

    temporary_path.replace(
        CHECKPOINT_PATH
    )


def save_results(
    results: list[dict],
    manifest: pd.DataFrame,
) -> pd.DataFrame:
    results_frame = pd.DataFrame(
        results
    )

    order = {
        str(key): index
        for index, key
        in enumerate(
            manifest[
                "inchikey14"
            ]
        )
    }

    results_frame[
        "_order"
    ] = (
        results_frame[
            "inchikey14"
        ].map(
            order
        )
    )

    results_frame = (
        results_frame
        .sort_values(
            "_order"
        )
        .drop(
            columns=[
                "_order",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    results_frame.to_parquet(
        RESULTS_PATH,
        index=False,
    )

    return results_frame


def summarize_results(
    frame: pd.DataFrame,
) -> pd.Series:
    if len(frame) == 0:
        return pd.Series(
            {
                "N": 0,
                "MRR@25": 0.0,
                "Top-1": 0.0,
                "Top-25": 0.0,
                "Truth survival": 0.0,
                "Median candidates": 0.0,
            }
        )

    return pd.Series(
        {
            "N": len(frame),
            "MRR@25": frame[
                "reciprocal_rank"
            ].mean(),
            "Top-1": (
                frame["rank"] == 1
            ).mean(),
            "Top-25": (
                frame["rank"].notna()
                & (
                    frame["rank"]
                    <= 25
                )
            ).mean(),
            "Truth survival": frame[
                "truth_survived"
            ].mean(),
            "Fallback": frame[
                "used_fallback"
            ].mean(),
            "Median candidates": frame[
                "candidate_count"
            ].median(),
        }
    )


def print_summary(
    results: pd.DataFrame,
) -> None:
    by_library = (
        results
        .groupby(
            "query_library"
        )
        .apply(
            summarize_results,
            include_groups=False,
        )
    )

    overall = summarize_results(
        results
    )

    print()
    print(
        "Results by query library"
    )
    print(
        "------------------------"
    )

    print(
        by_library.to_string(
            formatters={
                "N": (
                    lambda value:
                    f"{int(value):,}"
                ),
                "MRR@25": (
                    lambda value:
                    f"{value:.4f}"
                ),
                "Top-1": (
                    lambda value:
                    f"{value:.2%}"
                ),
                "Top-25": (
                    lambda value:
                    f"{value:.2%}"
                ),
                "Truth survival": (
                    lambda value:
                    f"{value:.2%}"
                ),
                "Fallback": (
                    lambda value:
                    f"{value:.2%}"
                ),
                "Median candidates": (
                    lambda value:
                    f"{value:,.0f}"
                ),
            }
        )
    )

    print()
    print(
        "Overall"
    )
    print(
        "-------"
    )

    print(
        f"Structures:        "
        f"{int(overall['N']):,}"
    )

    print(
        f"MRR@25:            "
        f"{overall['MRR@25']:.4f}"
    )

    print(
        f"Top-1:             "
        f"{overall['Top-1']:.2%}"
    )

    print(
        f"Top-25:            "
        f"{overall['Top-25']:.2%}"
    )

    print(
        f"Truth survival:    "
        f"{overall['Truth survival']:.2%}"
    )

    print(
        f"Fallback used:     "
        f"{overall['Fallback']:.2%}"
    )

    print(
        f"Median candidates: "
        f"{overall['Median candidates']:,.0f}"
    )

    print()
    print(
        "Candidate-count distribution"
    )
    print(
        "----------------------------"
    )

    candidate_counts = (
        results[
            "candidate_count"
        ]
    )

    print(
        f"Min:       "
        f"{candidate_counts.min():,.0f}"
    )

    print(
        f"Median:    "
        f"{candidate_counts.median():,.0f}"
    )

    print(
        f"Mean:      "
        f"{candidate_counts.mean():,.2f}"
    )

    print(
        f"90th pct:  "
        f"{candidate_counts.quantile(0.90):,.0f}"
    )

    print(
        f"95th pct:  "
        f"{candidate_counts.quantile(0.95):,.0f}"
    )

    print(
        f"99th pct:  "
        f"{candidate_counts.quantile(0.99):,.0f}"
    )

    print(
        f"Max:       "
        f"{candidate_counts.max():,.0f}"
    )

    filtered_truth = results[
        ~results[
            "truth_survived"
        ]
    ]

    print()
    print(
        "Truth filtered out"
    )
    print(
        "------------------"
    )

    print(
        f"Structures: "
        f"{len(filtered_truth):,}"
    )

    if len(
        filtered_truth
    ):
        print()

        print(
            filtered_truth[
                [
                    "inchikey14",
                    "query_library",
                    "query_spectrum_count",
                    "known_query_mass_count",
                    "candidate_count",
                    "used_fallback",
                ]
            ].to_string(
                index=False
            )
        )


def main() -> None:
    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ---------------------------------------------------------
    # Load frozen retrieval index and dev manifest.
    # ---------------------------------------------------------

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

    print(
        f"Index spectra:       "
        f"{len(metadata):,}"
    )

    print(
        f"Dev structures:      "
        f"{len(manifest):,}"
    )

    # ---------------------------------------------------------
    # Build exactly the same reference pool as Baseline 1.
    # ---------------------------------------------------------

    print()
    print(
        "Building dev reference pool..."
    )

    (
        reference_metadata,
        reference_vectors,
    ) = build_reference_pool(
        metadata=metadata,
        vectors=vectors,
        manifest=manifest,
    )

    removed = (
        len(metadata)
        - len(reference_metadata)
    )

    print(
        f"Reference spectra:   "
        f"{len(reference_metadata):,}"
    )

    print(
        f"Query spectra removed: "
        f"{removed:,}"
    )

    if removed != 19_812:
        raise RuntimeError(
            "Expected exactly 19,812 "
            "query spectra to be removed, "
            f"but removed {removed:,}"
        )

    # ---------------------------------------------------------
    # Encode candidate structures.
    # ---------------------------------------------------------

    print()
    print(
        "Encoding candidate structures..."
    )

    reference_keys = (
        reference_metadata[
            "inchikey14"
        ]
        .astype(str)
        .to_numpy()
    )

    (
        candidate_keys,
        reference_candidate_ids,
    ) = np.unique(
        reference_keys,
        return_inverse=True,
    )

    reference_candidate_ids = (
        reference_candidate_ids.astype(
            np.int64,
            copy=False,
        )
    )

    candidate_id_by_key = {
        key: candidate_id
        for candidate_id, key
        in enumerate(
            candidate_keys
        )
    }

    reference_modes = (
        reference_metadata[
            "ionization_mode"
        ]
        .astype(str)
        .to_numpy()
    )

    print(
        f"Candidate structures: "
        f"{len(candidate_keys):,}"
    )

    # ---------------------------------------------------------
    # Infer reference neutral masses and build the lookup index.
    # ---------------------------------------------------------

    print()
    print(
        "Inferring reference neutral masses..."
    )

    reference_precursor_mz = (
        reference_metadata[
            "precursor_mz"
        ].to_numpy()
    )

    reference_adducts = (
        reference_metadata[
            "adduct"
        ]
        .astype(str)
        .to_numpy()
    )

    reference_neutral_masses = (
        infer_neutral_masses(
            reference_precursor_mz,
            reference_adducts,
        )
    )

    known_reference_masses = int(
        np.count_nonzero(
            np.isfinite(
                reference_neutral_masses
            )
        )
    )

    print(
        f"Reference spectra with "
        f"neutral mass: "
        f"{known_reference_masses:,} / "
        f"{len(reference_metadata):,} "
        f"("
        f"{known_reference_masses / len(reference_metadata):.2%}"
        f")"
    )

    print(
        "Building neutral-mass index..."
    )

    mass_index = (
        build_neutral_mass_index(
            neutral_masses=(
                reference_neutral_masses
            ),
            modes=reference_modes,
        )
    )

    print(
        f"Mass-index partitions: "
        f"{', '.join(sorted(mass_index.partitions))}"
    )

    # ---------------------------------------------------------
    # Arrays used to locate each structure's query spectra.
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
    # Resume checkpoint if present.
    # ---------------------------------------------------------

    results = load_checkpoint()

    completed_keys = {
        str(
            result[
                "inchikey14"
            ]
        )
        for result in results
    }

    if results:
        print()
        print(
            "Resuming from checkpoint: "
            f"{len(results):,} structures "
            "already complete."
        )

    # ---------------------------------------------------------
    # Run Baseline 2.
    # ---------------------------------------------------------

    print()
    print(
        "Running retrieval-dev "
        "Baseline 2..."
    )

    print(
        f"Neutral-mass tolerance: "
        f"±{NEUTRAL_MASS_TOLERANCE_DA:.3f} Da"
    )

    print()

    run_started = (
        time.monotonic()
    )

    structures_run = 0

    for manifest_position, row in enumerate(
        manifest.itertuples(
            index=False
        ),
        start=1,
    ):
        truth_key = str(
            row.inchikey14
        )

        if truth_key in completed_keys:
            continue

        if (
            MAX_STRUCTURES is not None
            and structures_run
            >= MAX_STRUCTURES
        ):
            break

        query_library = str(
            row.query_library
        )

        # -----------------------------------------------------
        # Locate this structure's query spectra.
        # -----------------------------------------------------

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

        query_indices = (
            np.flatnonzero(
                query_mask
            )
        )

        if len(
            query_indices
        ) == 0:
            raise RuntimeError(
                "No query spectra found for "
                f"{truth_key} / "
                f"{query_library}"
            )

        query_metadata = (
            metadata.iloc[
                query_indices
            ]
        )

        query_vectors = (
            vectors[
                query_indices
            ]
        )

        query_modes = (
            query_metadata[
                "ionization_mode"
            ]
            .astype(str)
            .to_numpy()
        )

        # -----------------------------------------------------
        # Infer query neutral masses.
        # -----------------------------------------------------

        query_neutral_masses = (
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

        known_query_mass_count = int(
            np.count_nonzero(
                np.isfinite(
                    query_neutral_masses
                )
            )
        )

        # -----------------------------------------------------
        # Use the mass index to obtain compatible reference rows.
        # -----------------------------------------------------

        filtered_reference_indices = (
            search_neutral_masses(
                index=mass_index,
                neutral_masses=(
                    query_neutral_masses
                ),
                modes=query_modes,
                tolerance_da=(
                    NEUTRAL_MASS_TOLERANCE_DA
                ),
            )
        )

        used_fallback = (
            known_query_mass_count == 0
            or len(filtered_reference_indices) == 0
        )

        if used_fallback:
            search_reference_indices = np.arange(
                len(reference_metadata),
                dtype=np.int64,
            )
        else:
            search_reference_indices = (
                filtered_reference_indices
            )

        # Count unique candidate structures represented by the
        # filtered reference spectra.
        if len(
            filtered_reference_indices
        ):
            filtered_candidate_ids = (
                np.unique(
                    reference_candidate_ids[
                        filtered_reference_indices
                    ]
                )
            )

            candidate_count = len(
                filtered_candidate_ids
            )
        else:
            candidate_count = 0

        truth_candidate_id = (
            candidate_id_by_key.get(
                truth_key
            )
        )

        truth_in_mass_filter = (
            truth_candidate_id
            is not None
            and len(
                filtered_reference_indices
            )
            and np.any(
                reference_candidate_ids[
                    filtered_reference_indices
                ]
                == truth_candidate_id
            )
        )

        # -----------------------------------------------------
        # Run same-polarity cosine retrieval.
        #
        # Normally search only neutral-mass-compatible reference
        # rows. If no query neutral mass is known or the mass
        # filter produces no candidates, fall back to the complete
        # reference pool. Indexed retrieval still enforces
        # same-polarity matching via query_modes/reference_modes.
        # -----------------------------------------------------

        structure_started = (
            time.monotonic()
        )

        query_structure_ids = (
            np.zeros(
                len(query_indices),
                dtype=np.int64,
            )
        )

        scores = (
            search_structures_indexed_blockwise(
                query_vectors=(
                    query_vectors
                ),
                query_structure_ids=(
                    query_structure_ids
                ),
                query_modes=(
                    query_modes
                ),
                reference_vectors=(
                    reference_vectors[
                        search_reference_indices
                    ]
                ),
                reference_candidate_ids=(
                    reference_candidate_ids[
                        search_reference_indices
                    ]
                ),
                reference_modes=(
                    reference_modes[
                        search_reference_indices
                    ]
                ),
                query_structure_count=1,
                candidate_count=(
                    len(candidate_keys)
                ),
                block_size=BLOCK_SIZE,
            )[0]
        )

        if truth_candidate_id is None:
            rank = None
        else:
            rank = find_indexed_rank(
                truth_candidate_id,
                scores,
            )

        truth_survived = (
            truth_in_mass_filter
            and rank is not None
        )

        elapsed = (
            time.monotonic()
            - structure_started
        )

        result = {
            "inchikey14": truth_key,
            "query_library": (
                query_library
            ),
            "query_spectrum_count": (
                len(query_indices)
            ),
            "known_query_mass_count": (
                known_query_mass_count
            ),
            "filtered_reference_spectrum_count": (
                len(
                    filtered_reference_indices
                )
            ),
            "candidate_count": (
                candidate_count
            ),
            "used_fallback": (
                bool(
                    used_fallback
                )
            ),
            "truth_in_mass_filter": (
                bool(
                    truth_in_mass_filter
                )
            ),
            "truth_survived": (
                bool(
                    truth_survived
                )
            ),
            "rank": rank,
            "reciprocal_rank": (
                reciprocal_rank_at_25(
                    rank
                )
            ),
            "elapsed_seconds": (
                elapsed
            ),
        }

        results.append(
            result
        )

        completed_keys.add(
            truth_key
        )

        structures_run += 1

        # -----------------------------------------------------
        # Progress.
        # -----------------------------------------------------

        total_elapsed = (
            time.monotonic()
            - run_started
        )

        average_seconds = (
            total_elapsed
            / structures_run
        )

        remaining_structures = (
            len(manifest)
            - len(results)
        )

        eta_seconds = (
            average_seconds
            * remaining_structures
        )

        rank_text = (
            str(rank)
            if rank is not None
            else "None"
        )

        if used_fallback:
            route_text = "FALLBACK"
        elif truth_survived:
            route_text = "PASS"
        else:
            route_text = "FILTERED"

        print(
            f"{manifest_position:>4}/"
            f"{len(manifest)}  "
            f"{truth_key}  "
            f"{query_library:<12}  "
            f"spectra="
            f"{len(query_indices):>3}  "
            f"mass="
            f"{known_query_mass_count:>3}  "
            f"candidates="
            f"{candidate_count:>5,}  "
            f"rank="
            f"{rank_text:>6}  "
            f"time="
            f"{elapsed:>6.3f}s  "
            f"{route_text}  "
            f"ETA="
            f"{eta_seconds / 60:>6.1f}m"
        )

        # -----------------------------------------------------
        # Checkpoint after every structure.
        # -----------------------------------------------------

        save_checkpoint(
            results
        )

        del scores

    # ---------------------------------------------------------
    # Save and summarize all results obtained so far.
    # ---------------------------------------------------------

    if not results:
        print(
            "No structures were evaluated."
        )
        return

    results_frame = save_results(
        results=results,
        manifest=manifest,
    )

    print_summary(
        results_frame
    )

    print()
    print(
        f"Results written to: "
        f"{RESULTS_PATH}"
    )

    if len(
        results
    ) < len(
        manifest
    ):
        print()
        print(
            "NOTE: This is a partial "
            "Baseline 2 validation run."
        )

        print(
            f"Completed "
            f"{len(results):,} of "
            f"{len(manifest):,} "
            f"structures."
        )

        print(
            "Do not record these metrics "
            "as the final Baseline 2 result."
        )

        print(
            "Inspect the checkpoint/results before "
            "resuming the incomplete run."
        )

    else:
        print()
        print(
            "Baseline 2 evaluation complete."
        )


if __name__ == "__main__":
    main()