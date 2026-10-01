from pathlib import Path
import json
import time

import numpy as np
import pandas as pd

from casmi26.retrieval.index import (
    load_retrieval_index,
)
from casmi26.retrieval.indexed import (
    search_structures_indexed_blockwise,
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
    / "retrieval_dev_baseline1_checkpoint.json"
)

RESULTS_PATH = (
    RESULTS_DIR
    / "retrieval_dev_baseline1.parquet"
)

BLOCK_SIZE = 25_000

# Target maximum number of query spectra in a batch.
#
# Structures are never split across batches, so a batch can exceed
# this value when a single structure has more spectra than the
# target.
MAX_QUERY_SPECTRA = 64

# Keep this at 5 for the initial performance test.
#
# Change to None only after the timing run looks good.
MAX_BATCHES: int | None = None


def build_reference_pool(
    metadata: pd.DataFrame,
    vectors,
    manifest: pd.DataFrame,
):
    """
    Build the retrieval-dev reference pool.

    For each benchmark structure, spectra from its assigned query
    library are removed. Spectra belonging to that same structure
    in other libraries remain valid retrieval references.
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


def build_query_batches(
    manifest: pd.DataFrame,
    metadata: pd.DataFrame,
    max_spectra: int = 64,
) -> list[list[dict]]:
    """
    Build batches containing complete query structures.

    Structures are never split between batches.
    """
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

    batches: list[
        list[dict]
    ] = []

    current_batch: list[
        dict
    ] = []

    current_spectra = 0

    for row in manifest.itertuples(
        index=False
    ):
        truth_key = str(
            row.inchikey14
        )

        query_library = str(
            row.query_library
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

        query_indices = np.flatnonzero(
            query_mask
        )

        if len(query_indices) == 0:
            raise RuntimeError(
                "No query spectra found for "
                f"{truth_key} / "
                f"{query_library}"
            )

        item = {
            "inchikey14": (
                truth_key
            ),
            "query_library": (
                query_library
            ),
            "query_indices": (
                query_indices
            ),
        }

        if (
            current_batch
            and (
                current_spectra
                + len(query_indices)
                > max_spectra
            )
        ):
            batches.append(
                current_batch
            )

            current_batch = []
            current_spectra = 0

        current_batch.append(
            item
        )

        current_spectra += len(
            query_indices
        )

    if current_batch:
        batches.append(
            current_batch
        )

    return batches


def find_indexed_rank(
    truth_candidate_id: int,
    scores: np.ndarray,
) -> int | None:
    """
    Return the 1-based rank of the truth candidate.

    Candidates that were never compared have a score of -inf and
    therefore do not participate in the ranking.
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
    Return reciprocal rank under the competition's @25 cutoff.
    """
    if (
        rank is None
        or rank > 25
    ):
        return 0.0

    return 1.0 / rank


def load_checkpoint() -> list[dict]:
    """
    Load completed structure results, if a checkpoint exists.
    """
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
    """
    Atomically save the current result checkpoint.
    """
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


def summarize_results(
    frame: pd.DataFrame,
) -> pd.Series:
    """
    Calculate benchmark metrics for a result subset.
    """
    if len(frame) == 0:
        return pd.Series(
            {
                "N": 0,
                "MRR@25": 0.0,
                "Top-1": 0.0,
                "Top-25": 0.0,
            }
        )

    return pd.Series(
        {
            "N": len(frame),
            "MRR@25": frame[
                "reciprocal_rank"
            ].mean(),
            "Top-1": (
                frame[
                    "rank"
                ]
                == 1
            ).mean(),
            "Top-25": (
                frame[
                    "rank"
                ].notna()
                & (
                    frame[
                        "rank"
                    ]
                    <= 25
                )
            ).mean(),
        }
    )


def print_summary(
    results: pd.DataFrame,
) -> None:
    """
    Print overall and per-query-library benchmark metrics.
    """
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
            }
        )
    )

    print()
    print("Overall")
    print("-------")

    print(
        f"Structures: "
        f"{int(overall['N']):,}"
    )

    print(
        f"MRR@25:     "
        f"{overall['MRR@25']:.4f}"
    )

    print(
        f"Top-1:      "
        f"{overall['Top-1']:.2%}"
    )

    print(
        f"Top-25:     "
        f"{overall['Top-25']:.2%}"
    )


def save_final_results(
    results: list[dict],
    manifest: pd.DataFrame,
) -> pd.DataFrame:
    """
    Save completed results in manifest order.
    """
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
        ]
        .map(order)
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


def main() -> None:
    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ---------------------------------------------------------
    # Load index and benchmark manifest.
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
    # Build the retrieval-dev reference pool.
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
            "dev query spectra to be removed, "
            f"but removed {removed:,}"
        )

    # ---------------------------------------------------------
    # Encode candidate structures once.
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
    # Build query batches.
    # ---------------------------------------------------------

    print()
    print(
        "Building query batches..."
    )

    batches = build_query_batches(
        manifest=manifest,
        metadata=metadata,
        max_spectra=MAX_QUERY_SPECTRA,
    )

    total_batches = len(
        batches
    )

    total_query_spectra = sum(
        len(
            item[
                "query_indices"
            ]
        )
        for batch in batches
        for item in batch
    )

    print(
        f"Query batches:       "
        f"{total_batches:,}"
    )

    print(
        f"Query spectra:       "
        f"{total_query_spectra:,}"
    )

    print(
        f"Target spectra/batch: "
        f"{MAX_QUERY_SPECTRA}"
    )

    if total_query_spectra != 19_812:
        raise RuntimeError(
            "Expected exactly 19,812 "
            "query spectra in batches, "
            f"but found "
            f"{total_query_spectra:,}"
        )

    # ---------------------------------------------------------
    # Resume from checkpoint if one exists.
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

    total_structures = len(
        manifest
    )

    # ---------------------------------------------------------
    # Evaluate batches.
    # ---------------------------------------------------------

    print()
    print(
        "Running retrieval-dev "
        "Baseline 1..."
    )
    print()

    run_started = time.monotonic()

    batches_run = 0

    for batch_number, batch in enumerate(
        batches,
        start=1,
    ):
        pending = [
            item
            for item in batch
            if (
                item[
                    "inchikey14"
                ]
                not in completed_keys
            )
        ]

        if not pending:
            continue

        # During the timing run, process only MAX_BATCHES new
        # batches. With MAX_BATCHES=None, run the entire benchmark.
        if (
            MAX_BATCHES is not None
            and batches_run
            >= MAX_BATCHES
        ):
            break

        # -----------------------------------------------------
        # Assemble query spectra for this batch.
        # -----------------------------------------------------

        query_indices = np.concatenate(
            [
                item[
                    "query_indices"
                ]
                for item in pending
            ]
        )

        query_structure_ids = (
            np.concatenate(
                [
                    np.repeat(
                        query_id,
                        len(
                            item[
                                "query_indices"
                            ]
                        ),
                    )
                    for query_id, item
                    in enumerate(
                        pending
                    )
                ]
            )
            .astype(
                np.int64,
                copy=False,
            )
        )

        query_modes = (
            metadata.iloc[
                query_indices
            ][
                "ionization_mode"
            ]
            .astype(str)
            .to_numpy()
        )

        query_vectors = vectors[
            query_indices
        ]

        # -----------------------------------------------------
        # Numeric-ID retrieval.
        # -----------------------------------------------------

        batch_started = (
            time.monotonic()
        )

        batch_scores = (
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
                    reference_vectors
                ),
                reference_candidate_ids=(
                    reference_candidate_ids
                ),
                reference_modes=(
                    reference_modes
                ),
                query_structure_count=(
                    len(pending)
                ),
                candidate_count=(
                    len(candidate_keys)
                ),
                block_size=BLOCK_SIZE,
            )
        )

        batch_elapsed = (
            time.monotonic()
            - batch_started
        )

        # -----------------------------------------------------
        # Rank the truth structure for each query.
        # -----------------------------------------------------

        for query_id, item in enumerate(
            pending
        ):
            truth_key = str(
                item[
                    "inchikey14"
                ]
            )

            truth_candidate_id = (
                candidate_id_by_key.get(
                    truth_key
                )
            )

            if truth_candidate_id is None:
                rank = None
            else:
                rank = (
                    find_indexed_rank(
                        truth_candidate_id,
                        batch_scores[
                            query_id
                        ],
                    )
                )

            result = {
                "inchikey14": (
                    truth_key
                ),
                "query_library": (
                    item[
                        "query_library"
                    ]
                ),
                "query_spectrum_count": (
                    len(
                        item[
                            "query_indices"
                        ]
                    )
                ),
                "rank": rank,
                "reciprocal_rank": (
                    reciprocal_rank_at_25(
                        rank
                    )
                ),
            }

            results.append(
                result
            )

            completed_keys.add(
                truth_key
            )

        batches_run += 1

        # -----------------------------------------------------
        # Progress and ETA.
        # -----------------------------------------------------

        total_elapsed = (
            time.monotonic()
            - run_started
        )

        average_batch_time = (
            total_elapsed
            / batches_run
        )

        remaining_batches = sum(
            any(
                item[
                    "inchikey14"
                ]
                not in completed_keys
                for item in future_batch
            )
            for future_batch in batches[
                batch_number:
            ]
        )

        eta_seconds = (
            average_batch_time
            * remaining_batches
        )

        print(
            f"Batch "
            f"{batch_number:>3}/"
            f"{total_batches}  "
            f"structures="
            f"{len(pending):>3}  "
            f"spectra="
            f"{len(query_indices):>3}  "
            f"time="
            f"{batch_elapsed:6.2f}s  "
            f"complete="
            f"{len(results):>4}/"
            f"{total_structures}  "
            f"ETA="
            f"{eta_seconds / 60:6.1f}m"
        )

        # -----------------------------------------------------
        # Checkpoint after every completed batch.
        # -----------------------------------------------------

        save_checkpoint(
            results
        )

        # The score matrix can be tens of MB. Explicitly release
        # it before processing the next batch.
        del batch_scores

    # ---------------------------------------------------------
    # Save all results obtained so far.
    # ---------------------------------------------------------

    if not results:
        print(
            "No structures were evaluated."
        )
        return

    results_frame = (
        save_final_results(
            results=results,
            manifest=manifest,
        )
    )

    print_summary(
        results_frame
    )

    print()
    print(
        f"Results written to: "
        f"{RESULTS_PATH}"
    )

    # ---------------------------------------------------------
    # Make it very clear whether this is a partial timing run or
    # the complete Baseline 1 result.
    # ---------------------------------------------------------

    if len(results) < total_structures:
        print()
        print(
            "NOTE: This is a partial timing run."
        )

        print(
            f"Completed "
            f"{len(results):,} of "
            f"{total_structures:,} "
            f"structures."
        )

        print(
            "Do not record these metrics as "
            "the Baseline 1 result."
        )

        print(
            "Set MAX_BATCHES = None for "
            "the complete evaluation."
        )
    else:
        print()
        print(
            "Baseline 1 evaluation complete."
        )


if __name__ == "__main__":
    main()