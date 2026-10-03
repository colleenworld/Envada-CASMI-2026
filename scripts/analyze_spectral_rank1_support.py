from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from casmi26.retrieval.index import load_retrieval_index


INDEX_DIR = Path(
    "data/processed/retrieval_index"
)

MANIFEST_PATH = Path(
    "data/processed/splits/retrieval_dev.parquet"
)

RESCUE_PATH = Path(
    "data/processed/results/"
    "retrieval_dev_spectral_rank1_rescue.parquet"
)

CANDIDATE_MAP_PATH = Path(
    "data/processed/results/"
    "retrieval_dev_baseline3_candidate_map.parquet"
)

OUTPUT_PATH = Path(
    "data/processed/results/"
    "retrieval_dev_spectral_rank1_support.parquet"
)


def build_reference_mask(
    metadata: pd.DataFrame,
    manifest: pd.DataFrame,
) -> np.ndarray:
    """
    Recreate the exact frozen retrieval-dev reference pool.
    """
    structure_values = (
        metadata["inchikey14"]
        .astype(str)
        .to_numpy()
    )

    library_values = (
        metadata["ingest_lib"]
        .astype(str)
        .to_numpy()
    )

    reference_mask = np.ones(
        len(metadata),
        dtype=bool,
    )

    for row in manifest.itertuples(
        index=False
    ):
        reference_mask &= ~(
            (
                structure_values
                == str(row.inchikey14)
            )
            & (
                library_values
                == str(row.query_library)
            )
        )

    return reference_mask


def finite_similarity_scores(
    query_vectors: sparse.csr_matrix,
    reference_vectors: sparse.csr_matrix,
) -> np.ndarray:
    """
    Calculate all query-spectrum x candidate-reference-spectrum
    cosine similarities.

    The stored vectors are already L2-normalized, so sparse matrix
    multiplication gives cosine similarity directly.
    """
    if (
        query_vectors.shape[0] == 0
        or reference_vectors.shape[0] == 0
    ):
        return np.empty(
            0,
            dtype=np.float32,
        )

    scores = (
        query_vectors
        @ reference_vectors.T
    )

    if sparse.issparse(scores):
        scores = scores.toarray()

    values = np.asarray(
        scores,
        dtype=np.float32,
    ).ravel()

    return values[
        np.isfinite(values)
    ]


def top_values(
    values: np.ndarray,
    n: int,
) -> np.ndarray:
    """
    Return the n largest values, descending.

    For these candidate-level diagnostics the arrays are small
    enough that a full sort is preferable to more complicated
    partial-selection logic.
    """
    if len(values) == 0:
        return np.empty(
            0,
            dtype=np.float32,
        )

    ordered = np.sort(
        values
    )[::-1]

    return ordered[:n]


def nth_best(
    values: np.ndarray,
    n: int,
) -> float:
    """
    1-based nth-best score.
    """
    best = top_values(
        values,
        n,
    )

    if len(best) < n:
        return float("nan")

    return float(
        best[n - 1]
    )


def top_mean(
    values: np.ndarray,
    n: int,
) -> float:
    """
    Mean of the best min(n, available) similarities.

    We also persist the number of available pairs separately, so a
    top-5 mean based on only two comparisons can be distinguished
    from one based on five or more.
    """
    best = top_values(
        values,
        n,
    )

    if len(best) == 0:
        return float("nan")

    return float(
        np.mean(best)
    )


def summarize_distribution(
    frame: pd.DataFrame,
    column: str,
) -> dict[str, float]:
    values = pd.to_numeric(
        frame[column],
        errors="coerce",
    ).to_numpy(
        dtype=np.float64
    )

    values = values[
        np.isfinite(values)
    ]

    if len(values) == 0:
        return {
            "n": 0,
            "median": float("nan"),
            "p25": float("nan"),
            "p75": float("nan"),
            "p90": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
        }

    return {
        "n": len(values),
        "median": float(
            np.quantile(
                values,
                0.50,
            )
        ),
        "p25": float(
            np.quantile(
                values,
                0.25,
            )
        ),
        "p75": float(
            np.quantile(
                values,
                0.75,
            )
        ),
        "p90": float(
            np.quantile(
                values,
                0.90,
            )
        ),
        "min": float(
            np.min(values)
        ),
        "max": float(
            np.max(values)
        ),
    }


def print_distribution(
    label: str,
    frame: pd.DataFrame,
    column: str,
) -> None:
    stats = summarize_distribution(
        frame,
        column,
    )

    print(
        f"{label:<12} "
        f"n={stats['n']:>3} "
        f"median={stats['median']:>10.6f} "
        f"p25={stats['p25']:>10.6f} "
        f"p75={stats['p75']:>10.6f} "
        f"p90={stats['p90']:>10.6f} "
        f"min={stats['min']:>10.6f} "
        f"max={stats['max']:>10.6f}"
    )


def main() -> None:
    print(
        "Loading retrieval artifacts..."
    )

    metadata, vectors = load_retrieval_index(
        INDEX_DIR
    )

    manifest = pd.read_parquet(
        MANIFEST_PATH
    )

    rescue = pd.read_parquet(
        RESCUE_PATH
    )

    candidate_map = pd.read_parquet(
        CANDIDATE_MAP_PATH
    )

    # ---------------------------------------------------------
    # Validate index schema.
    # ---------------------------------------------------------

    required_columns = {
        "inchikey14",
        "ingest_lib",
        "ionization_mode",
    }

    missing = (
        required_columns
        - set(metadata.columns)
    )

    if missing:
        raise RuntimeError(
            "Retrieval metadata is missing columns: "
            f"{sorted(missing)}"
        )

    if vectors.shape[0] != len(metadata):
        raise RuntimeError(
            "Metadata/vector row mismatch."
        )

    # ---------------------------------------------------------
    # Recreate the population from the previous diagnostic:
    #
    # spectral rank 1 AND Tier 3.
    # ---------------------------------------------------------

    population = rescue[
        rescue["rank1_present"].astype(bool)
        & (
            rescue["rank1_tier"]
            == 3
        )
    ].copy()

    if len(population) != 459:
        raise RuntimeError(
            "Expected 459 Tier-3 spectral-rank-1 "
            f"queries, found {len(population)}."
        )

    truth_count = int(
        population[
            "rank1_is_truth"
        ].sum()
    )

    if truth_count != 61:
        raise RuntimeError(
            "Expected 61 correct Tier-3 rank-1 "
            f"candidates, found {truth_count}."
        )

    print(
        f"Tier-3 rank-1 population: "
        f"{len(population):,}"
    )

    print(
        f"Correct: "
        f"{truth_count:,}"
    )

    print(
        f"Incorrect: "
        f"{len(population) - truth_count:,}"
    )

    # ---------------------------------------------------------
    # Recreate frozen reference pool.
    # ---------------------------------------------------------

    reference_mask = build_reference_mask(
        metadata=metadata,
        manifest=manifest,
    )

    removed = int(
        (~reference_mask).sum()
    )

    if removed != 19_812:
        raise RuntimeError(
            "Reference-pool reproduction failed: "
            f"removed {removed:,}, expected 19,812."
        )

    reference_indices = np.flatnonzero(
        reference_mask
    )

    print(
        f"Reference rows: "
        f"{len(reference_indices):,}"
    )

    print(
        f"Removed query rows: "
        f"{removed:,}"
    )

    # ---------------------------------------------------------
    # Extract metadata arrays.
    # ---------------------------------------------------------

    structure_values = (
        metadata["inchikey14"]
        .astype(str)
        .to_numpy()
    )

    library_values = (
        metadata["ingest_lib"]
        .astype(str)
        .to_numpy()
    )

    mode_values = (
        metadata["ionization_mode"]
        .astype(str)
        .to_numpy()
    )

    # ---------------------------------------------------------
    # Validate/load authoritative candidate namespace.
    # ---------------------------------------------------------

    candidate_map = (
        candidate_map
        .sort_values(
            "candidate_id"
        )
        .reset_index(
            drop=True
        )
    )

    if len(candidate_map) != 275_810:
        raise RuntimeError(
            "Unexpected candidate-map size: "
            f"{len(candidate_map):,}; "
            "expected 275,810."
        )

    candidate_ids = (
        candidate_map[
            "candidate_id"
        ]
        .to_numpy(
            dtype=np.int64
        )
    )

    expected_ids = np.arange(
        len(candidate_map),
        dtype=np.int64,
    )

    if not np.array_equal(
        candidate_ids,
        expected_ids,
    ):
        raise RuntimeError(
            "Candidate IDs are not contiguous."
        )

    candidate_keys = (
        candidate_map[
            "inchikey14"
        ]
        .astype(str)
        .to_numpy()
    )

    candidate_id_by_key = {
        key: candidate_id
        for candidate_id, key
        in enumerate(candidate_keys)
    }

    print(
        f"Candidate map: "
        f"{len(candidate_keys):,} structures"
    )

    # ---------------------------------------------------------
    # Determine which candidate structures we actually need.
    #
    # Only the 459 rank-1 candidates are needed, so we avoid
    # building lookup structures for all 275,810 candidates.
    # ---------------------------------------------------------

    target_candidate_ids = set(
        population[
            "rank1_candidate_id"
        ]
        .astype(int)
        .tolist()
    )

    target_candidate_keys = {
        str(
            candidate_keys[
                candidate_id
            ]
        )
        for candidate_id
        in target_candidate_ids
    }

    print(
        f"Distinct rank-1 candidates: "
        f"{len(target_candidate_ids):,}"
    )

    # ---------------------------------------------------------
    # Build reference-row lookup only for target candidates.
    # ---------------------------------------------------------

    target_reference_mask = (
        reference_mask
        & np.isin(
            structure_values,
            list(
                target_candidate_keys
            ),
        )
    )

    target_reference_indices = np.flatnonzero(
        target_reference_mask
    )

    candidate_reference_rows: dict[
        int,
        np.ndarray,
    ] = {}

    target_reference_keys = structure_values[
        target_reference_indices
    ]

    target_reference_candidate_ids = np.asarray(
        [
            candidate_id_by_key[key]
            for key in target_reference_keys
        ],
        dtype=np.int64,
    )

    order = np.argsort(
        target_reference_candidate_ids,
        kind="stable",
    )

    sorted_candidate_ids = (
        target_reference_candidate_ids[
            order
        ]
    )

    sorted_reference_indices = (
        target_reference_indices[
            order
        ]
    )

    unique_ids, starts = np.unique(
        sorted_candidate_ids,
        return_index=True,
    )

    ends = np.r_[
        starts[1:],
        len(sorted_candidate_ids),
    ]

    for candidate_id, start, end in zip(
        unique_ids,
        starts,
        ends,
        strict=True,
    ):
        candidate_reference_rows[
            int(candidate_id)
        ] = sorted_reference_indices[
            start:end
        ]

    missing_candidate_rows = (
        target_candidate_ids
        - set(
            candidate_reference_rows
        )
    )

    if missing_candidate_rows:
        raise RuntimeError(
            "Some Tier-3 rank-1 candidates have "
            "no reference spectra: "
            f"{sorted(missing_candidate_rows)[:10]}"
        )

    # ---------------------------------------------------------
    # Locate query spectra.
    # ---------------------------------------------------------

    query_rows_by_key: dict[
        tuple[str, str],
        np.ndarray,
    ] = {}

    for row in manifest.itertuples(
        index=False
    ):
        key = (
            str(row.inchikey14),
            str(row.query_library),
        )

        query_rows = np.flatnonzero(
            (
                structure_values
                == key[0]
            )
            & (
                library_values
                == key[1]
            )
        )

        if len(query_rows) == 0:
            raise RuntimeError(
                "No query spectra found for "
                f"{key[0]} / {key[1]}"
            )

        query_rows_by_key[
            key
        ] = query_rows

    # ---------------------------------------------------------
    # Calculate candidate support.
    #
    # IMPORTANT:
    #
    # Baseline 1/3 spectral retrieval is same-polarity. Therefore
    # support features are calculated only from query/reference
    # pairs with the same ionization mode.
    #
    # We measure both pair-level support and library-level support.
    # ---------------------------------------------------------

    print(
        "Calculating spectral support..."
    )

    rows: list[
        dict[str, object]
    ] = []

    for position, row in enumerate(
        population.itertuples(
            index=False
        ),
        start=1,
    ):
        structure_key = str(
            row.inchikey14
        )

        query_library = str(
            row.query_library
        )

        candidate_id = int(
            row.rank1_candidate_id
        )

        candidate_key = str(
            candidate_keys[
                candidate_id
            ]
        )

        query_rows = query_rows_by_key[
            (
                structure_key,
                query_library,
            )
        ]

        candidate_rows = (
            candidate_reference_rows[
                candidate_id
            ]
        )

        all_scores: list[
            np.ndarray
        ] = []

        # Each entry is:
        #
        #   library -> best same-polarity cosine from that library
        #
        library_best_scores: dict[
            str,
            float,
        ] = {}

        same_polarity_reference_count = 0

        for query_mode in np.unique(
            mode_values[
                query_rows
            ]
        ):
            mode_query_rows = query_rows[
                mode_values[
                    query_rows
                ]
                == query_mode
            ]

            mode_candidate_rows = candidate_rows[
                mode_values[
                    candidate_rows
                ]
                == query_mode
            ]

            if (
                len(mode_query_rows) == 0
                or len(mode_candidate_rows) == 0
            ):
                continue

            same_polarity_reference_count += len(
                mode_candidate_rows
            )

            scores = finite_similarity_scores(
                vectors[
                    mode_query_rows
                ],
                vectors[
                    mode_candidate_rows
                ],
            )

            if len(scores):
                all_scores.append(
                    scores
                )

            # Calculate the best score independently for each
            # reference library.
            mode_libraries = np.unique(
                library_values[
                    mode_candidate_rows
                ]
            )

            for library in mode_libraries:
                library_rows = mode_candidate_rows[
                    library_values[
                        mode_candidate_rows
                    ]
                    == library
                ]

                library_scores = (
                    finite_similarity_scores(
                        vectors[
                            mode_query_rows
                        ],
                        vectors[
                            library_rows
                        ],
                    )
                )

                if len(library_scores) == 0:
                    continue

                library_best = float(
                    np.max(
                        library_scores
                    )
                )

                previous = (
                    library_best_scores.get(
                        str(library)
                    )
                )

                if (
                    previous is None
                    or library_best > previous
                ):
                    library_best_scores[
                        str(library)
                    ] = library_best

        if all_scores:
            pair_scores = np.concatenate(
                all_scores
            )
        else:
            pair_scores = np.empty(
                0,
                dtype=np.float32,
            )

        library_scores = np.asarray(
            list(
                library_best_scores.values()
            ),
            dtype=np.float32,
        )

        # These fixed similarity levels are descriptive only.
        # We are NOT selecting one as a rescue threshold.
        pair_ge_090 = int(
            np.sum(
                pair_scores >= 0.90
            )
        )

        pair_ge_095 = int(
            np.sum(
                pair_scores >= 0.95
            )
        )

        pair_ge_099 = int(
            np.sum(
                pair_scores >= 0.99
            )
        )

        library_ge_090 = int(
            np.sum(
                library_scores >= 0.90
            )
        )

        library_ge_095 = int(
            np.sum(
                library_scores >= 0.95
            )
        )

        library_ge_099 = int(
            np.sum(
                library_scores >= 0.99
            )
        )

        rows.append(
            {
                "inchikey14": structure_key,
                "query_library": query_library,
                "rank1_candidate_id": candidate_id,
                "rank1_candidate_inchikey14": candidate_key,
                "rank1_is_truth": bool(
                    row.rank1_is_truth
                ),
                "current_rank": row.current_rank,
                "rank1_cosine": row.rank1_cosine,

                "query_spectrum_count": int(
                    len(query_rows)
                ),

                "candidate_reference_spectrum_count": int(
                    len(candidate_rows)
                ),

                "same_polarity_reference_spectrum_count": int(
                    same_polarity_reference_count
                ),

                "same_polarity_pair_count": int(
                    len(pair_scores)
                ),

                "support_best": nth_best(
                    pair_scores,
                    1,
                ),

                "support_second": nth_best(
                    pair_scores,
                    2,
                ),

                "support_third": nth_best(
                    pair_scores,
                    3,
                ),

                "support_top5_mean": top_mean(
                    pair_scores,
                    5,
                ),

                "support_top10_mean": top_mean(
                    pair_scores,
                    10,
                ),

                "support_pair_ge_090": (
                    pair_ge_090
                ),

                "support_pair_ge_095": (
                    pair_ge_095
                ),

                "support_pair_ge_099": (
                    pair_ge_099
                ),

                "support_library_count": int(
                    len(
                        library_best_scores
                    )
                ),

                "support_library_ge_090": (
                    library_ge_090
                ),

                "support_library_ge_095": (
                    library_ge_095
                ),

                "support_library_ge_099": (
                    library_ge_099
                ),

                "support_library_second": nth_best(
                    library_scores,
                    2,
                ),

                "support_library_third": nth_best(
                    library_scores,
                    3,
                ),
            }
        )

        if (
            position % 50 == 0
            or position
            == len(population)
        ):
            print(
                f"  {position:,} / "
                f"{len(population):,}"
            )

    analysis = pd.DataFrame(
        rows
    )

    # ---------------------------------------------------------
    # Integrity checks.
    # ---------------------------------------------------------

    if len(analysis) != 459:
        raise RuntimeError(
            "Expected 459 output rows, "
            f"found {len(analysis)}."
        )

    output_truth_count = int(
        analysis[
            "rank1_is_truth"
        ].sum()
    )

    if output_truth_count != 61:
        raise RuntimeError(
            "Expected 61 truth rows, "
            f"found {output_truth_count}."
        )

    expected_keys = candidate_keys[
        analysis[
            "rank1_candidate_id"
        ].to_numpy(
            dtype=np.int64
        )
    ]

    if not np.array_equal(
        expected_keys,
        analysis[
            "rank1_candidate_inchikey14"
        ]
        .astype(str)
        .to_numpy(),
    ):
        raise RuntimeError(
            "Candidate mapping validation failed."
        )

    # The best support score should reproduce the stored spectral
    # rank-1 cosine, except for tiny floating-point differences.
    stored_best = pd.to_numeric(
        analysis[
            "rank1_cosine"
        ],
        errors="coerce",
    ).to_numpy(
        dtype=np.float64
    )

    calculated_best = pd.to_numeric(
        analysis[
            "support_best"
        ],
        errors="coerce",
    ).to_numpy(
        dtype=np.float64
    )

    comparable = (
        np.isfinite(stored_best)
        & np.isfinite(
            calculated_best
        )
    )

    if not np.allclose(
        stored_best[
            comparable
        ],
        calculated_best[
            comparable
        ],
        rtol=1e-5,
        atol=1e-6,
    ):
        differences = np.abs(
            stored_best[
                comparable
            ]
            - calculated_best[
                comparable
            ]
        )

        raise RuntimeError(
            "Best-score reproduction failed. "
            f"Maximum difference: "
            f"{np.max(differences):.8f}"
        )

    print(
        "Spectral-best reproduction: PASS"
    )

    # ---------------------------------------------------------
    # Correct vs incorrect distributions.
    # ---------------------------------------------------------

    correct = analysis[
        analysis[
            "rank1_is_truth"
        ]
    ].copy()

    incorrect = analysis[
        ~analysis[
            "rank1_is_truth"
        ]
    ].copy()

    print()
    print(
        "Tier-3 spectral-rank-1 support analysis"
    )

    print(
        "--------------------------------------"
    )

    print(
        f"Population: "
        f"{len(analysis):,}"
    )

    print(
        f"Correct:    "
        f"{len(correct):,}"
    )

    print(
        f"Incorrect:  "
        f"{len(incorrect):,}"
    )

    metrics = [
        (
            "support_best",
            "Best pair cosine",
        ),
        (
            "support_second",
            "Second-best pair cosine",
        ),
        (
            "support_third",
            "Third-best pair cosine",
        ),
        (
            "support_top5_mean",
            "Top-5 pair cosine mean",
        ),
        (
            "support_top10_mean",
            "Top-10 pair cosine mean",
        ),
        (
            "support_pair_ge_090",
            "Pair count >= 0.90",
        ),
        (
            "support_pair_ge_095",
            "Pair count >= 0.95",
        ),
        (
            "support_pair_ge_099",
            "Pair count >= 0.99",
        ),
        (
            "support_library_count",
            "Supporting library count",
        ),
        (
            "support_library_second",
            "Second-best library cosine",
        ),
        (
            "support_library_third",
            "Third-best library cosine",
        ),
        (
            "support_library_ge_090",
            "Library count >= 0.90",
        ),
        (
            "support_library_ge_095",
            "Library count >= 0.95",
        ),
        (
            "support_library_ge_099",
            "Library count >= 0.99",
        ),
    ]

    for column, title in metrics:
        print()
        print(
            title
        )

        print(
            "-" * len(title)
        )

        print_distribution(
            "Correct",
            correct,
            column,
        )

        print_distribution(
            "Incorrect",
            incorrect,
            column,
        )

    # ---------------------------------------------------------
    # The seven currently recoverable Top-25 misses.
    # ---------------------------------------------------------

    current_rank_numeric = pd.to_numeric(
        correct[
            "current_rank"
        ],
        errors="coerce",
    )

    recoverable = correct[
        current_rank_numeric
        > 25
    ].copy()

    print()
    print(
        "Correct Tier-3 rank-1 candidates currently outside Top 25"
    )

    print(
        "--------------------------------------------------------"
    )

    print(
        f"Count: "
        f"{len(recoverable):,}"
    )

    if len(recoverable):
        columns = [
            "inchikey14",
            "query_library",
            "current_rank",
            "support_best",
            "support_second",
            "support_third",
            "support_top5_mean",
            "support_pair_ge_095",
            "support_pair_ge_099",
            "support_library_count",
            "support_library_ge_095",
            "support_library_ge_099",
        ]

        print()

        print(
            recoverable[
                columns
            ]
            .sort_values(
                "current_rank"
            )
            .to_string(
                index=False
            )
        )

    # ---------------------------------------------------------
    # Show the strongest incorrect examples according to
    # repeated support. These are important counterexamples to
    # any future rescue rule.
    # ---------------------------------------------------------

    print()
    print(
        "Incorrect candidates with strongest repeated support"
    )

    print(
        "----------------------------------------------------"
    )

    strongest_incorrect = (
        incorrect
        .sort_values(
            [
                "support_top5_mean",
                "support_pair_ge_095",
                "support_library_ge_095",
            ],
            ascending=[
                False,
                False,
                False,
            ],
        )
        .head(30)
    )

    print(
        strongest_incorrect[
            [
                "inchikey14",
                "query_library",
                "current_rank",
                "support_best",
                "support_second",
                "support_third",
                "support_top5_mean",
                "support_pair_ge_095",
                "support_pair_ge_099",
                "support_library_count",
                "support_library_ge_095",
                "support_library_ge_099",
            ]
        ].to_string(
            index=False
        )
    )

    # ---------------------------------------------------------
    # Save complete diagnostic artifact.
    # ---------------------------------------------------------

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    analysis.to_parquet(
        OUTPUT_PATH,
        index=False,
    )

    print()
    print(
        "Saved:"
    )

    print(
        f"  {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()