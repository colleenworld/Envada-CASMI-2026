from pathlib import Path

import numpy as np
import pandas as pd

from casmi26.chemistry.adducts import neutral_mass
from casmi26.retrieval.index import load_retrieval_index


INDEX_DIR = Path(
    "data/processed/retrieval_index"
)

MANIFEST_PATH = Path(
    "data/processed/splits/retrieval_dev.parquet"
)

FEATURES_PATH = Path(
    "data/processed/results/"
    "retrieval_dev_baseline3_ranking_features.parquet"
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
    "retrieval_dev_spectral_rank1_mass_distance.parquet"
)


def infer_neutral_masses(
    precursor_mz: np.ndarray,
    adducts: np.ndarray,
) -> np.ndarray:
    masses = np.full(
        len(precursor_mz),
        np.nan,
        dtype=np.float64,
    )

    for i, (mz, adduct) in enumerate(
        zip(
            precursor_mz,
            adducts,
            strict=True,
        )
    ):
        if not np.isfinite(mz):
            continue

        mass = neutral_mass(
            float(mz),
            str(adduct),
        )

        if mass is not None:
            masses[i] = mass

    return masses


def build_reference_mask(
    metadata: pd.DataFrame,
    manifest: pd.DataFrame,
) -> np.ndarray:
    """
    Recreate the frozen retrieval-dev reference pool.

    For every benchmark structure, remove spectra belonging to that
    structure's assigned query library. All other spectra remain in
    the reference pool.
    """
    reference_mask = np.ones(
        len(metadata),
        dtype=bool,
    )

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


def minimum_mass_delta(
    query_masses: np.ndarray,
    reference_masses: np.ndarray,
) -> float | None:
    """
    Calculate the minimum absolute neutral-mass difference between
    any query mass and any candidate-reference mass.
    """
    query_masses = np.asarray(
        query_masses,
        dtype=np.float64,
    )

    reference_masses = np.asarray(
        reference_masses,
        dtype=np.float64,
    )

    query_masses = query_masses[
        np.isfinite(query_masses)
    ]

    reference_masses = reference_masses[
        np.isfinite(reference_masses)
    ]

    if (
        len(query_masses) == 0
        or len(reference_masses) == 0
    ):
        return None

    deltas = np.abs(
        query_masses[:, None]
        - reference_masses[None, :]
    )

    return float(
        np.min(deltas)
    )


def percentile(
    values: pd.Series,
    q: float,
) -> float:
    values = pd.to_numeric(
        values,
        errors="coerce",
    )

    values = values[
        np.isfinite(values)
    ]

    if len(values) == 0:
        return float("nan")

    return float(
        np.quantile(
            values,
            q,
        )
    )


def print_distribution(
    label: str,
    values: pd.Series,
) -> None:
    values = pd.to_numeric(
        values,
        errors="coerce",
    )

    finite = values[
        np.isfinite(values)
    ]

    minimum = (
        float(finite.min())
        if len(finite)
        else float("nan")
    )

    maximum = (
        float(finite.max())
        if len(finite)
        else float("nan")
    )

    print(
        f"{label:<12} "
        f"n={len(finite):>3} "
        f"median={percentile(finite, 0.50):>10.6f} "
        f"p25={percentile(finite, 0.25):>10.6f} "
        f"p75={percentile(finite, 0.75):>10.6f} "
        f"p90={percentile(finite, 0.90):>10.6f} "
        f"min={minimum:>10.6f} "
        f"max={maximum:>10.6f}"
    )


def main() -> None:
    print(
        "Loading retrieval artifacts..."
    )

    metadata, _ = load_retrieval_index(
        INDEX_DIR
    )

    manifest = pd.read_parquet(
        MANIFEST_PATH
    )

    features = pd.read_parquet(
        FEATURES_PATH
    )

    rescue = pd.read_parquet(
        RESCUE_PATH
    )

    candidate_map = pd.read_parquet(
        CANDIDATE_MAP_PATH
    )

    # ---------------------------------------------------------
    # Validate the metadata schema that this diagnostic depends on.
    # ---------------------------------------------------------

    required_metadata_columns = {
        "inchikey14",
        "ingest_lib",
        "ionization_mode",
        "adduct",
        "precursor_mz",
    }

    missing_metadata_columns = (
        required_metadata_columns
        - set(metadata.columns)
    )

    if missing_metadata_columns:
        raise RuntimeError(
            "Retrieval metadata is missing columns: "
            f"{sorted(missing_metadata_columns)}"
        )

    # ---------------------------------------------------------
    # Identify the population under investigation:
    #
    #   spectral rank 1
    #   AND existing Baseline 3c source tier 3
    #
    # We already measured this population as:
    #
    #   459 total
    #    61 correct
    #   398 incorrect
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
            "Expected 61 correct Tier-3 "
            "spectral-rank-1 candidates, "
            f"found {truth_count}."
        )

    # ---------------------------------------------------------
    # Reconstruct the exact frozen retrieval-dev reference pool.
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
    # Extract the index metadata arrays.
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

    precursor_values = pd.to_numeric(
        metadata["precursor_mz"],
        errors="coerce",
    ).to_numpy(
        dtype=np.float64
    )

    adduct_values = (
        metadata["adduct"]
        .astype(str)
        .to_numpy()
    )

    # ---------------------------------------------------------
    # Infer neutral masses for every indexed spectrum.
    # ---------------------------------------------------------

    print(
        "Inferring neutral masses..."
    )

    neutral_masses = infer_neutral_masses(
        precursor_mz=precursor_values,
        adducts=adduct_values,
    )

    finite_mass_count = int(
        np.isfinite(
            neutral_masses
        ).sum()
    )

    print(
        f"Spectra with usable neutral mass: "
        f"{finite_mass_count:,} / "
        f"{len(neutral_masses):,}"
    )

    # ---------------------------------------------------------
    # Load the authoritative frozen candidate-ID namespace.
    #
    # Do NOT reconstruct candidate IDs from the current feature
    # rows. The ranking-feature artifact contains only candidates
    # participating in the 2,000 hybrid candidate sets, whereas
    # the frozen candidate map contains the complete namespace.
    # ---------------------------------------------------------

    required_candidate_map_columns = {
        "candidate_id",
        "inchikey14",
    }

    missing_candidate_map_columns = (
        required_candidate_map_columns
        - set(candidate_map.columns)
    )

    if missing_candidate_map_columns:
        raise RuntimeError(
            "Candidate map is missing columns: "
            f"{sorted(missing_candidate_map_columns)}"
        )

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

    expected_candidate_ids = np.arange(
        len(candidate_map),
        dtype=np.int64,
    )

    actual_candidate_ids = (
        candidate_map[
            "candidate_id"
        ]
        .to_numpy(
            dtype=np.int64
        )
    )

    if not np.array_equal(
        actual_candidate_ids,
        expected_candidate_ids,
    ):
        raise RuntimeError(
            "Candidate IDs are not contiguous "
            "from 0 through N-1."
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
    # Map every frozen reference row into that candidate namespace.
    # ---------------------------------------------------------

    reference_keys = structure_values[
        reference_indices
    ]

    missing_reference_keys = sorted(
        set(reference_keys)
        - set(candidate_id_by_key)
    )

    if missing_reference_keys:
        raise RuntimeError(
            "Reference pool contains structures "
            "missing from the frozen candidate map. "
            f"Example: {missing_reference_keys[:5]}"
        )

    reference_candidate_ids = np.asarray(
        [
            candidate_id_by_key[key]
            for key in reference_keys
        ],
        dtype=np.int64,
    )

    print(
        f"Reference rows mapped: "
        f"{len(reference_candidate_ids):,}"
    )

    # ---------------------------------------------------------
    # Validate the frozen candidate map against every candidate
    # appearing in the ranking feature artifact.
    # ---------------------------------------------------------

    required_feature_columns = {
        "inchikey14",
        "query_library",
        "candidate_id",
        "candidate_inchikey14",
        "is_truth",
        "in_same_mass",
        "in_opposite_mass",
        "in_spectral_top25",
        "spectral_rank",
        "same_polarity_cosine",
    }

    missing_feature_columns = (
        required_feature_columns
        - set(features.columns)
    )

    if missing_feature_columns:
        raise RuntimeError(
            "Ranking feature artifact is missing columns: "
            f"{sorted(missing_feature_columns)}"
        )

    feature_pairs = (
        features[
            [
                "candidate_id",
                "candidate_inchikey14",
            ]
        ]
        .drop_duplicates()
        .copy()
    )

    feature_candidate_ids = (
        feature_pairs[
            "candidate_id"
        ]
        .to_numpy(
            dtype=np.int64
        )
    )

    if not np.all(
        (
            feature_candidate_ids
            >= 0
        )
        & (
            feature_candidate_ids
            < len(candidate_keys)
        )
    ):
        raise RuntimeError(
            "Ranking features contain candidate IDs "
            "outside the frozen candidate map."
        )

    mapped_feature_keys = candidate_keys[
        feature_candidate_ids
    ]

    feature_keys = (
        feature_pairs[
            "candidate_inchikey14"
        ]
        .astype(str)
        .to_numpy()
    )

    if not np.array_equal(
        mapped_feature_keys,
        feature_keys,
    ):
        mismatch_mask = (
            mapped_feature_keys
            != feature_keys
        )

        first_bad = int(
            np.flatnonzero(
                mismatch_mask
            )[0]
        )

        raise RuntimeError(
            "Frozen candidate map does not match "
            "the ranking feature artifact. "
            f"candidate_id="
            f"{feature_candidate_ids[first_bad]}, "
            f"map={mapped_feature_keys[first_bad]}, "
            f"features={feature_keys[first_bad]}"
        )

    print(
        "Candidate-map validation: PASS"
    )

    # ---------------------------------------------------------
    # Precompute the frozen reference rows belonging to each
    # candidate ID.
    #
    # Sort once by candidate ID so that looking up the spectra
    # belonging to one candidate is inexpensive.
    # ---------------------------------------------------------

    order = np.argsort(
        reference_candidate_ids,
        kind="stable",
    )

    sorted_candidate_ids = (
        reference_candidate_ids[
            order
        ]
    )

    sorted_reference_indices = (
        reference_indices[
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

    candidate_reference_rows: dict[
        int,
        np.ndarray,
    ] = {}

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

    # ---------------------------------------------------------
    # Locate the query spectra for each retrieval-dev query.
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

    if len(
        query_rows_by_key
    ) != len(manifest):
        raise RuntimeError(
            "Query-row lookup size does not "
            "match the retrieval-dev manifest."
        )

    # ---------------------------------------------------------
    # Calculate neutral-mass distances for all 459 Tier-3
    # spectral-rank-1 candidates.
    #
    # We calculate:
    #
    #   1. same-polarity minimum neutral-mass delta
    #   2. opposite-polarity minimum neutral-mass delta
    #   3. best delta across either polarity
    #
    # This is descriptive only. We are not selecting a threshold.
    # ---------------------------------------------------------

    print(
        "Calculating Tier-3 spectral-rank-1 "
        "mass distances..."
    )

    rows: list[
        dict[str, object]
    ] = []

    for row in population.itertuples(
        index=False
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

        key = (
            structure_key,
            query_library,
        )

        query_rows = query_rows_by_key[
            key
        ]

        candidate_rows = (
            candidate_reference_rows.get(
                candidate_id
            )
        )

        if candidate_rows is None:
            raise RuntimeError(
                "No frozen reference rows found "
                f"for candidate {candidate_id}."
            )

        query_mass = neutral_masses[
            query_rows
        ]

        candidate_mass = neutral_masses[
            candidate_rows
        ]

        query_modes = mode_values[
            query_rows
        ]

        candidate_modes = mode_values[
            candidate_rows
        ]

        same_deltas: list[
            float
        ] = []

        opposite_deltas: list[
            float
        ] = []

        # Compare each query mode independently. This avoids
        # accidentally pairing a query mass from one polarity with
        # the wrong candidate polarity when calculating the
        # same-polarity result.
        for query_mode in np.unique(
            query_modes
        ):
            mode_query_mass = query_mass[
                query_modes
                == query_mode
            ]

            same_reference_mass = (
                candidate_mass[
                    candidate_modes
                    == query_mode
                ]
            )

            opposite_reference_mass = (
                candidate_mass[
                    candidate_modes
                    != query_mode
                ]
            )

            same_delta = minimum_mass_delta(
                mode_query_mass,
                same_reference_mass,
            )

            if same_delta is not None:
                same_deltas.append(
                    same_delta
                )

            opposite_delta = minimum_mass_delta(
                mode_query_mass,
                opposite_reference_mass,
            )

            if opposite_delta is not None:
                opposite_deltas.append(
                    opposite_delta
                )

        if same_deltas:
            same_delta = float(
                min(same_deltas)
            )
        else:
            same_delta = None

        if opposite_deltas:
            opposite_delta = float(
                min(opposite_deltas)
            )
        else:
            opposite_delta = None

        finite_deltas = [
            value
            for value in (
                same_delta,
                opposite_delta,
            )
            if value is not None
            and np.isfinite(value)
        ]

        if finite_deltas:
            best_delta = float(
                min(finite_deltas)
            )
        else:
            best_delta = None

        rows.append(
            {
                "inchikey14": structure_key,
                "query_library": query_library,
                "rank1_candidate_id": candidate_id,
                "rank1_candidate_inchikey14": str(
                    candidate_keys[
                        candidate_id
                    ]
                ),
                "rank1_is_truth": bool(
                    row.rank1_is_truth
                ),
                "current_rank": row.current_rank,
                "rank1_cosine": row.rank1_cosine,
                "same_polarity_mass_delta": (
                    same_delta
                ),
                "opposite_polarity_mass_delta": (
                    opposite_delta
                ),
                "best_mass_delta": (
                    best_delta
                ),
                "query_mass_count": int(
                    np.isfinite(
                        query_mass
                    ).sum()
                ),
                "candidate_mass_count": int(
                    np.isfinite(
                        candidate_mass
                    ).sum()
                ),
            }
        )

    analysis = pd.DataFrame(
        rows
    )

    # ---------------------------------------------------------
    # Integrity checks.
    # ---------------------------------------------------------

    if len(analysis) != 459:
        raise RuntimeError(
            "Expected 459 result rows, "
            f"found {len(analysis)}."
        )

    output_truth_count = int(
        analysis[
            "rank1_is_truth"
        ].sum()
    )

    if output_truth_count != 61:
        raise RuntimeError(
            "Truth count changed unexpectedly: "
            f"{output_truth_count}."
        )

    # Verify that candidate IDs still map to the expected
    # candidate structures.
    expected_candidate_keys = candidate_keys[
        analysis[
            "rank1_candidate_id"
        ].to_numpy(
            dtype=np.int64
        )
    ]

    if not np.array_equal(
        expected_candidate_keys,
        analysis[
            "rank1_candidate_inchikey14"
        ]
        .astype(str)
        .to_numpy(),
    ):
        raise RuntimeError(
            "Output candidate mapping validation failed."
        )

    print(
        "Mass-distance population validation: PASS"
    )

    # ---------------------------------------------------------
    # Separate correct and incorrect spectral-rank-1 candidates.
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

    # ---------------------------------------------------------
    # Distribution comparison.
    # ---------------------------------------------------------

    print()
    print(
        "Tier-3 spectral-rank-1 mass-distance analysis"
    )

    print(
        "--------------------------------------------"
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

    for column, title in [
        (
            "same_polarity_mass_delta",
            "Same-polarity neutral-mass delta",
        ),
        (
            "opposite_polarity_mass_delta",
            "Opposite-polarity neutral-mass delta",
        ),
        (
            "best_mass_delta",
            "Best neutral-mass delta",
        ),
    ]:
        print()
        print(
            title
        )

        print(
            "-" * len(title)
        )

        print_distribution(
            "Correct",
            correct[
                column
            ],
        )

        print_distribution(
            "Incorrect",
            incorrect[
                column
            ],
        )

    # ---------------------------------------------------------
    # Descriptive cutoff table.
    #
    # These values are NOT being used to select a model
    # threshold. They simply make the distributions easier to
    # understand.
    # ---------------------------------------------------------

    print()
    print(
        "Best-mass-delta descriptive cutoffs"
    )

    print(
        "-----------------------------------"
    )

    print(
        f"{'delta <=':<12}"
        f"{'correct':>10}"
        f"{'incorrect':>12}"
        f"{'precision':>12}"
        f"{'truth recall':>14}"
    )

    best_correct = pd.to_numeric(
        correct[
            "best_mass_delta"
        ],
        errors="coerce",
    )

    best_incorrect = pd.to_numeric(
        incorrect[
            "best_mass_delta"
        ],
        errors="coerce",
    )

    for threshold in [
        0.02,
        0.05,
        0.10,
        0.25,
        0.50,
        1.00,
        2.00,
        5.00,
        10.00,
    ]:
        correct_count = int(
            (
                best_correct
                <= threshold
            ).sum()
        )

        incorrect_count = int(
            (
                best_incorrect
                <= threshold
            ).sum()
        )

        selected = (
            correct_count
            + incorrect_count
        )

        if selected:
            precision = (
                correct_count
                / selected
            )
        else:
            precision = float(
                "nan"
            )

        recall = (
            correct_count
            / len(correct)
        )

        print(
            f"{threshold:<12.2f}"
            f"{correct_count:>10}"
            f"{incorrect_count:>12}"
            f"{precision:>11.2%}"
            f"{recall:>13.2%}"
        )

    # ---------------------------------------------------------
    # Inspect the seven known recoverable Top-25 misses.
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
        print()

        print(
            recoverable[
                [
                    "inchikey14",
                    "query_library",
                    "current_rank",
                    "rank1_cosine",
                    "same_polarity_mass_delta",
                    "opposite_polarity_mass_delta",
                    "best_mass_delta",
                ]
            ]
            .sort_values(
                "current_rank"
            )
            .to_string(
                index=False
            )
        )

    # ---------------------------------------------------------
    # Show the incorrect Tier-3 rank-1 candidates closest in
    # neutral mass. These are the dangerous counterexamples to
    # any future mass-distance rescue rule.
    # ---------------------------------------------------------

    print()
    print(
        "Closest incorrect Tier-3 rank-1 candidates by mass"
    )

    print(
        "-------------------------------------------------"
    )

    closest_incorrect = (
        incorrect[
            incorrect[
                "best_mass_delta"
            ].notna()
        ]
        .sort_values(
            [
                "best_mass_delta",
                "rank1_cosine",
            ],
            ascending=[
                True,
                False,
            ],
        )
        .head(30)
    )

    if len(
        closest_incorrect
    ):
        print(
            closest_incorrect[
                [
                    "inchikey14",
                    "query_library",
                    "current_rank",
                    "rank1_cosine",
                    "same_polarity_mass_delta",
                    "opposite_polarity_mass_delta",
                    "best_mass_delta",
                ]
            ]
            .to_string(
                index=False
            )
        )

    # ---------------------------------------------------------
    # Missing-mass counts are also useful. A missing delta and a
    # genuinely large mass difference are different situations.
    # ---------------------------------------------------------

    print()
    print(
        "Mass-distance availability"
    )

    print(
        "--------------------------"
    )

    for label, frame in [
        (
            "Correct",
            correct,
        ),
        (
            "Incorrect",
            incorrect,
        ),
    ]:
        same_available = int(
            frame[
                "same_polarity_mass_delta"
            ].notna().sum()
        )

        opposite_available = int(
            frame[
                "opposite_polarity_mass_delta"
            ].notna().sum()
        )

        best_available = int(
            frame[
                "best_mass_delta"
            ].notna().sum()
        )

        print(
            f"{label:<10} "
            f"same={same_available:>3}/{len(frame):<3} "
            f"opposite={opposite_available:>3}/{len(frame):<3} "
            f"best={best_available:>3}/{len(frame):<3}"
        )

    # ---------------------------------------------------------
    # Save the complete 459-query diagnostic artifact.
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