from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from casmi26.chemistry.adducts import neutral_mass
from casmi26.retrieval.aggregation import (
    aggregate_max_by_candidate,
    merge_candidate_scores,
)
from casmi26.retrieval.filtering import neutral_mass_mask
from casmi26.spectra.binning import bin_spectrum


PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAIN_PATH = PROJECT_ROOT / "data" / "raw" / "train.parquet"

DOMAIN_LIBRARY = "enveda-np-examples"

BIN_WIDTH = 1.0
MAX_MZ = 1500.0
NEUTRAL_MASS_TOLERANCE_DA = 0.01
PARQUET_BATCH_SIZE = 10_000


def bin_spectra(df: pd.DataFrame) -> np.ndarray:
    """
    Convert spectra into L2-normalized binned vectors.
    """
    vectors = np.empty(
        (len(df), int(MAX_MZ / BIN_WIDTH)),
        dtype=np.float32,
    )

    for i, (mzs, intensities) in enumerate(
        zip(
            df["ms2_mzs"],
            df["ms2_normalized_intensities"],
            strict=True,
        )
    ):
        vectors[i] = bin_spectrum(
            mzs,
            intensities,
            max_mz=MAX_MZ,
            bin_width=BIN_WIDTH,
        )

    return vectors


def infer_neutral_masses(
    precursor_mzs: np.ndarray,
    adducts: np.ndarray,
) -> np.ndarray:
    """
    Infer neutral molecular masses from precursor m/z and adduct.

    Unsupported adducts are represented by NaN so they naturally
    fail the neutral-mass filter.
    """
    masses = np.empty(
        len(precursor_mzs),
        dtype=np.float64,
    )

    for i, (precursor_mz, adduct) in enumerate(
        zip(
            precursor_mzs,
            adducts,
            strict=True,
        )
    ):
        mass = neutral_mass(
            float(precursor_mz),
            str(adduct),
        )

        masses[i] = (
            np.nan
            if mass is None
            else mass
        )

    return masses


def load_queries() -> pd.DataFrame:
    """
    Load the enveda-np-examples spectra used as domain queries.
    """
    return (
        pd.read_parquet(
            TRAIN_PATH,
            columns=[
                "inchikey14",
                "ingest_lib",
                "ionization_mode",
                "adduct",
                "precursor_mz",
                "ms2_mzs",
                "ms2_normalized_intensities",
            ],
            filters=[
                (
                    "ingest_lib",
                    "==",
                    DOMAIN_LIBRARY,
                ),
            ],
        )
        .reset_index(drop=True)
    )


def search_mode(
    queries: pd.DataFrame,
    mode: str,
) -> dict[str, dict[str, float]]:
    """
    Search queries belonging to one ionization mode.

    Baseline 3b differs from Baseline 3a in the filtered path:

        Baseline 3a:
            same ionization mode
            + any supported adduct
            + neutral mass ±0.01 Da

        Baseline 3b:
            either ionization mode
            + any supported adduct
            + neutral mass ±0.01 Da

    The fallback deliberately remains same-ionization-mode cosine
    retrieval. This keeps the fallback behavior comparable with 3a
    and isolates cross-polarity retrieval as the experimental change.
    """
    mode_queries = (
        queries.loc[
            queries["ionization_mode"] == mode
        ]
        .reset_index(drop=True)
    )

    print(f"\n{mode}")
    print("-" * len(mode))
    print(
        f"query spectra: {len(mode_queries):,}"
    )

    if mode_queries.empty:
        return {}

    print("Binning query spectra...")

    query_vectors = bin_spectra(
        mode_queries
    )

    query_keys = (
        mode_queries["inchikey14"]
        .astype(str)
        .to_numpy()
    )

    query_adducts = (
        mode_queries["adduct"]
        .astype(str)
        .to_numpy()
    )

    query_precursor_mzs = (
        mode_queries["precursor_mz"]
        .to_numpy(dtype=np.float64)
    )

    query_neutral_masses = infer_neutral_masses(
        query_precursor_mzs,
        query_adducts,
    )

    supported_queries = np.isfinite(
        query_neutral_masses
    )

    print(
        "Queries with supported adduct: "
        f"{np.sum(supported_queries):,}"
    )
    print(
        "Queries with unsupported adduct: "
        f"{np.sum(~supported_queries):,}"
    )

    # Scores are maintained separately for every query spectrum.
    #
    # We cannot decide whether a query needs the fallback until the
    # complete reference library has been scanned.
    fallback_scores: list[
        dict[str, float]
    ] = [
        {}
        for _ in range(len(mode_queries))
    ]

    filtered_scores: list[
        dict[str, float]
    ] = [
        {}
        for _ in range(len(mode_queries))
    ]

    parquet = pq.ParquetFile(
        TRAIN_PATH
    )

    processed = 0

    print(
        "Streaming cross-polarity reference library..."
    )

    for batch in parquet.iter_batches(
        columns=[
            "inchikey14",
            "ingest_lib",
            "ionization_mode",
            "adduct",
            "precursor_mz",
            "ms2_mzs",
            "ms2_normalized_intensities",
        ],
        batch_size=PARQUET_BATCH_SIZE,
    ):
        chunk = batch.to_pandas()

        # ----------------------------------------------------------
        # Remove the query library FIRST.
        #
        # It is important that every mask and candidate array below
        # is constructed from this exact same filtered DataFrame.
        # ----------------------------------------------------------

        chunk = (
            chunk.loc[
                chunk["ingest_lib"]
                != DOMAIN_LIBRARY
            ]
            .reset_index(drop=True)
        )

        if chunk.empty:
            continue

        # ----------------------------------------------------------
        # Same-mode mask
        #
        # Baseline 3b allows either polarity for neutral-mass
        # retrieval, but the ordinary fallback remains restricted
        # to the query's ionization mode.
        # ----------------------------------------------------------

        same_mode_mask = (
            chunk["ionization_mode"]
            .astype(str)
            .to_numpy()
            == mode
        )

        candidate_vectors = bin_spectra(
            chunk
        )

        candidate_keys = (
            chunk["inchikey14"]
            .astype(str)
            .to_numpy()
        )

        candidate_adducts = (
            chunk["adduct"]
            .astype(str)
            .to_numpy()
        )

        candidate_precursor_mzs = (
            chunk["precursor_mz"]
            .to_numpy(dtype=np.float64)
        )

        candidate_neutral_masses = (
            infer_neutral_masses(
                candidate_precursor_mzs,
                candidate_adducts,
            )
        )

        # ----------------------------------------------------------
        # Defensive alignment checks
        #
        # These protect us against accidentally constructing a mask
        # before filtering the candidate DataFrame.
        # ----------------------------------------------------------

        assert len(chunk) == len(
            candidate_vectors
        )

        assert len(chunk) == len(
            candidate_keys
        )

        assert len(chunk) == len(
            candidate_adducts
        )

        assert len(chunk) == len(
            candidate_precursor_mzs
        )

        assert len(chunk) == len(
            candidate_neutral_masses
        )

        assert len(chunk) == len(
            same_mode_mask
        )

        # Every query spectrum is compared against every candidate
        # spectrum in this chunk, regardless of polarity.
        #
        # Because bin_spectrum() L2-normalizes its output, the dot
        # product is cosine similarity.
        similarities = (
            query_vectors
            @ candidate_vectors.T
        )

        for (
            query_index,
            spectrum_scores,
        ) in enumerate(similarities):
            assert len(
                spectrum_scores
            ) == len(candidate_keys)

            assert len(
                spectrum_scores
            ) == len(same_mode_mask)

            # ------------------------------------------------------
            # Fallback path
            #
            # Ordinary cosine retrieval restricted to the same
            # ionization mode as the query.
            # ------------------------------------------------------

            if np.any(same_mode_mask):
                (
                    fallback_candidate_keys,
                    fallback_candidate_scores,
                ) = aggregate_max_by_candidate(
                    spectrum_scores[
                        same_mode_mask
                    ],
                    candidate_keys[
                        same_mode_mask
                    ],
                )

                merge_candidate_scores(
                    fallback_scores[
                        query_index
                    ],
                    fallback_candidate_keys,
                    fallback_candidate_scores,
                )

            # ------------------------------------------------------
            # Neutral-mass path
            #
            # Any supported adduct and either ionization mode may
            # participate.
            # ------------------------------------------------------

            query_neutral_mass = (
                query_neutral_masses[
                    query_index
                ]
            )

            if not np.isfinite(
                query_neutral_mass
            ):
                continue

            mask = neutral_mass_mask(
                query_neutral_mass=(
                    query_neutral_mass
                ),
                candidate_neutral_masses=(
                    candidate_neutral_masses
                ),
                tolerance_da=(
                    NEUTRAL_MASS_TOLERANCE_DA
                ),
            )

            if not np.any(mask):
                continue

            (
                filtered_candidate_keys,
                filtered_candidate_scores,
            ) = aggregate_max_by_candidate(
                spectrum_scores[mask],
                candidate_keys[mask],
            )

            merge_candidate_scores(
                filtered_scores[
                    query_index
                ],
                filtered_candidate_keys,
                filtered_candidate_scores,
            )

        processed += len(chunk)

        if (
            processed % 100_000
            < len(chunk)
        ):
            print(
                f"  processed "
                f"{processed:,} "
                f"reference spectra"
            )

    # --------------------------------------------------------------
    # Select filtered or fallback result independently for every
    # query spectrum.
    #
    # We then merge all spectra belonging to the same query molecule
    # using maximum similarity.
    # --------------------------------------------------------------

    molecule_scores: dict[
        str,
        dict[str, float],
    ] = {}

    filtered_query_count = 0
    fallback_query_count = 0

    for (
        query_index,
        query_key,
    ) in enumerate(query_keys):
        if filtered_scores[
            query_index
        ]:
            selected_scores = (
                filtered_scores[
                    query_index
                ]
            )

            filtered_query_count += 1

        else:
            selected_scores = (
                fallback_scores[
                    query_index
                ]
            )

            fallback_query_count += 1

        molecule_result = (
            molecule_scores.setdefault(
                query_key,
                {},
            )
        )

        for (
            candidate_key,
            score,
        ) in selected_scores.items():
            molecule_result[
                candidate_key
            ] = max(
                molecule_result.get(
                    candidate_key,
                    -np.inf,
                ),
                score,
            )

    print(
        f"Filtered queries: "
        f"{filtered_query_count:,}"
    )
    print(
        f"Fallback queries: "
        f"{fallback_query_count:,}"
    )
    print(
        f"Finished {mode}: "
        f"{processed:,} "
        f"reference spectra"
    )

    return molecule_scores


def merge_mode_results(
    combined_scores: dict[
        str,
        dict[str, float],
    ],
    mode_scores: dict[
        str,
        dict[str, float],
    ],
) -> None:
    """
    Merge results from positive and negative query spectra belonging
    to the same query molecule.

    Maximum spectral similarity is retained for each candidate.
    """
    for (
        query_key,
        candidate_scores,
    ) in mode_scores.items():
        combined = (
            combined_scores.setdefault(
                query_key,
                {},
            )
        )

        for (
            candidate_key,
            score,
        ) in candidate_scores.items():
            combined[
                candidate_key
            ] = max(
                combined.get(
                    candidate_key,
                    -np.inf,
                ),
                score,
            )


def calculate_metrics(
    molecule_scores: dict[
        str,
        dict[str, float],
    ],
) -> np.ndarray:
    """
    Calculate one reciprocal rank per query molecule.

    Candidates below rank 25 receive zero, giving MRR@25.
    """
    reciprocal_ranks = []

    for (
        truth_key,
        candidate_scores,
    ) in molecule_scores.items():
        ranked = sorted(
            candidate_scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )

        ranked_keys = [
            candidate_key
            for candidate_key, _ in ranked
        ]

        try:
            rank = (
                ranked_keys.index(
                    truth_key
                )
                + 1
            )
        except ValueError:
            rank = None

        if (
            rank is not None
            and rank <= 25
        ):
            reciprocal_rank = (
                1.0 / rank
            )
        else:
            reciprocal_rank = 0.0

        reciprocal_ranks.append(
            reciprocal_rank
        )

    return np.asarray(
        reciprocal_ranks,
        dtype=np.float64,
    )


def main() -> None:
    print(
        "Baseline 3b — neutral-mass "
        "cross-polarity filtering"
    )
    print(
        "any ionization mode, "
        "any supported adduct"
    )
    print(
        f"neutral-mass tolerance: "
        f"±{NEUTRAL_MASS_TOLERANCE_DA} Da"
    )
    print()

    print(
        "Loading domain queries..."
    )

    queries = load_queries()

    print(
        f"Query spectra:   "
        f"{len(queries):,}"
    )
    print(
        f"Query molecules: "
        f"{queries['inchikey14'].nunique():,}"
    )

    combined_scores: dict[
        str,
        dict[str, float],
    ] = {}

    # We still process query spectra by their own ionization mode.
    # The difference in 3b is that the neutral-mass reference search
    # is allowed to use candidate spectra from either polarity.
    for mode in sorted(
        queries[
            "ionization_mode"
        ].unique()
    ):
        mode_scores = search_mode(
            queries,
            mode,
        )

        merge_mode_results(
            combined_scores,
            mode_scores,
        )

    scores = calculate_metrics(
        combined_scores
    )

    print("\nResults")
    print("=======")
    print(
        f"Molecules evaluated: "
        f"{len(scores):,}"
    )
    print(
        f"MRR@25:              "
        f"{np.mean(scores):.4f}"
    )
    print(
        f"Top-1 accuracy:       "
        f"{np.mean(scores == 1.0):.1%}"
    )
    print(
        f"Top-25 recall:        "
        f"{np.mean(scores > 0):.1%}"
    )


if __name__ == "__main__":
    main()