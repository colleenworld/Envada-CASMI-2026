from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from casmi26.retrieval.aggregation import (
    aggregate_max_by_candidate,
    merge_candidate_scores,
)
from casmi26.retrieval.filtering import (
    precursor_adduct_mask,
)
from casmi26.spectra.binning import bin_spectrum


PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAIN_PATH = PROJECT_ROOT / "data" / "raw" / "train.parquet"

DOMAIN_LIBRARY = "enveda-np-examples"

BIN_WIDTH = 1.0
MAX_MZ = 1500.0
PRECURSOR_TOLERANCE_DA = 0.01
PARQUET_BATCH_SIZE = 10_000


def bin_spectra(df: pd.DataFrame) -> np.ndarray:
    """
    Convert a DataFrame of spectra into L2-normalized binned vectors.
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


def load_queries() -> pd.DataFrame:
    """
    Load the enveda-np-examples spectra that act as our domain
    validation queries.
    """
    df = pd.read_parquet(
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
            ("ingest_lib", "==", DOMAIN_LIBRARY),
        ],
    )

    return df.reset_index(drop=True)


def search_mode(
    queries: pd.DataFrame,
    mode: str,
) -> dict[str, dict[str, float]]:
    """
    Search all non-domain reference spectra for queries from one
    ionization mode.

    Two sets of scores are maintained for each query spectrum:

    1. fallback_scores:
       All reference spectra having the same ionization mode.

    2. filtered_scores:
       Reference spectra having the same ionization mode, the same
       adduct, and precursor m/z within PRECURSOR_TOLERANCE_DA.

    After the entire reference library has been scanned, filtered
    scores are used when any filtered candidates were found.
    Otherwise the ordinary same-ionization scores are used.
    """
    mode_queries = (
        queries.loc[
            queries["ionization_mode"] == mode
        ]
        .reset_index(drop=True)
    )

    print(f"\n{mode}")
    print("-" * len(mode))
    print(f"query spectra: {len(mode_queries):,}")

    if mode_queries.empty:
        return {}

    print("Binning query spectra...")
    query_vectors = bin_spectra(mode_queries)

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

    # Scores are kept per query spectrum while streaming the reference
    # library. We cannot decide whether fallback is needed until every
    # reference chunk has been examined.
    fallback_scores: list[dict[str, float]] = [
        {}
        for _ in range(len(mode_queries))
    ]

    filtered_scores: list[dict[str, float]] = [
        {}
        for _ in range(len(mode_queries))
    ]

    parquet = pq.ParquetFile(TRAIN_PATH)

    processed = 0

    print("Streaming reference library...")

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

        # enveda-np-examples supplies the query spectra, so those
        # spectra must not also be available in the reference library.
        #
        # We also restrict the reference chunk to the same ionization
        # mode as the current query group.
        chunk = (
            chunk.loc[
                (chunk["ingest_lib"] != DOMAIN_LIBRARY)
                & (chunk["ionization_mode"] == mode)
            ]
            .reset_index(drop=True)
        )

        if chunk.empty:
            continue

        candidate_vectors = bin_spectra(chunk)

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

        # Matrix shape:
        #
        #     query spectra × candidate spectra
        #
        # Since bin_spectrum L2-normalizes the vectors, the dot
        # product is cosine similarity.
        similarities = (
            query_vectors @ candidate_vectors.T
        )

        for query_index, spectrum_scores in enumerate(
            similarities
        ):
            # ------------------------------------------------------
            # Fallback search
            # ------------------------------------------------------
            #
            # Every candidate in this chunk has already been
            # restricted to the same ionization mode.

            candidate_molecule_keys, scores = (
                aggregate_max_by_candidate(
                    spectrum_scores,
                    candidate_keys,
                )
            )

            merge_candidate_scores(
                fallback_scores[query_index],
                candidate_molecule_keys,
                scores,
            )

            # ------------------------------------------------------
            # Precursor/adduct-filtered search
            # ------------------------------------------------------

            mask = precursor_adduct_mask(
                query_adduct=query_adducts[query_index],
                query_precursor_mz=(
                    query_precursor_mzs[query_index]
                ),
                candidate_adducts=candidate_adducts,
                candidate_precursor_mzs=(
                    candidate_precursor_mzs
                ),
                tolerance_da=PRECURSOR_TOLERANCE_DA,
            )

            if np.any(mask):
                (
                    filtered_candidate_keys,
                    filtered_candidate_scores,
                ) = aggregate_max_by_candidate(
                    spectrum_scores[mask],
                    candidate_keys[mask],
                )

                merge_candidate_scores(
                    filtered_scores[query_index],
                    filtered_candidate_keys,
                    filtered_candidate_scores,
                )

        processed += len(chunk)

        if processed % 100_000 < len(chunk):
            print(
                f"  processed "
                f"{processed:,} reference spectra"
            )

    # --------------------------------------------------------------
    # Select filtered or fallback result for each query spectrum,
    # then aggregate spectra belonging to the same query molecule.
    # --------------------------------------------------------------

    molecule_scores: dict[
        str,
        dict[str, float],
    ] = {}

    filtered_query_count = 0
    fallback_query_count = 0

    for query_index, query_key in enumerate(query_keys):
        if filtered_scores[query_index]:
            selected_scores = filtered_scores[query_index]
            filtered_query_count += 1
        else:
            selected_scores = fallback_scores[query_index]
            fallback_query_count += 1

        molecule_result = molecule_scores.setdefault(
            query_key,
            {},
        )

        for candidate_key, score in selected_scores.items():
            molecule_result[candidate_key] = max(
                molecule_result.get(
                    candidate_key,
                    -np.inf,
                ),
                score,
            )

    print(
        f"Filtered queries: {filtered_query_count:,}"
    )
    print(
        f"Fallback queries: {fallback_query_count:,}"
    )
    print(
        f"Finished {mode}: "
        f"{processed:,} reference spectra"
    )

    return molecule_scores


def merge_mode_results(
    combined_scores: dict[str, dict[str, float]],
    mode_scores: dict[str, dict[str, float]],
) -> None:
    """
    Merge positive- and negative-ionization results for each query
    molecule using maximum similarity.
    """
    for query_key, candidate_scores in (
        mode_scores.items()
    ):
        combined = combined_scores.setdefault(
            query_key,
            {},
        )

        for candidate_key, score in (
            candidate_scores.items()
        ):
            combined[candidate_key] = max(
                combined.get(
                    candidate_key,
                    -np.inf,
                ),
                score,
            )


def calculate_metrics(
    molecule_scores: dict[str, dict[str, float]],
) -> np.ndarray:
    """
    Calculate one reciprocal rank per query molecule.

    Predictions below rank 25 receive zero, matching MRR@25.
    """
    reciprocal_ranks = []

    for truth_key, candidate_scores in (
        molecule_scores.items()
    ):
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
            rank = ranked_keys.index(truth_key) + 1
        except ValueError:
            rank = None

        if rank is not None and rank <= 25:
            reciprocal_rank = 1.0 / rank
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
    print("Loading domain queries...")

    queries = load_queries()

    print(
        f"Query spectra:   {len(queries):,}"
    )
    print(
        f"Query molecules: "
        f"{queries['inchikey14'].nunique():,}"
    )

    combined_scores: dict[
        str,
        dict[str, float],
    ] = {}

    for mode in sorted(
        queries["ionization_mode"].unique()
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