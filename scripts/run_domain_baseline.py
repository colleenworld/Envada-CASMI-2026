from pathlib import Path

import numpy as np
import pandas as pd

from casmi26.retrieval.aggregation import (
    aggregate_max_by_candidate,
    merge_candidate_scores,
)
from casmi26.spectra.binning import bin_spectrum


PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAIN_PATH = PROJECT_ROOT / "data" / "raw" / "train.parquet"

DOMAIN_LIBRARY = "enveda-np-examples"

BIN_WIDTH = 1.0
MAX_MZ = 1500.0


def bin_spectra(df: pd.DataFrame) -> np.ndarray:
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


def evaluate_mode(
    candidates: pd.DataFrame,
    queries: pd.DataFrame,
    mode: str,
) -> dict[str, dict[str, float]]:
    # Ionization mode is a search constraint, not an evaluation unit.
    mode_candidates = (
        candidates.loc[
            candidates["ionization_mode"] == mode
        ]
        .reset_index(drop=True)
    )

    mode_queries = (
        queries.loc[
            queries["ionization_mode"] == mode
        ]
        .reset_index(drop=True)
    )

    print(f"\n{mode}")
    print("-" * len(mode))
    print(
        f"candidate spectra: "
        f"{len(mode_candidates):,}"
    )
    print(
        f"query spectra:     "
        f"{len(mode_queries):,}"
    )

    if mode_candidates.empty or mode_queries.empty:
        return {}

    print("Binning candidate spectra...")
    candidate_vectors = bin_spectra(mode_candidates)

    print("Binning query spectra...")
    query_vectors = bin_spectra(mode_queries)

    candidate_keys = (
        mode_candidates["inchikey14"]
        .astype(str)
        .to_numpy()
    )

    molecule_scores: dict[str, dict[str, float]] = {}

    print("Searching...")

    batch_size = 32

    for start in range(0, len(mode_queries), batch_size):
        end = min(
            start + batch_size,
            len(mode_queries),
        )

        similarities = (
            query_vectors[start:end]
            @ candidate_vectors.T
        )

        for row_offset, spectrum_scores in enumerate(
            similarities
        ):
            query_index = start + row_offset

            query_key = str(
                mode_queries.iloc[
                    query_index
                ]["inchikey14"]
            )

            (
                candidate_molecule_keys,
                scores,
            ) = aggregate_max_by_candidate(
                spectrum_scores,
                candidate_keys,
            )

            query_scores = molecule_scores.setdefault(
                query_key,
                {},
            )

            merge_candidate_scores(
                query_scores,
                candidate_molecule_keys,
                scores,
            )

    return molecule_scores


def merge_mode_results(
    combined_scores: dict[str, dict[str, float]],
    mode_scores: dict[str, dict[str, float]],
) -> None:
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
                combined.get(candidate_key, -np.inf),
                score,
            )


def calculate_metrics(
    molecule_scores: dict[str, dict[str, float]],
) -> np.ndarray:
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

        reciprocal_ranks.append(reciprocal_rank)

    return np.asarray(
        reciprocal_ranks,
        dtype=np.float64,
    )


def main() -> None:
    print("Loading relevant columns...")

    df = pd.read_parquet(
        TRAIN_PATH,
        columns=[
            "inchikey14",
            "ingest_lib",
            "ionization_mode",
            "ms2_mzs",
            "ms2_normalized_intensities",
        ],
    )

    domain_keys = set(
        df.loc[
            df["ingest_lib"] == DOMAIN_LIBRARY,
            "inchikey14",
        ]
    )

    queries = df.loc[
        df["ingest_lib"] == DOMAIN_LIBRARY
    ].copy()

    candidates = df.loc[
        df["inchikey14"].isin(domain_keys)
        & (df["ingest_lib"] != DOMAIN_LIBRARY)
    ].copy()

    print(
        f"Candidate spectra: {len(candidates):,}"
    )
    print(
        f"Query spectra:     {len(queries):,}"
    )
    print(
        f"Query molecules:   "
        f"{queries['inchikey14'].nunique():,}"
    )

    combined_scores: dict[
        str,
        dict[str, float],
    ] = {}

    for mode in sorted(
        queries["ionization_mode"].unique()
    ):
        mode_scores = evaluate_mode(
            candidates,
            queries,
            mode,
        )

        merge_mode_results(
            combined_scores,
            mode_scores,
        )

    scores = calculate_metrics(combined_scores)

    print("\nResults")
    print("=======")
    print(
        f"Molecules evaluated: {len(scores):,}"
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