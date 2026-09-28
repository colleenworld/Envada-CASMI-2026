from pathlib import Path

import numpy as np
import pandas as pd

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
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    candidates = (
        candidates[candidates["ionization_mode"] == mode]
        .reset_index(drop=True)
    )

    queries = (
        queries[queries["ionization_mode"] == mode]
        .reset_index(drop=True)
    )

    print(f"\n{mode}")
    print("-" * len(mode))
    print(f"candidate spectra: {len(candidates):,}")
    print(f"query spectra:     {len(queries):,}")

    if candidates.empty or queries.empty:
        return []

    print("Binning candidate spectra...")
    candidate_vectors = bin_spectra(candidates)

    print("Binning query spectra...")
    query_vectors = bin_spectra(queries)

    candidate_keys = candidates["inchikey14"].to_numpy()

    # Build an integer molecule index so we can efficiently aggregate
    # spectrum similarities into candidate-molecule similarities.
    unique_candidate_keys, candidate_group = np.unique(
        candidate_keys,
        return_inverse=True,
    )

    molecule_scores: dict[str, np.ndarray] = {}

    print("Searching...")

    # Work in batches to avoid constructing the complete
    # query × candidate similarity matrix.
    batch_size = 32

    for start in range(0, len(queries), batch_size):
        end = min(start + batch_size, len(queries))

        similarities = (
            query_vectors[start:end] @ candidate_vectors.T
        )

        for row_offset, spectrum_scores in enumerate(similarities):
            query_index = start + row_offset
            query_key = queries.iloc[query_index]["inchikey14"]

            # Maximum spectrum similarity for each candidate molecule.
            scores = np.full(
                len(unique_candidate_keys),
                -np.inf,
                dtype=np.float32,
            )

            np.maximum.at(
                scores,
                candidate_group,
                spectrum_scores,
            )

            if query_key not in molecule_scores:
                molecule_scores[query_key] = scores
            else:
                # Maximum across all query spectra for this molecule.
                molecule_scores[query_key] = np.maximum(
                    molecule_scores[query_key],
                    scores,
                )

    reciprocal_ranks = []

    return unique_candidate_keys, molecule_scores


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

    queries = df[
        df["ingest_lib"] == DOMAIN_LIBRARY
    ].copy()

    candidates = df[
        df["inchikey14"].isin(domain_keys)
        & (df["ingest_lib"] != DOMAIN_LIBRARY)
    ].copy()

    print(f"Candidate spectra: {len(candidates):,}")
    print(f"Query spectra:     {len(queries):,}")
    print(
        f"Query molecules:   "
        f"{queries['inchikey14'].nunique():,}"
    )

    all_scores = []

    combined_scores: dict[str, dict[str, float]] = {}

    for mode in sorted(queries["ionization_mode"].unique()):
        candidate_keys, mode_scores = evaluate_mode(
            candidates,
            queries,
            mode,
        )

        for query_key, scores in mode_scores.items():
            molecule_candidates = combined_scores.setdefault(
                query_key,
                {},
            )

            for candidate_key, score in zip(
                    candidate_keys,
                    scores,
                    strict=True,
            ):
                previous = molecule_candidates.get(
                    candidate_key,
                    -np.inf,
                )

                molecule_candidates[candidate_key] = max(
                    previous,
                    float(score),
                )

    reciprocal_ranks = []

    for truth_key, candidate_scores in combined_scores.items():
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

    scores = np.asarray(reciprocal_ranks)

    print("\nResults")
    print("=======")
    print(f"Molecules evaluated: {len(scores):,}")
    print(f"MRR@25:              {np.mean(scores):.4f}")
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