from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem


BENCHMARK_DIR = Path("data/processed/mist/benchmark_100")

PREDICTIONS_PATH = (
    BENCHMARK_DIR
    / "mist_predictions"
    / "fp_preds"
    / "fp_preds_casmi_benchmark.p"
)

CANDIDATES_PATH = BENCHMARK_DIR / "candidates.parquet"


def morgan4096(smiles: str) -> np.ndarray:
    """Generate the fingerprint used by MIST's `morgan4096` feature."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Could not parse SMILES: {smiles}")

    fingerprint = AllChem.GetMorganFingerprintAsBitVect(
        mol,
        radius=2,
        nBits=4096,
    )

    array = np.zeros((4096,), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fingerprint, array)

    return array


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denominator = np.linalg.norm(a) * np.linalg.norm(b)

    if denominator == 0:
        return 0.0

    return float(np.dot(a, b) / denominator)


def main() -> None:
    with open(PREDICTIONS_PATH, "rb") as f:
        predictions = pickle.load(f)

    names = predictions["names"]
    predicted_fps = np.asarray(predictions["preds"], dtype=np.float32)

    print(f"MIST predictions: {predicted_fps.shape}")

    if predicted_fps.shape[1] != 4096:
        raise ValueError(
            f"Expected 4096 fingerprint dimensions, "
            f"got {predicted_fps.shape[1]}"
        )

    # MIST name:
    #   casmi_IWVCMVBTMGNXQD
    #
    # Candidate query key:
    #   IWVCMVBTMGNXQD
    prediction_by_query = {
        name.removeprefix("casmi_"): fp
        for name, fp in zip(names, predicted_fps, strict=True)
    }

    candidates = pd.read_parquet(CANDIDATES_PATH)

    print(f"Candidate rows: {len(candidates):,}")
    print(
        "Candidate queries:",
        candidates["query_inchikey14"].nunique(),
    )

    benchmark_queries = set(prediction_by_query)
    candidate_queries = set(candidates["query_inchikey14"])

    missing_candidates = benchmark_queries - candidate_queries

    if missing_candidates:
        raise ValueError(
            f"{len(missing_candidates)} MIST queries have no candidates: "
            f"{sorted(missing_candidates)[:5]}"
        )

    # Restrict candidates to the 82 spectra actually processed by MIST.
    candidates = candidates[
        candidates["query_inchikey14"].isin(benchmark_queries)
    ].copy()

    print(f"Benchmark candidate rows: {len(candidates):,}")
    print(
        "Benchmark candidate queries:",
        candidates["query_inchikey14"].nunique(),
    )

    # Cache fingerprints because the same structure may occur in more than
    # one query candidate set.
    fingerprint_cache: dict[str, np.ndarray] = {}

    scores = []

    for row in candidates.itertuples(index=False):
        smiles = row.smiles

        if smiles not in fingerprint_cache:
            fingerprint_cache[smiles] = morgan4096(smiles)

        candidate_fp = fingerprint_cache[smiles]
        predicted_fp = prediction_by_query[row.query_inchikey14]

        score = cosine_similarity(predicted_fp, candidate_fp)
        scores.append(score)

    candidates["mist_cosine"] = scores

    # Deterministic tie-breaker. It should almost never matter for cosine
    # scores, but makes results reproducible.
    ranked = candidates.sort_values(
        [
            "query_inchikey14",
            "mist_cosine",
            "candidate_inchikey14",
        ],
        ascending=[True, False, True],
        kind="stable",
    ).copy()

    ranked["mist_rank"] = (
        ranked.groupby("query_inchikey14").cumcount() + 1
    )

    truth = ranked[ranked["is_truth"]].copy()

    truth_count_per_query = truth.groupby(
        "query_inchikey14"
    ).size()

    if not (truth_count_per_query == 1).all():
        raise ValueError(
            "Expected exactly one truth candidate per query"
        )

    ranks = truth["mist_rank"].to_numpy()

    reciprocal_rank_at_25 = np.where(
        ranks <= 25,
        1.0 / ranks,
        0.0,
    )

    mrr25 = reciprocal_rank_at_25.mean()
    top1 = np.mean(ranks <= 1)
    top5 = np.mean(ranks <= 5)
    top10 = np.mean(ranks <= 10)
    top25 = np.mean(ranks <= 25)

    print()
    print("MIST same-formula structure ranking")
    print("-----------------------------------")
    print(f"Queries evaluated: {len(ranks)}")
    print(f"MRR@25:           {mrr25:.4f}")
    print(f"Top-1 accuracy:   {top1:.2%}")
    print(f"Top-5 recall:     {top5:.2%}")
    print(f"Top-10 recall:    {top10:.2%}")
    print(f"Top-25 recall:    {top25:.2%}")

    print()
    print("Truth rank distribution")
    print("-----------------------")
    print(pd.Series(ranks).describe(
        percentiles=[0.25, 0.5, 0.75, 0.9]
    ))

    output_path = BENCHMARK_DIR / "mist_ranked_candidates.parquet"

    ranked.to_parquet(output_path, index=False)

    print()
    print(f"Saved ranked candidates to: {output_path}")


if __name__ == "__main__":
    main()