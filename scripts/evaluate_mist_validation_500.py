from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem


ROOT = Path("data/processed/mist/validation_500")

MANIFEST_PATH = ROOT / "manifest.parquet"
MAPPING_PATH = ROOT / "mist_multiformula_mapping_processable.parquet"
CANDIDATES_PATH = ROOT / "mist_multiformula_candidates.parquet"
FP_DIR = ROOT / "mist_predictions" / "fp_preds"

OUTPUT_PATH = ROOT / "mist_validation_ranked_candidates.parquet"

MASS_PENALTY = 0.010


def morgan4096(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    fp = AllChem.GetMorganFingerprintAsBitVect(
        mol,
        radius=2,
        nBits=4096,
    )

    arr = np.zeros(4096, dtype=np.float32)
    for idx in fp.GetOnBits():
        arr[idx] = 1.0

    return arr


def cosine_similarity(a, b):
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def rr25(rank):
    if rank is None or rank > 25:
        return 0.0
    return 1.0 / rank


def main():
    manifest = pd.read_parquet(MANIFEST_PATH)
    mapping = pd.read_parquet(MAPPING_PATH)
    candidates = pd.read_parquet(CANDIDATES_PATH)

    pickle_files = sorted(FP_DIR.glob("*.p"))
    if len(pickle_files) != 1:
        raise RuntimeError(
            f"Expected exactly one prediction pickle, found {len(pickle_files)}"
        )

    with pickle_files[0].open("rb") as f:
        obj = pickle.load(f)

    names = list(obj["names"])
    preds = np.asarray(obj["preds"], dtype=np.float32)

    if len(names) != len(preds):
        raise ValueError("Prediction name/count mismatch")

    pred_by_hypothesis = {
        name: pred
        for name, pred in zip(names, preds)
    }

    expected_names = set(mapping["hypothesis_spec"])
    actual_names = set(pred_by_hypothesis)

    if expected_names != actual_names:
        raise ValueError(
            f"Prediction/mapping mismatch: "
            f"missing={len(expected_names - actual_names)}, "
            f"unexpected={len(actual_names - expected_names)}"
        )

    fp_cache = {}

    def get_fp(smiles):
        if smiles not in fp_cache:
            fp_cache[smiles] = morgan4096(smiles)
        return fp_cache[smiles]

    ranked_parts = []

    for query_key, query_candidates in candidates.groupby(
        "query_inchikey14",
        sort=False,
    ):
        scored_rows = []

        for row in query_candidates.itertuples(index=False):
            pred_fp = pred_by_hypothesis[row.hypothesis_spec]
            cand_fp = get_fp(row.smiles)

            if cand_fp is None:
                continue

            mist_cosine = cosine_similarity(pred_fp, cand_fp)

            final_score = (
                mist_cosine
                - MASS_PENALTY * abs(row.mass_error_ppm)
            )

            scored_rows.append(
                {
                    "query_inchikey14": query_key,
                    "hypothesis_spec": row.hypothesis_spec,
                    "candidate_formula": row.candidate_formula,
                    "candidate_inchikey14": row.candidate_inchikey14,
                    "is_truth": bool(row.is_truth),
                    "mist_cosine": mist_cosine,
                    "formula_mass_error_ppm": row.mass_error_ppm,
                    "final_score": final_score,
                }
            )

        if not scored_rows:
            continue

        q = pd.DataFrame(scored_rows)

        q = (
            q.sort_values(
                [
                    "final_score",
                    "candidate_inchikey14",
                ],
                ascending=[False, True],
                kind="stable",
            )
            .drop_duplicates(
                "candidate_inchikey14",
                keep="first",
            )
            .reset_index(drop=True)
        )

        q["rank"] = np.arange(len(q)) + 1
        ranked_parts.append(q)

    ranked = pd.concat(
        ranked_parts,
        ignore_index=True,
    )

    ranked.to_parquet(
        OUTPUT_PATH,
        index=False,
    )

    results = []

    for row in manifest.itertuples(index=False):
        query_key = row.query_inchikey14

        q = ranked[
            ranked["query_inchikey14"] == query_key
        ]

        truth = q[
            q["candidate_inchikey14"] == query_key
        ]

        rank = (
            None
            if truth.empty
            else int(truth["rank"].min())
        )

        results.append(
            {
                "query_inchikey14": query_key,
                "truth_rank": rank,
                "rr25": rr25(rank),
                "top1": rank == 1,
                "top5": rank is not None and rank <= 5,
                "top10": rank is not None and rank <= 10,
                "top25": rank is not None and rank <= 25,
            }
        )

    results = pd.DataFrame(results)

    print("Frozen MIST validation-500")
    print("--------------------------")
    print("Queries:", len(results))
    print(f"MRR@25: {results['rr25'].mean():.4f}")
    print(f"Top1: {results['top1'].mean() * 100:.2f}%")
    print(f"Top5: {results['top5'].mean() * 100:.2f}%")
    print(f"Top10: {results['top10'].mean() * 100:.2f}%")
    print(f"Top25: {results['top25'].mean() * 100:.2f}%")

    found = results["truth_rank"].notna()

    print(
        "Truth ranked:",
        int(found.sum()),
        "/",
        len(results),
    )

    if found.any():
        print("\nTruth rank distribution:")
        print(
            results.loc[found, "truth_rank"]
            .describe(
                percentiles=[
                    0.25,
                    0.5,
                    0.75,
                    0.9,
                    0.95,
                    0.99,
                ]
            )
            .to_string()
        )

        conditional = results.loc[found]

        print("\nConditional on truth being rankable:")
        print(
            f"MRR@25: "
            f"{conditional['rr25'].mean():.4f}"
        )
        print(
            f"Top1: "
            f"{conditional['top1'].mean() * 100:.2f}%"
        )
        print(
            f"Top25: "
            f"{conditional['top25'].mean() * 100:.2f}%"
        )

    print(f"\nWrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()