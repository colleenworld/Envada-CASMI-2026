from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem


ROOT = Path("data/processed/mist/benchmark_100")

MAPPING_PATH = ROOT / "mist_multiformula_mapping.parquet"
CANDIDATES_PATH = ROOT / "mist_multiformula_candidates.parquet"

FP_DIR = ROOT / "mist_multiformula_predictions" / "fp_preds"

OUTPUT_PATH = (
    ROOT
    / "mist_multiformula_ranked_candidates.parquet"
)


def morgan4096(smiles: str) -> np.ndarray | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    fp = AllChem.GetMorganFingerprintAsBitVect(
        mol,
        radius=2,
        nBits=4096,
    )

    arr = np.zeros((4096,), dtype=np.float32)
    for idx in fp.GetOnBits():
        arr[idx] = 1.0

    return arr


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)

    if denom == 0:
        return 0.0

    return float(np.dot(a, b) / denom)


def reciprocal_rank_at_25(rank: int | None) -> float:
    if rank is None or rank > 25:
        return 0.0

    return 1.0 / rank


def main():
    mapping = pd.read_parquet(MAPPING_PATH)
    candidates = pd.read_parquet(CANDIDATES_PATH)

    pickle_files = sorted(FP_DIR.glob("*.p"))
    if len(pickle_files) != 1:
        raise RuntimeError(
            f"Expected one prediction pickle, found {len(pickle_files)}"
        )

    with pickle_files[0].open("rb") as f:
        preds_obj = pickle.load(f)

    names = list(preds_obj["names"])
    preds = np.asarray(preds_obj["preds"], dtype=np.float32)

    print("Predictions:", len(names), preds.shape)

    if len(names) != len(preds):
        raise ValueError("Prediction name/count mismatch")

    pred_by_hypothesis = {
        name: pred
        for name, pred in zip(names, preds)
    }

    mapping_names = set(mapping["hypothesis_spec"])
    prediction_names = set(pred_by_hypothesis)

    if mapping_names != prediction_names:
        print(
            "Missing predictions:",
            len(mapping_names - prediction_names),
        )
        print(
            "Unexpected predictions:",
            len(prediction_names - mapping_names),
        )
        raise ValueError(
            "Mapping/prediction hypothesis IDs do not match"
        )

    #
    # Build candidate-structure table indexed by
    # query and molecular formula.
    #
    # candidates.parquet was originally generated for the
    # true-formula benchmark, so candidate_formula is present
    # on every candidate structure row.
    #
    required = {
        "query_inchikey14",
        "candidate_inchikey14",
        "candidate_formula",
        "smiles",
        "is_truth",
    }

    missing = required - set(candidates.columns)
    if missing:
        raise ValueError(
            f"Candidate table missing columns: {sorted(missing)}"
        )

    #
    # Cache structure fingerprints because structures can appear
    # in more than one query.
    #
    fp_cache: dict[str, np.ndarray | None] = {}

    def get_candidate_fp(smiles: str):
        if smiles not in fp_cache:
            fp_cache[smiles] = morgan4096(smiles)
        return fp_cache[smiles]

    ranked_rows = []

    for query_key, query_mapping in mapping.groupby(
        "query_inchikey14",
        sort=False,
    ):
        query_candidates = candidates[
            candidates["query_inchikey14"] == query_key
        ]

        if query_candidates.empty:
            continue

        scored = []

        for hypothesis in query_mapping.itertuples(index=False):
            pred_fp = pred_by_hypothesis[
                hypothesis.hypothesis_spec
            ]

            formula_candidates = query_candidates[
                query_candidates["candidate_formula"]
                == hypothesis.candidate_formula
            ]

            for cand in formula_candidates.itertuples(index=False):
                candidate_fp = get_candidate_fp(cand.smiles)

                if candidate_fp is None:
                    continue

                score = cosine_similarity(
                    pred_fp,
                    candidate_fp,
                )

                scored.append(
                    {
                        "query_inchikey14": query_key,
                        "hypothesis_spec":
                            hypothesis.hypothesis_spec,
                        "candidate_formula":
                            hypothesis.candidate_formula,
                        "candidate_inchikey14":
                            cand.candidate_inchikey14,
                        "is_truth": bool(cand.is_truth),
                        "mist_cosine": score,
                        "formula_mass_error_ppm":
                            hypothesis.mass_error_ppm,
                    }
                )

        if not scored:
            continue

        scored_df = pd.DataFrame(scored)

        #
        # A structure should normally occur under only one molecular
        # formula. Still, collapse defensively in case source data
        # contains duplicates.
        #
        scored_df = (
            scored_df
            .sort_values(
                [
                    "mist_cosine",
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

        scored_df["rank"] = (
            np.arange(len(scored_df)) + 1
        )

        ranked_rows.append(scored_df)

    if ranked_rows:
        ranked = pd.concat(
            ranked_rows,
            ignore_index=True,
        )
    else:
        ranked = pd.DataFrame()

    ranked.to_parquet(
        OUTPUT_PATH,
        index=False,
    )

    #
    # Evaluate over all 82 original benchmark queries, including
    # queries with no formula candidates.
    #
    manifest = pd.read_parquet(
        ROOT / "manifest.parquet"
    )

    results = []

    for row in manifest.itertuples(index=False):
        query_key = row.query_inchikey14

        query_ranked = ranked[
            ranked["query_inchikey14"] == query_key
        ]

        truth_rows = query_ranked[
            query_ranked["candidate_inchikey14"]
            == query_key
        ]

        if truth_rows.empty:
            truth_rank = None
        else:
            truth_rank = int(
                truth_rows["rank"].min()
            )

        results.append(
            {
                "query_inchikey14": query_key,
                "truth_rank": truth_rank,
                "rr25": reciprocal_rank_at_25(
                    truth_rank
                ),
                "top1":
                    truth_rank == 1,
                "top5":
                    truth_rank is not None
                    and truth_rank <= 5,
                "top10":
                    truth_rank is not None
                    and truth_rank <= 10,
                "top25":
                    truth_rank is not None
                    and truth_rank <= 25,
            }
        )

    results = pd.DataFrame(results)

    print("\nMulti-formula MIST ranking")
    print("--------------------------")
    print("Queries:", len(results))

    print(
        f"MRR@25: {results['rr25'].mean():.4f}"
    )
    print(
        f"Top1: {results['top1'].mean() * 100:.2f}%"
    )
    print(
        f"Top5: {results['top5'].mean() * 100:.2f}%"
    )
    print(
        f"Top10: {results['top10'].mean() * 100:.2f}%"
    )
    print(
        f"Top25: {results['top25'].mean() * 100:.2f}%"
    )

    found = results[
        results["truth_rank"].notna()
    ]

    print(
        "Truth ranked:",
        len(found),
        "/",
        len(results),
    )

    if not found.empty:
        print("\nTruth rank distribution:")
        print(
            found["truth_rank"]
            .describe(
                percentiles=[
                    0.25,
                    0.5,
                    0.75,
                    0.9,
                    0.95,
                ]
            )
            .to_string()
        )

    print(f"\nWrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()