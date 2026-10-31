from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import inchi
from rdkit.Chem.MolStandardize import rdMolStandardize


DEFAULT_ROOT = Path(
    "data/processed/blind_pseudotest"
)

PROJECT_TEST_ROOT = Path(
    "data/processed/test"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
    )
    parser.add_argument(
        "--submission",
        type=Path,
        default=Path("submission.csv"),
    )
    return parser.parse_args()


TAUTOMER_ENUMERATOR = (
    rdMolStandardize.TautomerEnumerator()
)


def smiles_to_key14(
    value: str,
) -> str | None:
    mol = Chem.MolFromSmiles(
        str(value)
    )

    if mol is None:
        return None

    try:
        mol = (
            TAUTOMER_ENUMERATOR
            .Canonicalize(
                mol
            )
        )
    except Exception:
        return None

    key = inchi.MolToInchiKey(
        mol
    )

    if not key:
        return None

    return str(key).split("-")[0]


def rr25(
    rank: float | int | None,
) -> float:
    if rank is None or pd.isna(rank):
        return 0.0

    rank = int(rank)

    if rank < 1 or rank > 25:
        return 0.0

    return 1.0 / rank


def main() -> None:
    args = parse_args()

    truth_path = (
        args.root
        / "truth.parquet"
    )

    for path in [
        truth_path,
        args.submission,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"Missing required artifact: {path}"
            )

    truth = pd.read_parquet(
        truth_path
    ).copy()

    truth["molecule_id"] = (
        truth["molecule_id"]
        .astype(str)
    )

    truth["inchikey14"] = (
        truth["inchikey14"]
        .astype(str)
    )

    truth["canonical_truth_key14"] = [
        smiles_to_key14(
            value
        )
        for value
        in truth[
            "normalized_smiles"
        ]
    ]

    if truth[
        "canonical_truth_key14"
    ].isna().any():
        raise RuntimeError(
            "Some truth SMILES could not be tautomer-canonicalized"
        )

    submission = pd.read_csv(
        args.submission
    )

    submission["molecule_id"] = (
        submission["molecule_id"]
        .astype(str)
    )

    expected_ids = set(
        truth["molecule_id"]
    )
    actual_ids = set(
        submission["molecule_id"]
    )

    if expected_ids != actual_ids:
        raise RuntimeError(
            "Submission molecule IDs do not match blind truth. "
            f"missing={len(expected_ids - actual_ids)}, "
            f"unexpected={len(actual_ids - expected_ids)}"
        )

    rows = []

    for row in submission.itertuples(
        index=False
    ):
        candidates = [
            item
            for item in str(
                row.smiles
            ).split(";")
            if item
        ]

        keys = [
            smiles_to_key14(
                smiles
            )
            for smiles in candidates
        ]

        rows.extend(
            {
                "molecule_id":
                    str(row.molecule_id),
                "submission_rank":
                    rank,
                "predicted_inchikey14":
                    key,
            }
            for rank, key
            in enumerate(
                keys,
                start=1,
            )
        )

    predicted = pd.DataFrame(
        rows
    )

    scored = truth[
        [
            "molecule_id",
            "inchikey14",
            "canonical_truth_key14",
            "molecular_formula",
        ]
    ].copy()

    first_truth = (
        predicted.merge(
            truth[
                [
                    "molecule_id",
                    "canonical_truth_key14",
                ]
            ],
            left_on=[
                "molecule_id",
                "predicted_inchikey14",
            ],
            right_on=[
                "molecule_id",
                "canonical_truth_key14",
            ],
            how="inner",
        )
        .groupby(
            "molecule_id"
        )[
            "submission_rank"
        ]
        .min()
        .rename(
            "truth_rank"
        )
    )

    scored = scored.merge(
        first_truth,
        on="molecule_id",
        how="left",
        validate="one_to_one",
    )

    hypotheses_path = (
        PROJECT_TEST_ROOT
        / "multispectrum_formula_hypotheses.parquet"
    )

    prior_path = (
        PROJECT_TEST_ROOT
        / "guard5_neighbor_top500_prior.parquet"
    )

    top500_path = (
        PROJECT_TEST_ROOT
        / "guard5_neighbor_top500.parquet"
    )

    positive_path = (
        PROJECT_TEST_ROOT
        / "positive_contrastive_ranked_candidates.parquet"
    )

    fallback_path = (
        PROJECT_TEST_ROOT
        / "negative_retrieval_ranked_candidates.parquet"
    )

    hypotheses = (
        pd.read_parquet(
            hypotheses_path
        )
        if hypotheses_path.exists()
        else pd.DataFrame()
    )

    prior = (
        pd.read_parquet(
            prior_path
        )
        if prior_path.exists()
        else pd.DataFrame()
    )

    top500 = (
        pd.read_parquet(
            top500_path
        )
        if top500_path.exists()
        else pd.DataFrame()
    )

    positive = (
        pd.read_parquet(
            positive_path
        )
        if positive_path.exists()
        else pd.DataFrame()
    )

    fallback = (
        pd.read_parquet(
            fallback_path
        )
        if fallback_path.exists()
        else pd.DataFrame()
    )

    for frame in [
        hypotheses,
        prior,
        top500,
        positive,
        fallback,
    ]:
        if (
            not frame.empty
            and "molecule_id"
            in frame.columns
        ):
            frame["molecule_id"] = (
                frame["molecule_id"]
                .astype(str)
            )

        if (
            not frame.empty
            and "candidate_inchikey14"
            in frame.columns
        ):
            frame["candidate_inchikey14"] = (
                frame[
                    "candidate_inchikey14"
                ]
                .astype(str)
            )

    formula_rank_lookup: dict[
        tuple[str, str],
        int,
    ] = {}

    if not hypotheses.empty:
        h = hypotheses.copy()

        h["candidate_formula"] = (
            h["candidate_formula"]
            .astype(str)
        )

        h = (
            h.sort_values(
                [
                    "molecule_id",
                    "min_abs_mass_error_ppm",
                    "candidate_formula",
                ],
                ascending=[
                    True,
                    True,
                    True,
                ],
                kind="stable",
            )
            .drop_duplicates(
                subset=[
                    "molecule_id",
                    "candidate_formula",
                ]
            )
        )

        h[
            "formula_rank"
        ] = (
            h.groupby(
                "molecule_id"
            )
            .cumcount()
            + 1
        )

        formula_rank_lookup = {
            (
                str(row.molecule_id),
                str(row.candidate_formula),
            ):
                int(row.formula_rank)
            for row in h.itertuples(
                index=False
            )
        }

    prior_pairs = set()

    if not prior.empty:
        prior_pairs = set(
            zip(
                prior[
                    "molecule_id"
                ].astype(str),
                prior[
                    "candidate_inchikey14"
                ].astype(str),
                strict=True,
            )
        )

    top500_pairs = set()

    if not top500.empty:
        top500_pairs = set(
            zip(
                top500[
                    "molecule_id"
                ].astype(str),
                top500[
                    "candidate_inchikey14"
                ].astype(str),
                strict=True,
            )
        )

    positive_ids = (
        set(
            positive[
                "molecule_id"
            ].astype(str)
        )
        if not positive.empty
        else set()
    )

    fallback_ids = (
        set(
            fallback[
                "molecule_id"
            ].astype(str)
        )
        if not fallback.empty
        else set()
    )

    miss_reasons = []
    formula_ranks = []

    for row in scored.itertuples(
        index=False
    ):
        molecule_id = str(
            row.molecule_id
        )
        truth_key = str(
            row.inchikey14
        )
        truth_formula = str(
            row.molecular_formula
        )

        formula_rank = (
            formula_rank_lookup.get(
                (
                    molecule_id,
                    truth_formula,
                )
            )
        )

        formula_ranks.append(
            formula_rank
        )

        if (
            pd.notna(
                row.truth_rank
            )
            and int(
                row.truth_rank
            )
            <= 25
        ):
            miss_reasons.append(
                "success_top25"
            )
            continue

        if molecule_id in fallback_ids:
            miss_reasons.append(
                "fallback_retrieval_miss"
            )
            continue

        if formula_rank is None:
            miss_reasons.append(
                "truth_formula_not_recovered"
            )
            continue

        if formula_rank > 3:
            miss_reasons.append(
                "truth_formula_outside_top3"
            )
            continue

        pair = (
            molecule_id,
            truth_key,
        )

        if pair not in prior_pairs:
            miss_reasons.append(
                "truth_structure_absent_candidate_universe"
            )
            continue

        if pair not in top500_pairs:
            miss_reasons.append(
                "truth_pruned_below_top500"
            )
            continue

        if molecule_id in positive_ids:
            miss_reasons.append(
                "truth_present_but_ranked_below25"
            )
            continue

        miss_reasons.append(
            "unclassified_positive_miss"
        )

    scored[
        "truth_formula_rank"
    ] = formula_ranks

    scored[
        "outcome"
    ] = miss_reasons

    scored[
        "rr25"
    ] = scored[
        "truth_rank"
    ].map(
        rr25
    )

    summary = (
        scored[
            "outcome"
        ]
        .value_counts()
        .rename_axis(
            "outcome"
        )
        .reset_index(
            name="queries"
        )
    )

    summary[
        "fraction"
    ] = (
        summary[
            "queries"
        ]
        / len(scored)
    )

    args.root.mkdir(
        parents=True,
        exist_ok=True,
    )

    scored_path = (
        args.root
        / "evaluation.parquet"
    )

    summary_path = (
        args.root
        / "failure_summary.csv"
    )

    scored.to_parquet(
        scored_path,
        index=False,
        compression="zstd",
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    print()
    print("Blind pseudo-test evaluation")
    print("=" * 76)
    print(
        "Queries:",
        f"{len(scored):,}",
    )
    print(
        "MRR@25:",
        f"{scored['rr25'].mean():.4f}",
    )
    print(
        "Top-1:",
        f"{scored['truth_rank'].eq(1).mean():.2%}",
    )
    print(
        "Top-5:",
        f"{scored['truth_rank'].between(1,5).mean():.2%}",
    )
    print(
        "Top-10:",
        f"{scored['truth_rank'].between(1,10).mean():.2%}",
    )
    print(
        "Top-25:",
        f"{scored['truth_rank'].between(1,25).mean():.2%}",
    )
    print()
    print(
        summary.to_string(
            index=False,
            formatters={
                "fraction":
                    lambda value:
                        f"{value:.2%}",
            },
        )
    )
    print()
    print(
        "Wrote:",
        scored_path,
    )
    print(
        "Wrote:",
        summary_path,
    )


if __name__ == "__main__":
    main()
