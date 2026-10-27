from __future__ import annotations

from pathlib import Path

import pandas as pd

from casmi26.pubchem.inference_asset import (
    load_pubchem_candidates_for_formulas,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent

ROOT = (
    PROJECT_ROOT
    / "data/processed/blind_pseudotest"
)

TRUTH_PATH = ROOT / "truth.parquet"
EVALUATION_PATH = ROOT / "evaluation.parquet"

COCONUT_PATH = (
    PROJECT_ROOT
    / "data/external/coconut/coconut_approved.parquet"
)

PUBCHEM_ROOT = (
    PROJECT_ROOT
    / "data/processed/pubchem_formula_index"
)

OUTPUT_PATH = (
    ROOT
    / "novelty_strata.parquet"
)

SUMMARY_PATH = (
    ROOT
    / "novelty_summary.csv"
)


def rr25(
    rank,
) -> float:
    if rank is None or pd.isna(rank):
        return 0.0

    rank = int(rank)

    if rank < 1 or rank > 25:
        return 0.0

    return 1.0 / rank


def summarize(
    frame: pd.DataFrame,
    *,
    group_column: str,
) -> pd.DataFrame:
    rows = []

    for value, group in frame.groupby(
        group_column,
        dropna=False,
        sort=False,
    ):
        ranks = group[
            "truth_rank"
        ]

        rows.append(
            {
                "dimension":
                    group_column,
                "stratum":
                    str(value),
                "queries":
                    len(group),
                "fraction":
                    len(group)
                    / len(frame),
                "mrr25":
                    ranks.map(
                        rr25
                    ).mean(),
                "top1":
                    ranks.eq(
                        1
                    ).mean(),
                "top5":
                    ranks.between(
                        1,
                        5,
                    ).mean(),
                "top10":
                    ranks.between(
                        1,
                        10,
                    ).mean(),
                "top25":
                    ranks.between(
                        1,
                        25,
                    ).mean(),
            }
        )

    return pd.DataFrame(
        rows
    )


def main() -> None:
    for path in [
        TRUTH_PATH,
        EVALUATION_PATH,
        COCONUT_PATH,
        PUBCHEM_ROOT
        / "formula_buckets.parquet",
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"Missing required artifact: {path}"
            )

    truth = pd.read_parquet(
        TRUTH_PATH
    ).copy()

    evaluation = pd.read_parquet(
        EVALUATION_PATH
    ).copy()

    required_truth = {
        "molecule_id",
        "inchikey14",
        "molecular_formula",
    }

    missing_truth = (
        required_truth
        - set(
            truth.columns
        )
    )

    if missing_truth:
        raise ValueError(
            "Truth file missing columns: "
            f"{sorted(missing_truth)}"
        )

    required_evaluation = {
        "molecule_id",
        "truth_rank",
        "outcome",
    }

    missing_evaluation = (
        required_evaluation
        - set(
            evaluation.columns
        )
    )

    if missing_evaluation:
        raise ValueError(
            "Evaluation file missing columns: "
            f"{sorted(missing_evaluation)}"
        )

    truth[
        "molecule_id"
    ] = (
        truth[
            "molecule_id"
        ]
        .astype(str)
    )

    truth[
        "inchikey14"
    ] = (
        truth[
            "inchikey14"
        ]
        .astype(str)
    )

    truth[
        "molecular_formula"
    ] = (
        truth[
            "molecular_formula"
        ]
        .astype(str)
    )

    evaluation[
        "molecule_id"
    ] = (
        evaluation[
            "molecule_id"
        ]
        .astype(str)
    )

    formulas = sorted(
        truth[
            "molecular_formula"
        ]
        .dropna()
        .astype(str)
        .unique()
    )

    print(
        "Blind queries:",
        f"{len(truth):,}",
    )

    print(
        "Unique truth formulas:",
        f"{len(formulas):,}",
    )

    coconut = pd.read_parquet(
        COCONUT_PATH,
        columns=[
            "inchikey14",
        ],
    )

    coconut_keys = set(
        coconut[
            "inchikey14"
        ]
        .dropna()
        .astype(str)
    )

    print(
        "Loading PubChem candidates for truth formulas..."
    )

    formula_buckets = pd.read_parquet(
        PUBCHEM_ROOT
        / "formula_buckets.parquet",
        columns=[
            "molecular_formula",
        ],
    )

    available_pubchem_formulas = set(
        formula_buckets[
            "molecular_formula"
        ]
        .dropna()
        .astype(str)
    )

    requested_pubchem_formulas = sorted(
        set(
            formulas
        )
        & available_pubchem_formulas
    )

    missing_pubchem_formulas = sorted(
        set(
            formulas
        )
        - available_pubchem_formulas
    )

    print(
        "Truth formulas present in PubChem asset:",
        f"{len(requested_pubchem_formulas):,}",
    )

    print(
        "Truth formulas absent from PubChem asset:",
        f"{len(missing_pubchem_formulas):,}",
    )

    if requested_pubchem_formulas:
        pubchem = (
            load_pubchem_candidates_for_formulas(
                PUBCHEM_ROOT,
                requested_pubchem_formulas,
            )
        )

        pubchem_pairs = set(
            zip(
                pubchem[
                    "molecular_formula"
                ].astype(str),
                pubchem[
                    "inchikey14"
                ].astype(str),
                strict=True,
            )
        )
    else:
        pubchem_pairs = set()

    annotated = truth.merge(
        evaluation[
            [
                "molecule_id",
                "truth_rank",
                "truth_formula_rank",
                "outcome",
                "rr25",
            ]
        ],
        on="molecule_id",
        how="left",
        validate="one_to_one",
    )

    annotated[
        "in_approved_coconut"
    ] = (
        annotated[
            "inchikey14"
        ]
        .isin(
            coconut_keys
        )
    )

    annotated[
        "pubchem_formula_present"
    ] = (
        annotated[
            "molecular_formula"
        ]
        .astype(str)
        .isin(
            available_pubchem_formulas
        )
    )

    annotated[
        "in_pubchem_asset"
    ] = [
        (
            str(
                row.molecular_formula
            ),
            str(
                row.inchikey14
            ),
        )
        in pubchem_pairs
        for row
        in annotated.itertuples(
            index=False
        )
    ]

    def database_stratum(
        row: pd.Series,
    ) -> str:
        in_coconut = bool(
            row[
                "in_approved_coconut"
            ]
        )

        in_pubchem = bool(
            row[
                "in_pubchem_asset"
            ]
        )

        pubchem_formula_present = bool(
            row[
                "pubchem_formula_present"
            ]
        )

        if in_coconut and in_pubchem:
            return "coconut+pubchem"

        if in_pubchem:
            return "pubchem_only"

        if in_coconut:
            return "coconut_only"

        if not pubchem_formula_present:
            return "pubchem_formula_absent"

        return "formula_present_truth_structure_absent"

    annotated[
        "database_stratum"
    ] = annotated.apply(
        database_stratum,
        axis=1,
    )

    annotated[
        "candidate_availability"
    ] = (
        annotated[
            "outcome"
        ]
        .map(
            {
                "success_top25":
                    "truth_reached_top25",
                "truth_present_but_ranked_below25":
                    "truth_present_after_top500",
                "truth_pruned_below_top500":
                    "truth_present_before_top500",
                "truth_structure_absent_candidate_universe":
                    "truth_absent_candidate_universe",
                "truth_formula_outside_top3":
                    "formula_gate_failure",
                "truth_formula_not_recovered":
                    "formula_recovery_failure",
                "fallback_retrieval_miss":
                    "fallback_branch",
            }
        )
        .fillna(
            "other"
        )
    )

    annotated.to_parquet(
        OUTPUT_PATH,
        index=False,
        compression="zstd",
    )

    summaries = [
        summarize(
            annotated,
            group_column=
                "database_stratum",
        ),
        summarize(
            annotated,
            group_column=
                "pubchem_formula_present",
        ),
        summarize(
            annotated,
            group_column=
                "in_pubchem_asset",
        ),
        summarize(
            annotated,
            group_column=
                "in_approved_coconut",
        ),
        summarize(
            annotated,
            group_column=
                "candidate_availability",
        ),
    ]

    summary = pd.concat(
        summaries,
        ignore_index=True,
    )

    summary.to_csv(
        SUMMARY_PATH,
        index=False,
    )

    print()
    print(
        "Novelty-stratified blind evaluation"
    )
    print(
        "=" * 92
    )

    for dimension in [
        "database_stratum",
        "pubchem_formula_present",
        "in_pubchem_asset",
        "in_approved_coconut",
        "candidate_availability",
    ]:
        frame = summary.loc[
            summary[
                "dimension"
            ]
            == dimension
        ].copy()

        print()
        print(
            dimension
        )
        print(
            "-" * len(
                dimension
            )
        )

        print(
            frame[
                [
                    "stratum",
                    "queries",
                    "fraction",
                    "mrr25",
                    "top1",
                    "top5",
                    "top10",
                    "top25",
                ]
            ]
            .to_string(
                index=False,
                formatters={
                    "fraction":
                        lambda value:
                            f"{value:.2%}",
                    "mrr25":
                        lambda value:
                            f"{value:.4f}",
                    "top1":
                        lambda value:
                            f"{value:.2%}",
                    "top5":
                        lambda value:
                            f"{value:.2%}",
                    "top10":
                        lambda value:
                            f"{value:.2%}",
                    "top25":
                        lambda value:
                            f"{value:.2%}",
                },
            )
        )

    print()
    print(
        "Database availability counts"
    )
    print(
        "----------------------------"
    )

    print(
        annotated[
            "database_stratum"
        ]
        .value_counts()
        .to_string()
    )

    print()
    print(
        "Wrote:",
        OUTPUT_PATH,
    )

    print(
        "Wrote:",
        SUMMARY_PATH,
    )


if __name__ == "__main__":
    main()
