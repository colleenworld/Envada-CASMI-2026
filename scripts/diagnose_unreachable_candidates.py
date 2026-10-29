from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent

BLIND_ROOT = (
    PROJECT_ROOT
    / "data/processed/blind_pseudotest"
)

UNION_PATH = (
    BLIND_ROOT
    / "candidate_source_union_recall.parquet"
)

TRUTH_PATH = (
    BLIND_ROOT
    / "truth.parquet"
)

MSBUDDY_PATH = (
    BLIND_ROOT
    / "msbuddy_formula_gate.parquet"
)

PRODUCTION_PATH = (
    PROJECT_ROOT
    / "data/processed/test/"
      "multispectrum_formula_hypotheses.parquet"
)

OUTPUT_PATH = (
    BLIND_ROOT
    / "unreachable_candidate_diagnostic.parquet"
)

SUMMARY_PATH = (
    BLIND_ROOT
    / "unreachable_candidate_diagnostic_summary.csv"
)


def formula_topk(
    frame: pd.DataFrame,
    rank_column: str,
    cutoff: int = 3,
) -> set[tuple[str, str]]:
    return set(
        zip(
            frame.loc[
                frame[rank_column] <= cutoff,
                "molecule_id",
            ].astype(str),
            frame.loc[
                frame[rank_column] <= cutoff,
                "candidate_formula",
            ].astype(str),
            strict=True,
        )
    )


def production_top3(
    frame: pd.DataFrame,
) -> set[tuple[str, str]]:
    work = frame[
        [
            "molecule_id",
            "candidate_formula",
            "min_abs_mass_error_ppm",
        ]
    ].copy()

    work["molecule_id"] = (
        work["molecule_id"]
        .astype(str)
    )

    work["candidate_formula"] = (
        work["candidate_formula"]
        .astype(str)
    )

    work = (
        work.sort_values(
            [
                "molecule_id",
                "min_abs_mass_error_ppm",
                "candidate_formula",
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

    work["rank"] = (
        work.groupby(
            "molecule_id",
            sort=False,
        )
        .cumcount()
        + 1
    )

    return formula_topk(
        work,
        "rank",
        3,
    )


def main() -> None:
    for path in [
        UNION_PATH,
        TRUTH_PATH,
        MSBUDDY_PATH,
        PRODUCTION_PATH,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"Missing required artifact: {path}"
            )

    union = pd.read_parquet(
        UNION_PATH
    ).copy()

    truth = pd.read_parquet(
        TRUTH_PATH
    ).copy()

    union["molecule_id"] = (
        union["molecule_id"]
        .astype(str)
    )

    truth["molecule_id"] = (
        truth["molecule_id"]
        .astype(str)
    )

    unreachable = union.loc[
        ~union["any_source"]
    ].copy()

    unreachable = unreachable.merge(
        truth,
        on=[
            "molecule_id",
            "molecular_formula",
            "inchikey14",
        ],
        how="left",
        validate="one_to_one",
        suffixes=(
            "",
            "_truth",
        ),
    )

    msbuddy = pd.read_parquet(
        MSBUDDY_PATH
    ).copy()

    msbuddy["molecule_id"] = (
        msbuddy["molecule_id"]
        .astype(str)
    )

    msbuddy["candidate_formula"] = (
        msbuddy["candidate_formula"]
        .astype(str)
    )

    msbuddy_top3 = formula_topk(
        msbuddy,
        "molecule_rank",
        3,
    )

    production = pd.read_parquet(
        PRODUCTION_PATH
    ).copy()

    production_top3_pairs = (
        production_top3(
            production
        )
    )

    unreachable[
        "msbuddy_truth_formula_top3"
    ] = [
        (
            str(row.molecule_id),
            str(row.molecular_formula),
        )
        in msbuddy_top3
        for row
        in unreachable.itertuples(
            index=False
        )
    ]

    unreachable[
        "production_truth_formula_top3"
    ] = [
        (
            str(row.molecule_id),
            str(row.molecular_formula),
        )
        in production_top3_pairs
        for row
        in unreachable.itertuples(
            index=False
        )
    ]

    unreachable[
        "either_formula_gate_top3"
    ] = (
        unreachable[
            "msbuddy_truth_formula_top3"
        ]
        | unreachable[
            "production_truth_formula_top3"
        ]
    )

    summary = pd.DataFrame(
        [
            {
                "unreachable_queries":
                    len(unreachable),
                "msbuddy_truth_formula_top3":
                    int(
                        unreachable[
                            "msbuddy_truth_formula_top3"
                        ].sum()
                    ),
                "production_truth_formula_top3":
                    int(
                        unreachable[
                            "production_truth_formula_top3"
                        ].sum()
                    ),
                "either_formula_gate_top3":
                    int(
                        unreachable[
                            "either_formula_gate_top3"
                        ].sum()
                    ),
                "formula_gate_top3_but_structure_absent":
                    int(
                        unreachable[
                            "either_formula_gate_top3"
                        ].sum()
                    ),
            }
        ]
    )

    unreachable.to_parquet(
        OUTPUT_PATH,
        index=False,
        compression="zstd",
    )

    summary.to_csv(
        SUMMARY_PATH,
        index=False,
    )

    print()
    print(
        "Unreachable blind-query diagnostic"
    )
    print(
        "=" * 72
    )

    print(
        summary.to_string(
            index=False
        )
    )

    print()
    print(
        "First unreachable formulas:"
    )

    print(
        unreachable[
            [
                "molecule_id",
                "molecular_formula",
                "msbuddy_truth_formula_top3",
                "production_truth_formula_top3",
            ]
        ]
        .head(25)
        .to_string(
            index=False
        )
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
