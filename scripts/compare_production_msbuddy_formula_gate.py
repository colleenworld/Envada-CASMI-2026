from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent

BLIND_ROOT = (
    PROJECT_ROOT
    / "data/processed/blind_pseudotest"
)

TRUTH_PATH = BLIND_ROOT / "truth.parquet"

PRODUCTION_HYPOTHESES_PATH = (
    PROJECT_ROOT
    / "data/processed/test/"
      "multispectrum_formula_hypotheses.parquet"
)

MSBUDDY_PATH = (
    BLIND_ROOT
    / "msbuddy_formula_gate.parquet"
)

OUTPUT_PATH = (
    BLIND_ROOT
    / "formula_gate_comparison.csv"
)


def production_ranks(
    hypotheses: pd.DataFrame,
) -> pd.DataFrame:
    required = {
        "molecule_id",
        "candidate_formula",
        "min_abs_mass_error_ppm",
    }

    missing = required - set(
        hypotheses.columns
    )

    if missing:
        raise ValueError(
            "Production hypotheses missing columns: "
            f"{sorted(missing)}"
        )

    work = hypotheses[
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
            ],
            keep="first",
        )
        .reset_index(
            drop=True
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

    return work


def formula_sets(
    ranked: pd.DataFrame,
    cutoff: int,
) -> dict[str, set[str]]:
    return {
        str(molecule_id): set(
            group.loc[
                group["rank"] <= cutoff,
                "candidate_formula",
            ].astype(str)
        )
        for molecule_id, group
        in ranked.groupby(
            "molecule_id",
            sort=False,
        )
    }


def main() -> None:
    for path in [
        TRUTH_PATH,
        PRODUCTION_HYPOTHESES_PATH,
        MSBUDDY_PATH,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"Missing required artifact: {path}"
            )

    truth = pd.read_parquet(
        TRUTH_PATH,
        columns=[
            "molecule_id",
            "molecular_formula",
        ],
    ).copy()

    truth["molecule_id"] = (
        truth["molecule_id"]
        .astype(str)
    )

    truth["molecular_formula"] = (
        truth["molecular_formula"]
        .astype(str)
    )

    production = production_ranks(
        pd.read_parquet(
            PRODUCTION_HYPOTHESES_PATH
        )
    )

    msbuddy = pd.read_parquet(
        MSBUDDY_PATH
    ).copy()

    required_msbuddy = {
        "molecule_id",
        "candidate_formula",
        "molecule_rank",
    }

    missing = required_msbuddy - set(
        msbuddy.columns
    )

    if missing:
        raise ValueError(
            "MSBuddy benchmark missing columns: "
            f"{sorted(missing)}"
        )

    msbuddy = msbuddy[
        [
            "molecule_id",
            "candidate_formula",
            "molecule_rank",
        ]
    ].rename(
        columns={
            "molecule_rank":
                "rank",
        }
    )

    msbuddy["molecule_id"] = (
        msbuddy["molecule_id"]
        .astype(str)
    )

    msbuddy["candidate_formula"] = (
        msbuddy["candidate_formula"]
        .astype(str)
    )

    rows = []

    for cutoff in [
        1,
        3,
        5,
    ]:
        prod_sets = formula_sets(
            production,
            cutoff,
        )

        buddy_sets = formula_sets(
            msbuddy,
            cutoff,
        )

        evaluated = truth.copy()

        evaluated[
            "production_hit"
        ] = [
            formula in prod_sets.get(
                molecule_id,
                set(),
            )
            for molecule_id, formula
            in zip(
                evaluated["molecule_id"],
                evaluated["molecular_formula"],
                strict=True,
            )
        ]

        evaluated[
            "msbuddy_hit"
        ] = [
            formula in buddy_sets.get(
                molecule_id,
                set(),
            )
            for molecule_id, formula
            in zip(
                evaluated["molecule_id"],
                evaluated["molecular_formula"],
                strict=True,
            )
        ]

        both = (
            evaluated[
                "production_hit"
            ]
            & evaluated[
                "msbuddy_hit"
            ]
        )

        production_only = (
            evaluated[
                "production_hit"
            ]
            & ~evaluated[
                "msbuddy_hit"
            ]
        )

        msbuddy_only = (
            ~evaluated[
                "production_hit"
            ]
            & evaluated[
                "msbuddy_hit"
            ]
        )

        neither = (
            ~evaluated[
                "production_hit"
            ]
            & ~evaluated[
                "msbuddy_hit"
            ]
        )

        union = (
            evaluated[
                "production_hit"
            ]
            | evaluated[
                "msbuddy_hit"
            ]
        )

        rows.append(
            {
                "cutoff":
                    cutoff,
                "queries":
                    len(
                        evaluated
                    ),
                "production_hits":
                    int(
                        evaluated[
                            "production_hit"
                        ].sum()
                    ),
                "production_recall":
                    float(
                        evaluated[
                            "production_hit"
                        ].mean()
                    ),
                "msbuddy_hits":
                    int(
                        evaluated[
                            "msbuddy_hit"
                        ].sum()
                    ),
                "msbuddy_recall":
                    float(
                        evaluated[
                            "msbuddy_hit"
                        ].mean()
                    ),
                "both_hit":
                    int(
                        both.sum()
                    ),
                "production_only":
                    int(
                        production_only.sum()
                    ),
                "msbuddy_only":
                    int(
                        msbuddy_only.sum()
                    ),
                "neither":
                    int(
                        neither.sum()
                    ),
                "union_hits":
                    int(
                        union.sum()
                    ),
                "union_recall":
                    float(
                        union.mean()
                    ),
            }
        )

    summary = pd.DataFrame(
        rows
    )

    summary.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print()
    print(
        "Production vs MSBuddy formula gate"
    )
    print("=" * 118)
    print(
        summary.to_string(
            index=False,
            formatters={
                "production_recall":
                    lambda value:
                        f"{value:.2%}",
                "msbuddy_recall":
                    lambda value:
                        f"{value:.2%}",
                "union_recall":
                    lambda value:
                        f"{value:.2%}",
            },
        )
    )

    print()
    print(
        "Union recall is an upper bound using up to "
        "2 × cutoff formulas, not a deployable Top-K policy."
    )
    print()
    print(
        "Wrote:",
        OUTPUT_PATH,
    )


if __name__ == "__main__":
    main()
