from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent

UNIVERSE_PATH = (
    PROJECT_ROOT
    / "data/processed/formula_index/training_prior_formula_universe_top25.parquet"
)

BLIND_SHORTLIST_PATH = (
    PROJECT_ROOT
    / "data/processed/blind_pseudotest/formula_prior_validation.parquet"
)

BLIND_TRUTH_PATH = (
    PROJECT_ROOT
    / "data/processed/blind_pseudotest/truth.parquet"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data/processed/blind_pseudotest/"
      "training_formula_universe_coverage.csv"
)

CHECKPOINTS = [
    1_000,
    2_500,
    5_000,
    10_000,
    20_000,
]


def main() -> None:
    for path in [
        UNIVERSE_PATH,
        BLIND_TRUTH_PATH,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"Missing required artifact: {path}"
            )

    universe = pd.read_parquet(
        UNIVERSE_PATH
    ).copy()

    required_universe = {
        "molecular_formula",
        "first_anchor_index",
    }

    missing = (
        required_universe
        - set(universe.columns)
    )

    if missing:
        raise ValueError(
            "Formula universe missing columns: "
            f"{sorted(missing)}"
        )

    truth = pd.read_parquet(
        BLIND_TRUTH_PATH,
        columns=[
            "molecule_id",
            "molecular_formula",
        ],
    ).copy()

    truth[
        "molecule_id"
    ] = truth[
        "molecule_id"
    ].astype(str)

    truth[
        "molecular_formula"
    ] = truth[
        "molecular_formula"
    ].astype(str)

    validation = None

    if BLIND_SHORTLIST_PATH.exists():
        validation = pd.read_parquet(
            BLIND_SHORTLIST_PATH
        ).copy()

    rows = []

    max_anchor = int(
        universe[
            "first_anchor_index"
        ].max()
    )

    checkpoints = sorted(
        {
            checkpoint
            for checkpoint in CHECKPOINTS
            if checkpoint <= max_anchor
        }
        | {
            max_anchor,
        }
    )

    for checkpoint in checkpoints:
        formulas = set(
            universe.loc[
                universe[
                    "first_anchor_index"
                ]
                <= checkpoint,
                "molecular_formula",
            ]
            .astype(str)
        )

        truth_hits = (
            truth[
                "molecular_formula"
            ]
            .isin(
                formulas
            )
        )

        row = {
            "anchor_count":
                checkpoint,
            "universe_formula_count":
                len(formulas),
            "truth_formulas_covered":
                int(
                    truth_hits.sum()
                ),
            "truth_formula_coverage":
                float(
                    truth_hits.mean()
                ),
        }

        if validation is not None:
            required = {
                "molecule_id",
                "truth_formula",
                "mass_rank",
                "prior_rank",
            }

            if required <= set(
                validation.columns
            ):
                generated = validation.loc[
                    validation[
                        "prior_rank"
                    ].notna()
                ].copy()

                generated[
                    "molecule_id"
                ] = generated[
                    "molecule_id"
                ].astype(str)

                generated[
                    "truth_formula"
                ] = generated[
                    "truth_formula"
                ].astype(str)

                generated_truth_hits = (
                    generated[
                        "truth_formula"
                    ]
                    .isin(
                        formulas
                    )
                )

                row[
                    "generated_truth_queries"
                ] = len(
                    generated
                )

                row[
                    "generated_truth_covered"
                ] = int(
                    generated_truth_hits.sum()
                )

                row[
                    "generated_truth_coverage"
                ] = float(
                    generated_truth_hits.mean()
                )

        rows.append(
            row
        )

    summary = pd.DataFrame(
        rows
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print()
    print(
        "Hidden-independent training formula universe coverage"
    )
    print("=" * 110)

    formatters = {
        "truth_formula_coverage":
            lambda value:
                f"{value:.2%}",
    }

    if (
        "generated_truth_coverage"
        in summary.columns
    ):
        formatters[
            "generated_truth_coverage"
        ] = (
            lambda value:
                f"{value:.2%}"
        )

    print(
        summary.to_string(
            index=False,
            formatters=formatters,
        )
    )

    print()
    print(
        "Wrote:",
        OUTPUT_PATH,
    )


if __name__ == "__main__":
    main()
