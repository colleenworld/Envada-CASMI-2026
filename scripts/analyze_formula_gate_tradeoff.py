from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent

ROOT = (
    PROJECT_ROOT
    / "data/processed/blind_pseudotest"
)

ENUMERATION_PATH = (
    ROOT
    / "blind_formula_enumeration.parquet"
)

OUTPUT_SUMMARY_PATH = (
    ROOT
    / "formula_gate_tradeoff.csv"
)

RECOMMENDED_CUTOFF = 25


def main() -> None:
    if not ENUMERATION_PATH.exists():
        raise FileNotFoundError(
            f"Missing required artifact: {ENUMERATION_PATH}"
        )

    frame = pd.read_parquet(
        ENUMERATION_PATH
    ).copy()

    required = {
        "molecule_id",
        "truth_formula",
        "generated_formula_count",
        "truth_generated",
        "truth_formula_rank_by_abs_ppm",
    }

    missing = (
        required
        - set(
            frame.columns
        )
    )

    if missing:
        raise ValueError(
            "Enumeration output missing columns: "
            f"{sorted(missing)}"
        )

    cutoffs = [
        3,
        5,
        10,
        25,
        50,
        100,
    ]

    rows = []

    for cutoff in cutoffs:
        retained = (
            frame[
                "truth_formula_rank_by_abs_ppm"
            ]
            .notna()
            & (
                frame[
                    "truth_formula_rank_by_abs_ppm"
                ]
                <= cutoff
            )
        )

        formulas_retained = (
            frame[
                "generated_formula_count"
            ]
            .clip(
                upper=cutoff
            )
        )

        rows.append(
            {
                "cutoff":
                    cutoff,
                "truth_retained":
                    int(
                        retained.sum()
                    ),
                "truth_retained_fraction":
                    retained.mean(),
                "mean_formulas_per_query":
                    formulas_retained.mean(),
                "median_formulas_per_query":
                    formulas_retained.median(),
                "p90_formulas_per_query":
                    formulas_retained.quantile(
                        0.90
                    ),
                "total_formula_slots":
                    int(
                        formulas_retained.sum()
                    ),
            }
        )

    summary = pd.DataFrame(
        rows
    )

    summary.to_csv(
        OUTPUT_SUMMARY_PATH,
        index=False,
    )

    print()
    print(
        "Formula gate tradeoff"
    )
    print(
        "=" * 92
    )
    print(
        summary.to_string(
            index=False,
            formatters={
                "truth_retained_fraction":
                    lambda value:
                        f"{value:.2%}",
                "mean_formulas_per_query":
                    lambda value:
                        f"{value:.1f}",
                "median_formulas_per_query":
                    lambda value:
                        f"{value:.1f}",
                "p90_formulas_per_query":
                    lambda value:
                        f"{value:.1f}",
            },
        )
    )

    recommended = (
        summary.loc[
            summary[
                "cutoff"
            ]
            == RECOMMENDED_CUTOFF
        ]
        .iloc[
            0
        ]
    )

    print()
    print(
        "Recommended diagnostic expansion cutoff:",
        RECOMMENDED_CUTOFF,
    )
    print(
        "Truth retained:",
        (
            f"{int(recommended['truth_retained'])} / "
            f"{len(frame)} "
            f"({recommended['truth_retained_fraction']:.2%})"
        ),
    )
    print(
        "Formula slots:",
        int(
            recommended[
                "total_formula_slots"
            ]
        ),
    )
    print()
    print(
        "Wrote:",
        OUTPUT_SUMMARY_PATH,
    )


if __name__ == "__main__":
    main()
