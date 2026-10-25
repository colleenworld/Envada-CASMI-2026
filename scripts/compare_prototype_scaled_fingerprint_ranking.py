from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(
    "data/processed/diagnostics/large_validation_audit/"
    "true_formula_neighbor_benchmark/"
    "learned_ranker_top500"
)

PROTOTYPE_PATHS = [
    ROOT / "spectrum_fingerprint_enrichment.parquet",
    ROOT / "spectrum_fingerprint_enrichment_prototype.parquet",
]

SCALED_PATH = (
    ROOT
    / "spectrum_fingerprint_enrichment_scaled150k.parquet"
)

FEATURES = [
    "predicted_fp_soft_jaccard",
    "predicted_fp_cosine",
    "predicted_fp_positive_mean",
    "predicted_fp_log_likelihood",
]


def rr25(rank) -> float:
    if rank is None or pd.isna(rank):
        return 0.0

    rank = int(rank)

    if rank < 1 or rank > 25:
        return 0.0

    return 1.0 / rank


def ranks_for(
    frame: pd.DataFrame,
    feature: str,
) -> pd.Series:
    ranked = (
        frame
        .sort_values(
            [
                "inchikey14",
                feature,
                "candidate_inchikey14",
            ],
            ascending=[
                True,
                False,
                True,
            ],
            kind="stable",
        )
        .copy()
    )

    ranked[
        "_rank"
    ] = (
        ranked.groupby(
            "inchikey14",
            sort=False,
        )
        .cumcount()
        + 1
    )

    truth = (
        ranked.loc[
            ranked[
                "candidate_inchikey14"
            ].astype(str)
            == ranked[
                "inchikey14"
            ].astype(str),
            [
                "inchikey14",
                "_rank",
            ],
        ]
        .drop_duplicates(
            subset=[
                "inchikey14"
            ]
        )
        .set_index(
            "inchikey14"
        )[
            "_rank"
        ]
    )

    query_keys = (
        frame[
            "inchikey14"
        ]
        .drop_duplicates()
        .astype(str)
    )

    return query_keys.map(
        truth
    )


def summarize(
    model: str,
    feature: str,
    ranks: pd.Series,
) -> dict[str, object]:
    return {
        "model":
            model,
        "feature":
            feature,
        "queries":
            len(ranks),
        "truth_returned":
            ranks.notna().mean(),
        "mrr25":
            ranks.map(
                rr25
            ).mean(),
        "top1":
            ranks.eq(1).mean(),
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
        "top100":
            ranks.between(
                1,
                100,
            ).mean(),
        "median_truth_rank":
            ranks.median(),
    }


def separation(
    frame: pd.DataFrame,
    feature: str,
) -> dict[str, float]:
    truth_mask = (
        frame[
            "candidate_inchikey14"
        ].astype(str)
        == frame[
            "inchikey14"
        ].astype(str)
    )

    truth = (
        frame.loc[
            truth_mask,
            feature,
        ]
        .dropna()
    )

    false = (
        frame.loc[
            ~truth_mask,
            feature,
        ]
        .dropna()
    )

    return {
        "truth_median":
            float(
                truth.median()
            ),
        "nontruth_median":
            float(
                false.median()
            ),
        "median_gap":
            float(
                truth.median()
                - false.median()
            ),
    }


def main() -> None:
    prototype_path = next(
        (
            path
            for path in PROTOTYPE_PATHS
            if path.exists()
        ),
        None,
    )

    if prototype_path is None:
        raise FileNotFoundError(
            "Could not find prototype fingerprint enrichment"
        )

    if not SCALED_PATH.exists():
        raise FileNotFoundError(
            f"Missing scaled enrichment: {SCALED_PATH}"
        )

    prototype = pd.read_parquet(
        prototype_path
    )

    scaled = pd.read_parquet(
        SCALED_PATH
    )

    key_columns = [
        "inchikey14",
        "candidate_formula",
        "candidate_inchikey14",
    ]

    for frame in [
        prototype,
        scaled,
    ]:
        for column in key_columns:
            frame[
                column
            ] = (
                frame[
                    column
                ]
                .astype(str)
            )

    prototype_keys = set(
        map(
            tuple,
            prototype[
                key_columns
            ].itertuples(
                index=False,
                name=None,
            ),
        )
    )

    scaled_keys = set(
        map(
            tuple,
            scaled[
                key_columns
            ].itertuples(
                index=False,
                name=None,
            ),
        )
    )

    print()
    print(
        "Prototype vs scaled fingerprint ranking"
    )
    print(
        "=" * 80
    )
    print(
        "Prototype artifact:",
        prototype_path,
    )
    print(
        "Scaled artifact:",
        SCALED_PATH,
    )
    print(
        "Prototype rows:",
        f"{len(prototype):,}",
    )
    print(
        "Scaled rows:",
        f"{len(scaled):,}",
    )
    print(
        "Candidate key sets identical:",
        prototype_keys
        == scaled_keys,
    )

    summary_rows = []

    for model_name, frame in [
        (
            "prototype",
            prototype,
        ),
        (
            "scaled150k",
            scaled,
        ),
    ]:
        print()
        print(
            model_name
        )
        print(
            "-" * 80
        )

        for feature in FEATURES:
            ranks = ranks_for(
                frame,
                feature,
            )

            summary_rows.append(
                summarize(
                    model_name,
                    feature,
                    ranks,
                )
            )

            sep = separation(
                frame,
                feature,
            )

            print(
                feature
            )
            print(
                "  truth median:",
                f"{sep['truth_median']:.6f}",
            )
            print(
                "  nontruth median:",
                f"{sep['nontruth_median']:.6f}",
            )
            print(
                "  median gap:",
                f"{sep['median_gap']:.6f}",
            )
            print(
                "  MRR@25:",
                f"{ranks.map(rr25).mean():.4f}",
            )
            print(
                "  Top-1:",
                f"{ranks.eq(1).mean():.2%}",
            )
            print(
                "  Top-25:",
                f"{ranks.between(1,25).mean():.2%}",
            )
            print(
                "  Top-100:",
                f"{ranks.between(1,100).mean():.2%}",
            )
            print(
                "  Median truth rank:",
                f"{ranks.median():.1f}",
            )

    summary = pd.DataFrame(
        summary_rows
    )

    print()
    print(
        "Summary"
    )
    print(
        "-" * 80
    )
    print(
        summary.to_string(
            index=False,
            formatters={
                "truth_returned":
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
                "top100":
                    lambda value:
                    f"{value:.2%}",
                "median_truth_rank":
                    lambda value:
                    f"{value:.1f}",
            },
        )
    )


if __name__ == "__main__":
    main()
