from pathlib import Path

import pandas as pd


RESULTS_PATH = Path(
    "data/processed/results/retrieval_dev_baseline1.parquet"
)

MANIFEST_PATH = Path(
    "data/processed/splits/retrieval_dev.parquet"
)


def describe_group(
    frame: pd.DataFrame,
    label: str,
) -> None:
    print()
    print(label)
    print("-" * len(label))

    print(
        f"Structures:               "
        f"{len(frame):,}"
    )

    if len(frame) == 0:
        return

    print(
        f"Query spectra median:     "
        f"{frame['query_spectrum_count'].median():.1f}"
    )

    print(
        f"Query spectra mean:       "
        f"{frame['query_spectrum_count'].mean():.2f}"
    )

    print(
        f"Reference libraries med:  "
        f"{frame['reference_library_count'].median():.1f}"
    )

    print(
        f"Reference libraries mean: "
        f"{frame['reference_library_count'].mean():.2f}"
    )

    print(
        f"Reference spectra median: "
        f"{frame['reference_spectrum_count'].median():.1f}"
    )

    print(
        f"Reference spectra mean:   "
        f"{frame['reference_spectrum_count'].mean():.2f}"
    )


def main() -> None:
    results = pd.read_parquet(
        RESULTS_PATH
    )

    manifest = pd.read_parquet(
        MANIFEST_PATH
    )

    if len(results) != 2_000:
        raise RuntimeError(
            "Expected 2,000 completed Baseline 1 results, "
            f"found {len(results):,}"
        )

    data = manifest.merge(
        results[
            [
                "inchikey14",
                "rank",
                "reciprocal_rank",
            ]
        ],
        on="inchikey14",
        how="left",
        validate="one_to_one",
    )

    if data["reciprocal_rank"].isna().any():
        raise RuntimeError(
            "Some manifest structures are missing "
            "Baseline 1 results"
        )

    data["top1"] = (
        data["rank"] == 1
    )

    data["top25"] = (
        data["rank"].notna()
        & (data["rank"] <= 25)
    )

    # ---------------------------------------------------------
    # Overall failure counts.
    # ---------------------------------------------------------

    print(
        "Baseline 1 failure analysis"
    )
    print(
        "==========================="
    )

    print()
    print(
        f"Structures:       "
        f"{len(data):,}"
    )

    print(
        f"Top-1:            "
        f"{data['top1'].sum():,}"
    )

    print(
        f"Ranks 2-25:       "
        f"{((~data['top1']) & data['top25']).sum():,}"
    )

    print(
        f"Outside Top-25:   "
        f"{(~data['top25']).sum():,}"
    )

    # ---------------------------------------------------------
    # Failure rate by query library.
    # ---------------------------------------------------------

    library_summary = (
        data
        .groupby(
            "query_library"
        )
        .agg(
            structures=(
                "inchikey14",
                "size",
            ),
            top1=(
                "top1",
                "sum",
            ),
            top25=(
                "top25",
                "sum",
            ),
        )
    )

    library_summary[
        "outside_top25"
    ] = (
        library_summary[
            "structures"
        ]
        - library_summary[
            "top25"
        ]
    )

    library_summary[
        "outside_top25_rate"
    ] = (
        library_summary[
            "outside_top25"
        ]
        / library_summary[
            "structures"
        ]
    )

    library_summary = (
        library_summary.sort_values(
            "outside_top25_rate",
            ascending=False,
        )
    )

    print()
    print(
        "Failures by query library"
    )
    print(
        "-------------------------"
    )

    print(
        library_summary.to_string(
            formatters={
                "outside_top25_rate": (
                    lambda value:
                    f"{value:.2%}"
                )
            }
        )
    )

    # ---------------------------------------------------------
    # Compare successful and failed structures.
    # ---------------------------------------------------------

    describe_group(
        data[
            data["top1"]
        ],
        "Top-1 structures",
    )

    describe_group(
        data[
            (
                ~data["top1"]
            )
            & data["top25"]
        ],
        "Ranks 2-25",
    )

    describe_group(
        data[
            ~data["top25"]
        ],
        "Outside Top-25",
    )

    # ---------------------------------------------------------
    # Query-spectrum-count buckets.
    # ---------------------------------------------------------

    data[
        "query_spectrum_bucket"
    ] = pd.cut(
        data[
            "query_spectrum_count"
        ],
        bins=[
            0,
            1,
            2,
            5,
            10,
            25,
            50,
            float("inf"),
        ],
        labels=[
            "1",
            "2",
            "3-5",
            "6-10",
            "11-25",
            "26-50",
            "51+",
        ],
    )

    spectrum_summary = (
        data
        .groupby(
            "query_spectrum_bucket",
            observed=True,
        )
        .agg(
            structures=(
                "inchikey14",
                "size",
            ),
            top1_rate=(
                "top1",
                "mean",
            ),
            top25_rate=(
                "top25",
                "mean",
            ),
        )
    )

    print()
    print(
        "Performance by query spectrum count"
    )
    print(
        "-----------------------------------"
    )

    print(
        spectrum_summary.to_string(
            formatters={
                "top1_rate": (
                    lambda value:
                    f"{value:.2%}"
                ),
                "top25_rate": (
                    lambda value:
                    f"{value:.2%}"
                ),
            }
        )
    )

    # ---------------------------------------------------------
    # Reference-library-count analysis.
    # ---------------------------------------------------------

    reference_library_summary = (
        data
        .groupby(
            "reference_library_count"
        )
        .agg(
            structures=(
                "inchikey14",
                "size",
            ),
            top1_rate=(
                "top1",
                "mean",
            ),
            top25_rate=(
                "top25",
                "mean",
            ),
        )
    )

    print()
    print(
        "Performance by reference library count"
    )
    print(
        "--------------------------------------"
    )

    print(
        reference_library_summary.to_string(
            formatters={
                "top1_rate": (
                    lambda value:
                    f"{value:.2%}"
                ),
                "top25_rate": (
                    lambda value:
                    f"{value:.2%}"
                ),
            }
        )
    )

    # ---------------------------------------------------------
    # Worst-ranked structures.
    # ---------------------------------------------------------

    ranked_failures = (
        data[
            data["rank"].notna()
        ]
        .sort_values(
            "rank",
            ascending=False,
        )
        .head(20)
    )

    print()
    print(
        "20 worst finite truth ranks"
    )
    print(
        "---------------------------"
    )

    print(
        ranked_failures[
            [
                "inchikey14",
                "query_library",
                "rank",
                "query_spectrum_count",
                "reference_library_count",
                "reference_spectrum_count",
            ]
        ].to_string(
            index=False
        )
    )

    missing_truth = data[
        data["rank"].isna()
    ]

    print()
    print(
        f"Truth absent from searchable "
        f"same-polarity candidates: "
        f"{len(missing_truth):,}"
    )

    if len(missing_truth):
        print()

        print(
            missing_truth[
                [
                    "inchikey14",
                    "query_library",
                    "query_spectrum_count",
                    "reference_library_count",
                    "reference_spectrum_count",
                ]
            ]
            .head(20)
            .to_string(
                index=False
            )
        )


if __name__ == "__main__":
    main()