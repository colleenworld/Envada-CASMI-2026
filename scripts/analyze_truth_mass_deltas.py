from pathlib import Path

import numpy as np
import pandas as pd

from casmi26.chemistry.adducts import neutral_mass
from casmi26.retrieval.index import load_retrieval_index


INDEX_DIR = Path(
    "data/processed/retrieval_index"
)

MANIFEST_PATH = Path(
    "data/processed/splits/retrieval_dev.parquet"
)

MASS_ANALYSIS_PATH = Path(
    "data/processed/results/retrieval_dev_mass_filter_analysis.parquet"
)

RESULTS_PATH = Path(
    "data/processed/results/retrieval_dev_truth_mass_deltas.parquet"
)


def infer_neutral_mass(
    precursor_mz,
    adduct,
) -> float | None:
    try:
        mz = float(
            precursor_mz
        )
    except (
        TypeError,
        ValueError,
    ):
        return None

    if not np.isfinite(mz):
        return None

    value = neutral_mass(
        mz,
        str(adduct),
    )

    if (
        value is None
        or not np.isfinite(value)
    ):
        return None

    return float(value)


def main() -> None:
    print(
        "Loading metadata..."
    )

    metadata, _ = (
        load_retrieval_index(
            INDEX_DIR
        )
    )

    manifest = pd.read_parquet(
        MANIFEST_PATH
    )

    previous = pd.read_parquet(
        MASS_ANALYSIS_PATH
    )

    # We only need cases for which the previous ±0.01
    # experiment failed despite having at least one known
    # query neutral mass.
    failures = previous[
        previous["status"].isin(
            [
                "truth_excluded",
                "zero_candidates",
            ]
        )
    ].copy()

    print(
        f"Problem structures: "
        f"{len(failures):,}"
    )

    metadata_keys = (
        metadata[
            "inchikey14"
        ]
        .astype(str)
        .to_numpy()
    )

    metadata_libraries = (
        metadata[
            "ingest_lib"
        ]
        .astype(str)
        .to_numpy()
    )

    manifest_by_key = (
        manifest
        .set_index(
            "inchikey14"
        )
    )

    results = []

    for position, failure in enumerate(
        failures.itertuples(
            index=False
        ),
        start=1,
    ):
        truth_key = str(
            failure.inchikey14
        )

        manifest_row = (
            manifest_by_key.loc[
                truth_key
            ]
        )

        query_library = str(
            manifest_row[
                "query_library"
            ]
        )

        structure_mask = (
            metadata_keys
            == truth_key
        )

        query_mask = (
            structure_mask
            & (
                metadata_libraries
                == query_library
            )
        )

        reference_mask = (
            structure_mask
            & (
                metadata_libraries
                != query_library
            )
        )

        query_rows = (
            metadata.iloc[
                np.flatnonzero(
                    query_mask
                )
            ]
        )

        truth_reference_rows = (
            metadata.iloc[
                np.flatnonzero(
                    reference_mask
                )
            ]
        )

        query_masses = []

        for row in (
            query_rows.itertuples(
                index=False
            )
        ):
            mass = infer_neutral_mass(
                row.precursor_mz,
                row.adduct,
            )

            if mass is not None:
                query_masses.append(
                    (
                        str(
                            row.ionization_mode
                        ),
                        mass,
                    )
                )

        reference_masses = []

        for row in (
            truth_reference_rows.itertuples(
                index=False
            )
        ):
            mass = infer_neutral_mass(
                row.precursor_mz,
                row.adduct,
            )

            if mass is not None:
                reference_masses.append(
                    (
                        str(
                            row.ionization_mode
                        ),
                        mass,
                    )
                )

        differences = []

        for (
            query_mode,
            query_mass,
        ) in query_masses:
            for (
                reference_mode,
                reference_mass,
            ) in reference_masses:
                if (
                    query_mode
                    != reference_mode
                ):
                    continue

                differences.append(
                    abs(
                        query_mass
                        - reference_mass
                    )
                )

        if differences:
            minimum_delta = float(
                min(differences)
            )
        else:
            minimum_delta = np.nan

        results.append(
            {
                "inchikey14": truth_key,
                "query_library": (
                    query_library
                ),
                "previous_status": (
                    failure.status
                ),
                "query_spectrum_count": (
                    len(query_rows)
                ),
                "truth_reference_spectrum_count": (
                    len(
                        truth_reference_rows
                    )
                ),
                "known_query_mass_count": (
                    len(query_masses)
                ),
                "known_truth_reference_mass_count": (
                    len(
                        reference_masses
                    )
                ),
                "same_polarity_comparison_count": (
                    len(differences)
                ),
                "minimum_truth_mass_delta": (
                    minimum_delta
                ),
            }
        )

        print(
            f"{position:>2}/"
            f"{len(failures)}  "
            f"{truth_key}  "
            f"{query_library:<12}  "
            f"delta="
            f"{minimum_delta:.6f}"
            if np.isfinite(
                minimum_delta
            )
            else
            f"{position:>2}/"
            f"{len(failures)}  "
            f"{truth_key}  "
            f"{query_library:<12}  "
            f"delta=None"
        )

    results_frame = pd.DataFrame(
        results
    )

    RESULTS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_frame.to_parquet(
        RESULTS_PATH,
        index=False,
    )

    finite = (
        results_frame[
            results_frame[
                "minimum_truth_mass_delta"
            ].notna()
        ]
    )

    no_comparison = (
        results_frame[
            results_frame[
                "minimum_truth_mass_delta"
            ].isna()
        ]
    )

    print()
    print(
        "Truth neutral-mass delta analysis"
    )
    print(
        "---------------------------------"
    )

    print(
        f"Problem structures:          "
        f"{len(results_frame):,}"
    )

    print(
        f"Comparable same-polarity:    "
        f"{len(finite):,}"
    )

    print(
        f"No comparable truth mass:    "
        f"{len(no_comparison):,}"
    )

    print()
    print(
        "Recovery by tolerance"
    )
    print(
        "---------------------"
    )

    tolerances = [
        0.01,
        0.02,
        0.05,
        0.10,
        0.25,
        0.50,
        1.00,
        2.00,
        5.00,
    ]

    for tolerance in tolerances:
        recovered = int(
            (
                finite[
                    "minimum_truth_mass_delta"
                ]
                <= tolerance
            ).sum()
        )

        print(
            f"±{tolerance:<5.2f} Da  "
            f"{recovered:>3}/"
            f"{len(results_frame)}"
        )

    print()
    print(
        "Delta distribution "
        "(comparable cases)"
    )
    print(
        "-------------------------------"
    )

    if len(finite):
        deltas = finite[
            "minimum_truth_mass_delta"
        ]

        print(
            f"Min:       "
            f"{deltas.min():.6f}"
        )
        print(
            f"Median:    "
            f"{deltas.median():.6f}"
        )
        print(
            f"75th pct:  "
            f"{deltas.quantile(0.75):.6f}"
        )
        print(
            f"90th pct:  "
            f"{deltas.quantile(0.90):.6f}"
        )
        print(
            f"95th pct:  "
            f"{deltas.quantile(0.95):.6f}"
        )
        print(
            f"Max:       "
            f"{deltas.max():.6f}"
        )

    print()
    print(
        "Cases ordered by closest truth delta"
    )
    print(
        "------------------------------------"
    )

    display = (
        results_frame
        .sort_values(
            "minimum_truth_mass_delta",
            na_position="last",
        )
    )

    print(
        display[
            [
                "inchikey14",
                "query_library",
                "previous_status",
                "known_query_mass_count",
                "known_truth_reference_mass_count",
                "minimum_truth_mass_delta",
            ]
        ].to_string(
            index=False
        )
    )

    print()
    print(
        f"Results written to: "
        f"{RESULTS_PATH}"
    )


if __name__ == "__main__":
    main()