from pathlib import Path
import time

import numpy as np
import pandas as pd

from casmi26.chemistry.adducts import neutral_mass
from casmi26.retrieval.index import load_retrieval_index
from casmi26.retrieval.mass_index import (
    build_neutral_mass_index,
    search_neutral_masses,
)


INDEX_DIR = Path(
    "data/processed/retrieval_index"
)

MANIFEST_PATH = Path(
    "data/processed/splits/retrieval_dev.parquet"
)

RESULTS_DIR = Path(
    "data/processed/results"
)

RESULTS_PATH = (
    RESULTS_DIR
    / "retrieval_dev_mass_filter_analysis.parquet"
)

TOLERANCE_DA = 0.01


def infer_neutral_masses(
    precursor_mz: np.ndarray,
    adducts: np.ndarray,
) -> np.ndarray:
    masses = np.full(
        len(precursor_mz),
        np.nan,
        dtype=np.float64,
    )

    for index, (mz, adduct) in enumerate(
        zip(
            precursor_mz,
            adducts,
            strict=True,
        )
    ):
        try:
            mz_value = float(mz)
        except (
            TypeError,
            ValueError,
        ):
            continue

        if not np.isfinite(mz_value):
            continue

        value = neutral_mass(
            mz_value,
            str(adduct),
        )

        if (
            value is not None
            and np.isfinite(value)
        ):
            masses[index] = value

    return masses


def build_reference_pool(
    metadata: pd.DataFrame,
    manifest: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build exactly the same dev reference pool used by the
    retrieval baselines.
    """
    query_pairs = set(
        zip(
            manifest[
                "inchikey14"
            ].astype(str),
            manifest[
                "query_library"
            ].astype(str),
            strict=True,
        )
    )

    reference_mask = np.fromiter(
        (
            (
                str(key),
                str(library),
            )
            not in query_pairs
            for key, library in zip(
                metadata["inchikey14"],
                metadata["ingest_lib"],
                strict=True,
            )
        ),
        dtype=bool,
        count=len(metadata),
    )

    reference_indices = np.flatnonzero(
        reference_mask
    )

    return (
        metadata.iloc[
            reference_indices
        ]
        .reset_index(drop=True)
    )


def classify(
    known_query_mass_count: int,
    candidate_count: int,
    truth_survived: bool,
) -> str:
    if known_query_mass_count == 0:
        return "no_query_mass"

    if candidate_count == 0:
        return "zero_candidates"

    if not truth_survived:
        return "truth_excluded"

    return "truth_retained"


def main() -> None:
    started = time.monotonic()

    print(
        "Loading retrieval metadata..."
    )

    metadata, _ = (
        load_retrieval_index(
            INDEX_DIR
        )
    )

    manifest = pd.read_parquet(
        MANIFEST_PATH
    )

    print(
        f"Index spectra:  {len(metadata):,}"
    )
    print(
        f"Dev structures: {len(manifest):,}"
    )

    # ---------------------------------------------------------
    # Build the frozen dev reference pool.
    # ---------------------------------------------------------

    print()
    print(
        "Building dev reference pool..."
    )

    reference_metadata = (
        build_reference_pool(
            metadata=metadata,
            manifest=manifest,
        )
    )

    removed = (
        len(metadata)
        - len(reference_metadata)
    )

    print(
        f"Reference spectra: "
        f"{len(reference_metadata):,}"
    )
    print(
        f"Query spectra removed: "
        f"{removed:,}"
    )

    if removed != 19_812:
        raise RuntimeError(
            "Expected exactly 19,812 query "
            f"spectra removed; got {removed:,}"
        )

    # ---------------------------------------------------------
    # Prepare reference mass index.
    # ---------------------------------------------------------

    reference_keys = (
        reference_metadata[
            "inchikey14"
        ]
        .astype(str)
        .to_numpy()
    )

    reference_modes = (
        reference_metadata[
            "ionization_mode"
        ]
        .astype(str)
        .to_numpy()
    )

    reference_neutral_masses = (
        infer_neutral_masses(
            reference_metadata[
                "precursor_mz"
            ].to_numpy(),
            reference_metadata[
                "adduct"
            ]
            .astype(str)
            .to_numpy(),
        )
    )

    known_reference_mass_count = int(
        np.count_nonzero(
            np.isfinite(
                reference_neutral_masses
            )
        )
    )

    print(
        f"Known reference masses: "
        f"{known_reference_mass_count:,} / "
        f"{len(reference_metadata):,} "
        f"("
        f"{known_reference_mass_count / len(reference_metadata):.2%}"
        f")"
    )

    print(
        "Building mass index..."
    )

    mass_index = (
        build_neutral_mass_index(
            neutral_masses=(
                reference_neutral_masses
            ),
            modes=reference_modes,
        )
    )

    # ---------------------------------------------------------
    # Prepare full-index arrays used to locate queries.
    # ---------------------------------------------------------

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

    results = []

    print()
    print(
        "Analyzing 2,000 dev structures..."
    )

    # ---------------------------------------------------------
    # Analyze every dev structure.
    # ---------------------------------------------------------

    for position, row in enumerate(
        manifest.itertuples(
            index=False
        ),
        start=1,
    ):
        truth_key = str(
            row.inchikey14
        )

        query_library = str(
            row.query_library
        )

        query_mask = (
            (
                metadata_keys
                == truth_key
            )
            & (
                metadata_libraries
                == query_library
            )
        )

        query_indices = (
            np.flatnonzero(
                query_mask
            )
        )

        if len(query_indices) == 0:
            raise RuntimeError(
                "No query spectra for "
                f"{truth_key} / "
                f"{query_library}"
            )

        query_metadata = (
            metadata.iloc[
                query_indices
            ]
        )

        query_modes = (
            query_metadata[
                "ionization_mode"
            ]
            .astype(str)
            .to_numpy()
        )

        query_neutral_masses = (
            infer_neutral_masses(
                query_metadata[
                    "precursor_mz"
                ].to_numpy(),
                query_metadata[
                    "adduct"
                ]
                .astype(str)
                .to_numpy(),
            )
        )

        known_query_mass_count = int(
            np.count_nonzero(
                np.isfinite(
                    query_neutral_masses
                )
            )
        )

        filtered_reference_indices = (
            search_neutral_masses(
                index=mass_index,
                neutral_masses=(
                    query_neutral_masses
                ),
                modes=query_modes,
                tolerance_da=(
                    TOLERANCE_DA
                ),
            )
        )

        if len(
            filtered_reference_indices
        ):
            filtered_keys = (
                reference_keys[
                    filtered_reference_indices
                ]
            )

            candidate_count = int(
                len(
                    np.unique(
                        filtered_keys
                    )
                )
            )

            truth_survived = bool(
                np.any(
                    filtered_keys
                    == truth_key
                )
            )
        else:
            candidate_count = 0
            truth_survived = False

        status = classify(
            known_query_mass_count=(
                known_query_mass_count
            ),
            candidate_count=(
                candidate_count
            ),
            truth_survived=(
                truth_survived
            ),
        )

        results.append(
            {
                "inchikey14": truth_key,
                "query_library": (
                    query_library
                ),
                "query_spectrum_count": (
                    len(query_indices)
                ),
                "known_query_mass_count": (
                    known_query_mass_count
                ),
                "filtered_reference_spectrum_count": (
                    len(
                        filtered_reference_indices
                    )
                ),
                "candidate_count": (
                    candidate_count
                ),
                "truth_survived": (
                    truth_survived
                ),
                "status": status,
            }
        )

        if (
            position % 100
            == 0
        ):
            print(
                f"{position:>4}/"
                f"{len(manifest)}"
            )

    # ---------------------------------------------------------
    # Save.
    # ---------------------------------------------------------

    results_frame = pd.DataFrame(
        results
    )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_frame.to_parquet(
        RESULTS_PATH,
        index=False,
    )

    # ---------------------------------------------------------
    # Overall status.
    # ---------------------------------------------------------

    print()
    print(
        "Mass-filter analysis"
    )
    print(
        "--------------------"
    )

    print(
        f"Structures: "
        f"{len(results_frame):,}"
    )

    print(
        f"Tolerance:  "
        f"±{TOLERANCE_DA:.3f} Da"
    )

    print()

    status_counts = (
        results_frame[
            "status"
        ]
        .value_counts()
    )

    for status in [
        "truth_retained",
        "no_query_mass",
        "zero_candidates",
        "truth_excluded",
    ]:
        count = int(
            status_counts.get(
                status,
                0,
            )
        )

        percentage = (
            count
            / len(results_frame)
        )

        print(
            f"{status:<18} "
            f"{count:>5,} "
            f"({percentage:>6.2%})"
        )

    # ---------------------------------------------------------
    # Candidate distribution.
    # ---------------------------------------------------------

    print()
    print(
        "Candidate counts"
    )
    print(
        "----------------"
    )

    candidate_counts = (
        results_frame[
            "candidate_count"
        ]
    )

    print(
        f"Min:      "
        f"{candidate_counts.min():,.0f}"
    )
    print(
        f"Median:   "
        f"{candidate_counts.median():,.0f}"
    )
    print(
        f"Mean:     "
        f"{candidate_counts.mean():,.2f}"
    )
    print(
        f"90th pct: "
        f"{candidate_counts.quantile(0.90):,.0f}"
    )
    print(
        f"95th pct: "
        f"{candidate_counts.quantile(0.95):,.0f}"
    )
    print(
        f"99th pct: "
        f"{candidate_counts.quantile(0.99):,.0f}"
    )
    print(
        f"Max:      "
        f"{candidate_counts.max():,.0f}"
    )

    # ---------------------------------------------------------
    # Breakdown by library and status.
    # ---------------------------------------------------------

    print()
    print(
        "Status by query library"
    )
    print(
        "-----------------------"
    )

    library_status = pd.crosstab(
        results_frame[
            "query_library"
        ],
        results_frame[
            "status"
        ],
    )

    library_status = (
        library_status.reindex(
            columns=[
                "truth_retained",
                "no_query_mass",
                "zero_candidates",
                "truth_excluded",
            ],
            fill_value=0,
        )
    )

    print(
        library_status.to_string()
    )

    # ---------------------------------------------------------
    # Most important diagnostic: non-empty filter that lost truth.
    # ---------------------------------------------------------

    truth_excluded = (
        results_frame[
            results_frame[
                "status"
            ]
            == "truth_excluded"
        ]
    )

    print()
    print(
        "Non-empty candidate sets "
        "that excluded truth"
    )
    print(
        "---------------------------------------"
    )

    print(
        f"Structures: "
        f"{len(truth_excluded):,}"
    )

    if len(
        truth_excluded
    ):
        print()
        print(
            truth_excluded[
                [
                    "inchikey14",
                    "query_library",
                    "query_spectrum_count",
                    "known_query_mass_count",
                    "candidate_count",
                ]
            ].to_string(
                index=False
            )
        )

    # ---------------------------------------------------------
    # Zero-candidate cases.
    # ---------------------------------------------------------

    zero_candidates = (
        results_frame[
            results_frame[
                "status"
            ]
            == "zero_candidates"
        ]
    )

    print()
    print(
        "Known query mass but zero candidates"
    )
    print(
        "------------------------------------"
    )

    print(
        f"Structures: "
        f"{len(zero_candidates):,}"
    )

    if len(
        zero_candidates
    ):
        print()
        print(
            zero_candidates[
                [
                    "inchikey14",
                    "query_library",
                    "query_spectrum_count",
                    "known_query_mass_count",
                ]
            ].to_string(
                index=False
            )
        )

    elapsed = (
        time.monotonic()
        - started
    )

    print()
    print(
        f"Results written to: "
        f"{RESULTS_PATH}"
    )

    print(
        f"Elapsed: "
        f"{elapsed:.2f}s"
    )


if __name__ == "__main__":
    main()