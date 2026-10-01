from dataclasses import dataclass, field

import numpy as np
import pandas as pd


DEFAULT_LIBRARY_QUOTAS = {
    "gnps": 400,
    "pluskal_ms2": 350,
    "riken": 350,
    "mona": 300,
    "msdial": 200,
    "massbank": 200,
    "enveda-180": 150,
    "drug_plus": 50,
}


@dataclass(frozen=True)
class RetrievalSplitConfig:
    library_quotas: dict[str, int] = field(
        default_factory=lambda: DEFAULT_LIBRARY_QUOTAS.copy()
    )
    seed: int = 42
    excluded_libraries: tuple[str, ...] = (
        "enveda-np-examples",
    )


@dataclass(frozen=True)
class RetrievalSplit:
    dev: pd.DataFrame
    holdout: pd.DataFrame


def create_retrieval_split(
    metadata: pd.DataFrame,
    config: RetrievalSplitConfig = RetrievalSplitConfig(),
) -> RetrievalSplit:
    """
    Create deterministic, library-stratified retrieval benchmarks.

    library_quotas specifies the number of structures assigned to each
    query library in EACH split. For example:

        {"gnps": 400}

    means 400 GNPS-query structures in dev and another disjoint 400
    GNPS-query structures in holdout.

    Every selected structure must:

      * occur in the assigned query library;
      * occur in at least one other non-excluded library;
      * appear in exactly one of dev or holdout.

    The returned manifest contains spectrum counts so the benchmark can
    be inspected before it is frozen.
    """
    required_columns = {
        "inchikey14",
        "ingest_lib",
    }

    missing = required_columns - set(metadata.columns)

    if missing:
        raise ValueError(
            "metadata is missing required columns: "
            + ", ".join(sorted(missing))
        )

    if not config.library_quotas:
        raise ValueError(
            "library_quotas must not be empty"
        )

    for library, quota in config.library_quotas.items():
        if quota <= 0:
            raise ValueError(
                f"quota for {library!r} must be positive"
            )

    data = metadata[
        ["inchikey14", "ingest_lib"]
    ].copy()

    data = data.dropna(
        subset=[
            "inchikey14",
            "ingest_lib",
        ]
    )

    if config.excluded_libraries:
        data = data[
            ~data["ingest_lib"].isin(
                config.excluded_libraries
            )
        ]

    # Number of spectra for every structure/library pair.
    spectrum_counts = (
        data
        .groupby(
            [
                "inchikey14",
                "ingest_lib",
            ]
        )
        .size()
        .rename("spectrum_count")
        .reset_index()
    )

    # Libraries represented for each structure.
    structure_libraries = (
        spectrum_counts
        .groupby("inchikey14")
        ["ingest_lib"]
        .agg(tuple)
    )

    # Only structures with at least two usable libraries can form a
    # cross-library retrieval example.
    eligible_keys = set(
        structure_libraries[
            structure_libraries.map(len) >= 2
        ].index
    )

    spectrum_counts = spectrum_counts[
        spectrum_counts["inchikey14"].isin(
            eligible_keys
        )
    ].copy()

    rng = np.random.default_rng(
        config.seed
    )

    used_keys: set[str] = set()

    dev_rows: list[dict] = []
    holdout_rows: list[dict] = []

    def assign_for_library(
        library: str,
        quota: int,
    ) -> None:
        candidates = (
            spectrum_counts.loc[
                spectrum_counts["ingest_lib"]
                == library,
                "inchikey14",
            ]
            .drop_duplicates()
            .sort_values()
            .to_numpy()
        )

        # A structure can only be assigned to one query-library stratum
        # across both dev and holdout.
        candidates = np.array(
            [
                key
                for key in candidates
                if key not in used_keys
            ],
            dtype=object,
        )

        required = quota * 2

        if len(candidates) < required:
            raise ValueError(
                f"Not enough eligible structures for "
                f"{library!r}: need {required:,}, "
                f"found {len(candidates):,}"
            )

        selected = rng.permutation(
            candidates
        )[:required]

        dev_keys = selected[:quota]
        holdout_keys = selected[quota:]

        def make_row(
            key: str,
        ) -> dict:
            rows = spectrum_counts[
                spectrum_counts["inchikey14"]
                == key
            ]

            query_rows = rows[
                rows["ingest_lib"]
                == library
            ]

            reference_rows = rows[
                rows["ingest_lib"]
                != library
            ]

            if query_rows.empty:
                raise AssertionError(
                    f"{key} has no spectra in "
                    f"query library {library}"
                )

            if reference_rows.empty:
                raise AssertionError(
                    f"{key} has no reference spectra "
                    f"outside {library}"
                )

            return {
                "inchikey14": key,
                "query_library": library,
                "reference_library_count": (
                    reference_rows[
                        "ingest_lib"
                    ].nunique()
                ),
                "query_spectrum_count": int(
                    query_rows[
                        "spectrum_count"
                    ].sum()
                ),
                "reference_spectrum_count": int(
                    reference_rows[
                        "spectrum_count"
                    ].sum()
                ),
            }

        for key in dev_keys:
            dev_rows.append(
                make_row(str(key))
            )

        for key in holdout_keys:
            holdout_rows.append(
                make_row(str(key))
            )

        used_keys.update(
            str(key)
            for key in selected
        )

    # Process larger quotas first. This reduces the chance that a
    # structure shared by several libraries is consumed by a small
    # stratum before a large stratum can use it.
    quota_items = sorted(
        config.library_quotas.items(),
        key=lambda item: (
            -item[1],
            item[0],
        ),
    )

    for library, quota in quota_items:
        assign_for_library(
            library,
            quota,
        )

    columns = [
        "inchikey14",
        "query_library",
        "reference_library_count",
        "query_spectrum_count",
        "reference_spectrum_count",
    ]

    dev = (
        pd.DataFrame(
            dev_rows,
            columns=columns,
        )
        .sort_values(
            [
                "query_library",
                "inchikey14",
            ]
        )
        .reset_index(drop=True)
    )

    holdout = (
        pd.DataFrame(
            holdout_rows,
            columns=columns,
        )
        .sort_values(
            [
                "query_library",
                "inchikey14",
            ]
        )
        .reset_index(drop=True)
    )

    return RetrievalSplit(
        dev=dev,
        holdout=holdout,
    )