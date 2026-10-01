from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class RetrievalBenchmark:
    queries: pd.DataFrame
    reference: pd.DataFrame


def build_retrieval_benchmark(
    spectra: pd.DataFrame,
    manifest: pd.DataFrame,
) -> RetrievalBenchmark:
    """
    Construct query and reference datasets from a retrieval manifest.

    For each structure in the manifest:

      * spectra from query_library become queries;
      * spectra for that same structure from query_library are excluded
        from the reference set;
      * spectra for that structure from other libraries remain available
        as correct reference candidates;
      * spectra for non-query structures remain available as distractors.

    This prevents direct query-spectrum leakage while preserving the
    full candidate-library retrieval problem.
    """
    required_spectra_columns = {
        "inchikey14",
        "ingest_lib",
    }

    required_manifest_columns = {
        "inchikey14",
        "query_library",
    }

    missing_spectra = (
        required_spectra_columns
        - set(spectra.columns)
    )

    if missing_spectra:
        raise ValueError(
            "spectra is missing required columns: "
            + ", ".join(
                sorted(missing_spectra)
            )
        )

    missing_manifest = (
        required_manifest_columns
        - set(manifest.columns)
    )

    if missing_manifest:
        raise ValueError(
            "manifest is missing required columns: "
            + ", ".join(
                sorted(missing_manifest)
            )
        )

    if manifest["inchikey14"].duplicated().any():
        raise ValueError(
            "manifest contains duplicate structures"
        )

    assignments = manifest[
        [
            "inchikey14",
            "query_library",
        ]
    ].copy()

    # Attach the query-library assignment to every spectrum belonging
    # to a benchmark structure.
    annotated = spectra.merge(
        assignments,
        on="inchikey14",
        how="left",
        validate="many_to_one",
    )

    is_benchmark_structure = (
        annotated["query_library"].notna()
    )

    is_query_library = (
        annotated["ingest_lib"]
        == annotated["query_library"]
    )

    query_mask = (
        is_benchmark_structure
        & is_query_library
    )

    queries = (
        annotated.loc[
            query_mask
        ]
        .drop(
            columns=["query_library"]
        )
        .copy()
    )

    # Every query spectrum is removed from the candidate/reference
    # collection. Everything else remains.
    reference = (
        annotated.loc[
            ~query_mask
        ]
        .drop(
            columns=["query_library"]
        )
        .copy()
    )

    # Verify that every manifest structure actually produced a query.
    query_keys = set(
        queries["inchikey14"]
    )

    manifest_keys = set(
        manifest["inchikey14"]
    )

    missing_queries = (
        manifest_keys - query_keys
    )

    if missing_queries:
        raise ValueError(
            "No query spectra found for "
            f"{len(missing_queries)} manifest structures"
        )

    return RetrievalBenchmark(
        queries=queries,
        reference=reference,
    )