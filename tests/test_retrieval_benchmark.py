import pandas as pd
import pytest

from casmi26.data.retrieval_benchmark import (
    build_retrieval_benchmark,
)


def make_spectra() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "inchikey14": [
                "A",
                "A",
                "A",
                "B",
                "B",
                "C",
            ],
            "ingest_lib": [
                "gnps",
                "gnps",
                "riken",
                "mona",
                "massbank",
                "gnps",
            ],
            "spectrum": [
                "A-gnps-1",
                "A-gnps-2",
                "A-riken",
                "B-mona",
                "B-massbank",
                "C-gnps",
            ],
        }
    )


def make_manifest() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "inchikey14": [
                "A",
                "B",
            ],
            "query_library": [
                "gnps",
                "mona",
            ],
        }
    )


def test_selects_query_spectra():
    benchmark = build_retrieval_benchmark(
        make_spectra(),
        make_manifest(),
    )

    assert set(
        benchmark.queries["spectrum"]
    ) == {
        "A-gnps-1",
        "A-gnps-2",
        "B-mona",
    }


def test_query_spectra_are_removed_from_reference():
    benchmark = build_retrieval_benchmark(
        make_spectra(),
        make_manifest(),
    )

    reference = set(
        benchmark.reference["spectrum"]
    )

    assert "A-gnps-1" not in reference
    assert "A-gnps-2" not in reference
    assert "B-mona" not in reference


def test_other_library_truth_spectra_remain():
    benchmark = build_retrieval_benchmark(
        make_spectra(),
        make_manifest(),
    )

    reference = set(
        benchmark.reference["spectrum"]
    )

    assert "A-riken" in reference
    assert "B-massbank" in reference


def test_non_benchmark_structures_remain_as_distractors():
    benchmark = build_retrieval_benchmark(
        make_spectra(),
        make_manifest(),
    )

    assert (
        "C-gnps"
        in set(
            benchmark.reference["spectrum"]
        )
    )


def test_manifest_structures_must_have_queries():
    spectra = make_spectra()

    manifest = pd.DataFrame(
        {
            "inchikey14": ["A"],
            "query_library": [
                "does-not-exist"
            ],
        }
    )

    with pytest.raises(
        ValueError,
        match="No query spectra found",
    ):
        build_retrieval_benchmark(
            spectra,
            manifest,
        )


def test_rejects_duplicate_manifest_structures():
    manifest = pd.DataFrame(
        {
            "inchikey14": [
                "A",
                "A",
            ],
            "query_library": [
                "gnps",
                "riken",
            ],
        }
    )

    with pytest.raises(
        ValueError,
        match="duplicate structures",
    ):
        build_retrieval_benchmark(
            make_spectra(),
            manifest,
        )


def test_rejects_missing_spectra_columns():
    spectra = pd.DataFrame(
        {
            "inchikey14": ["A"],
        }
    )

    with pytest.raises(
        ValueError,
        match="spectra is missing required columns",
    ):
        build_retrieval_benchmark(
            spectra,
            make_manifest(),
        )


def test_rejects_missing_manifest_columns():
    manifest = pd.DataFrame(
        {
            "inchikey14": ["A"],
        }
    )

    with pytest.raises(
        ValueError,
        match="manifest is missing required columns",
    ):
        build_retrieval_benchmark(
            make_spectra(),
            manifest,
        )