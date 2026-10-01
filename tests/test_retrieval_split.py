import pandas as pd
import pytest

from casmi26.data.retrieval_split import (
    RetrievalSplitConfig,
    create_retrieval_split,
)


def make_metadata() -> pd.DataFrame:
    rows = []

    # Six structures available in lib1 plus another reference library.
    for i in range(6):
        key = f"A{i}"

        rows.extend(
            [
                {
                    "inchikey14": key,
                    "ingest_lib": "lib1",
                },
                {
                    "inchikey14": key,
                    "ingest_lib": "ref1",
                },
            ]
        )

    # Six different structures available in lib2 plus another
    # reference library.
    for i in range(6):
        key = f"B{i}"

        rows.extend(
            [
                {
                    "inchikey14": key,
                    "ingest_lib": "lib2",
                },
                {
                    "inchikey14": key,
                    "ingest_lib": "ref2",
                },
            ]
        )

    # Duplicate spectra for one structure/library pair so we can test
    # spectrum counts.
    rows.extend(
        [
            {
                "inchikey14": "A0",
                "ingest_lib": "lib1",
            },
            {
                "inchikey14": "A0",
                "ingest_lib": "lib1",
            },
            {
                "inchikey14": "A0",
                "ingest_lib": "ref1",
            },
        ]
    )

    # Single-library structure: must never be eligible.
    rows.append(
        {
            "inchikey14": "SINGLE",
            "ingest_lib": "lib1",
        }
    )

    # This looks multi-library until the excluded library is removed.
    rows.extend(
        [
            {
                "inchikey14": "EXCLUDED",
                "ingest_lib": "lib1",
            },
            {
                "inchikey14": "EXCLUDED",
                "ingest_lib": "enveda-np-examples",
            },
        ]
    )

    return pd.DataFrame(rows)


def make_config(
    seed: int = 42,
) -> RetrievalSplitConfig:
    return RetrievalSplitConfig(
        library_quotas={
            "lib1": 2,
            "lib2": 2,
        },
        seed=seed,
    )


def test_library_quotas_are_satisfied():
    split = create_retrieval_split(
        make_metadata(),
        make_config(),
    )

    expected = {
        "lib1": 2,
        "lib2": 2,
    }

    assert (
        split.dev[
            "query_library"
        ]
        .value_counts()
        .to_dict()
        == expected
    )

    assert (
        split.holdout[
            "query_library"
        ]
        .value_counts()
        .to_dict()
        == expected
    )


def test_dev_and_holdout_are_disjoint():
    split = create_retrieval_split(
        make_metadata(),
        make_config(),
    )

    dev_keys = set(
        split.dev["inchikey14"]
    )

    holdout_keys = set(
        split.holdout["inchikey14"]
    )

    assert dev_keys.isdisjoint(
        holdout_keys
    )


def test_structure_is_used_only_once():
    split = create_retrieval_split(
        make_metadata(),
        make_config(),
    )

    manifest = pd.concat(
        [
            split.dev,
            split.holdout,
        ],
        ignore_index=True,
    )

    assert (
        manifest["inchikey14"]
        .is_unique
    )


def test_query_library_contains_structure():
    metadata = make_metadata()

    split = create_retrieval_split(
        metadata,
        make_config(),
    )

    manifest = pd.concat(
        [
            split.dev,
            split.holdout,
        ]
    )

    for row in manifest.itertuples():
        matching = metadata[
            (
                metadata["inchikey14"]
                == row.inchikey14
            )
            & (
                metadata["ingest_lib"]
                == row.query_library
            )
        ]

        assert not matching.empty


def test_every_structure_has_reference_library():
    split = create_retrieval_split(
        make_metadata(),
        make_config(),
    )

    manifest = pd.concat(
        [
            split.dev,
            split.holdout,
        ]
    )

    assert (
        manifest[
            "reference_library_count"
        ]
        >= 1
    ).all()

    assert (
        manifest[
            "reference_spectrum_count"
        ]
        >= 1
    ).all()


def test_excluded_library_does_not_create_eligibility():
    split = create_retrieval_split(
        make_metadata(),
        make_config(),
    )

    selected = set(
        split.dev["inchikey14"]
    ) | set(
        split.holdout["inchikey14"]
    )

    assert "EXCLUDED" not in selected


def test_single_library_structure_is_not_selected():
    split = create_retrieval_split(
        make_metadata(),
        make_config(),
    )

    selected = set(
        split.dev["inchikey14"]
    ) | set(
        split.holdout["inchikey14"]
    )

    assert "SINGLE" not in selected


def test_spectrum_counts_are_correct():
    metadata = make_metadata()

    config = RetrievalSplitConfig(
        library_quotas={
            "lib1": 3,
        },
        seed=42,
    )

    split = create_retrieval_split(
        metadata,
        config,
    )

    manifest = pd.concat(
        [
            split.dev,
            split.holdout,
        ]
    )

    # With quota 3 and six eligible A structures, all A structures
    # are selected. Therefore A0 must appear in one of the manifests.
    row = manifest[
        manifest["inchikey14"]
        == "A0"
    ].iloc[0]

    assert row[
        "query_spectrum_count"
    ] == 3

    assert row[
        "reference_spectrum_count"
    ] == 2


def test_split_is_deterministic():
    metadata = make_metadata()
    config = make_config(seed=123)

    first = create_retrieval_split(
        metadata,
        config,
    )

    second = create_retrieval_split(
        metadata.sample(
            frac=1,
            random_state=999,
        ),
        config,
    )

    pd.testing.assert_frame_equal(
        first.dev,
        second.dev,
    )

    pd.testing.assert_frame_equal(
        first.holdout,
        second.holdout,
    )


def test_rejects_insufficient_library_population():
    metadata = make_metadata()

    with pytest.raises(
        ValueError,
        match="Not enough eligible structures",
    ):
        create_retrieval_split(
            metadata,
            RetrievalSplitConfig(
                library_quotas={
                    "lib1": 100,
                }
            ),
        )


def test_rejects_missing_columns():
    metadata = pd.DataFrame(
        {
            "inchikey14": [
                "A",
                "B",
            ]
        }
    )

    with pytest.raises(
        ValueError,
        match="missing required columns",
    ):
        create_retrieval_split(
            metadata
        )


def test_rejects_empty_quotas():
    with pytest.raises(
        ValueError,
        match="library_quotas must not be empty",
    ):
        create_retrieval_split(
            make_metadata(),
            RetrievalSplitConfig(
                library_quotas={}
            ),
        )


def test_rejects_nonpositive_quota():
    with pytest.raises(
        ValueError,
        match="must be positive",
    ):
        create_retrieval_split(
            make_metadata(),
            RetrievalSplitConfig(
                library_quotas={
                    "lib1": 0,
                }
            ),
        )