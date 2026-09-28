from pathlib import Path

import pandas as pd

from casmi26.data.split import SplitConfig, create_split_manifest


def make_dataset(tmp_path: Path) -> Path:
    path = tmp_path / "train.parquet"

    pd.DataFrame(
        {
            "inchikey14": [
                "A",
                "A",
                "B",
                "B",
                "C",
                "D",
                "E",
                "E",
                "F",
                "G",
            ],
            "ingest_lib": [
                "library-a",
                "library-b",
                "library-a",
                "enveda-np-examples",
                "library-a",
                "library-a",
                "library-a",
                "enveda-np-examples",
                "library-a",
                "library-a",
            ],
        }
    ).to_parquet(path, index=False)

    return path


def test_train_and_validation_are_structure_disjoint(tmp_path: Path):
    path = make_dataset(tmp_path)

    manifest = create_split_manifest(
        path,
        SplitConfig(validation_fraction=0.3, seed=42),
    )

    train_keys = set(manifest.train["inchikey14"])
    validation_keys = set(manifest.validation["inchikey14"])

    assert train_keys.isdisjoint(validation_keys)


def test_every_structure_is_assigned(tmp_path: Path):
    path = make_dataset(tmp_path)

    manifest = create_split_manifest(
        path,
        SplitConfig(validation_fraction=0.3, seed=42),
    )

    assigned = set(manifest.train["inchikey14"]) | set(
        manifest.validation["inchikey14"]
    )

    assert assigned == {"A", "B", "C", "D", "E", "F", "G"}


def test_split_is_reproducible(tmp_path: Path):
    path = make_dataset(tmp_path)

    first = create_split_manifest(
        path,
        SplitConfig(validation_fraction=0.3, seed=42),
    )
    second = create_split_manifest(
        path,
        SplitConfig(validation_fraction=0.3, seed=42),
    )

    pd.testing.assert_frame_equal(first.train, second.train)
    pd.testing.assert_frame_equal(
        first.validation,
        second.validation,
    )


def test_domain_validation_contains_configured_library(tmp_path: Path):
    path = make_dataset(tmp_path)

    manifest = create_split_manifest(path)

    assert set(manifest.domain_validation["inchikey14"]) == {"B", "E"}


def test_domain_structures_may_also_be_in_training(tmp_path: Path):
    path = make_dataset(tmp_path)

    manifest = create_split_manifest(
        path,
        SplitConfig(validation_fraction=0.1, seed=42),
    )

    domain_keys = set(manifest.domain_validation["inchikey14"])
    train_keys = set(manifest.train["inchikey14"])

    assert domain_keys & train_keys