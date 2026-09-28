from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


@dataclass(frozen=True)
class SplitConfig:
    validation_fraction: float = 0.10
    seed: int = 42
    domain_library: str = "enveda-np-examples"


@dataclass(frozen=True)
class SplitManifest:
    train: pd.DataFrame
    validation: pd.DataFrame
    domain_validation: pd.DataFrame


def create_split_manifest(
    train_path: Path,
    config: SplitConfig = SplitConfig(),
) -> SplitManifest:
    """
    Create reproducible molecule-level train/validation manifests.

    The ordinary validation split is structure-disjoint: every spectrum for a
    given InChIKey14 is assigned to either training or validation.

    The domain-validation set contains spectra from the configured library.
    Its structures are intentionally allowed to occur in the training set
    through other libraries.
    """
    metadata = pd.read_parquet(
        train_path,
        columns=["inchikey14", "ingest_lib"],
    )

    if metadata["inchikey14"].isna().any():
        raise ValueError("Training data contains missing inchikey14 values")

    unique_structures = np.sort(metadata["inchikey14"].unique())

    rng = np.random.default_rng(config.seed)
    shuffled = rng.permutation(unique_structures)

    validation_count = round(
        len(unique_structures) * config.validation_fraction
    )

    validation_keys = set(shuffled[:validation_count])
    train_keys = set(shuffled[validation_count:])

    train_manifest = pd.DataFrame(
        {"inchikey14": sorted(train_keys)}
    )

    validation_manifest = pd.DataFrame(
        {"inchikey14": sorted(validation_keys)}
    )

    domain_validation = (
        metadata.loc[
            metadata["ingest_lib"] == config.domain_library,
            ["inchikey14", "ingest_lib"],
        ]
        .drop_duplicates()
        .sort_values("inchikey14")
        .reset_index(drop=True)
    )

    return SplitManifest(
        train=train_manifest,
        validation=validation_manifest,
        domain_validation=domain_validation,
    )


def write_split_manifest(
    manifest: SplitManifest,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest.train.to_parquet(
        output_dir / "train_structures.parquet",
        index=False,
    )
    manifest.validation.to_parquet(
        output_dir / "validation_structures.parquet",
        index=False,
    )
    manifest.domain_validation.to_parquet(
        output_dir / "domain_validation_structures.parquet",
        index=False,
    )