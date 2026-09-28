from pathlib import Path

from casmi26.data.split import (
    SplitConfig,
    create_split_manifest,
    write_split_manifest,
)

manifest = create_split_manifest(
    Path("data/raw/train.parquet"),
    SplitConfig(
        validation_fraction=0.10,
        seed=42,
    ),
)

write_split_manifest(
    manifest,
    Path("data/processed/splits"),
)

print(f"Training structures:   {len(manifest.train):,}")
print(f"Validation structures: {len(manifest.validation):,}")
print(
    f"Domain structures:     "
    f"{len(manifest.domain_validation):,}"
)