from pathlib import Path

import pandas as pd

from casmi26.data.retrieval_split import (
    RetrievalSplitConfig,
    create_retrieval_split,
)


TRAIN_PATH = Path(
    "data/raw/train.parquet"
)

OUTPUT_DIR = Path(
    "data/processed/splits"
)

DEV_PATH = OUTPUT_DIR / "retrieval_dev.parquet"
HOLDOUT_PATH = (
    OUTPUT_DIR / "retrieval_holdout.parquet"
)


def main() -> None:
    print(
        f"Reading metadata from {TRAIN_PATH}..."
    )

    metadata = pd.read_parquet(
        TRAIN_PATH,
        columns=[
            "inchikey14",
            "ingest_lib",
        ],
    )

    config = RetrievalSplitConfig()

    print(
        "Creating retrieval splits..."
    )

    split = create_retrieval_split(
        metadata,
        config,
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    split.dev.to_parquet(
        DEV_PATH,
        index=False,
    )

    split.holdout.to_parquet(
        HOLDOUT_PATH,
        index=False,
    )

    print()
    print(
        f"Dev:     {len(split.dev):,} structures"
    )
    print(
        f"Holdout: {len(split.holdout):,} structures"
    )

    print()
    print("Dev query libraries")
    print("-------------------")
    print(
        split.dev[
            "query_library"
        ]
        .value_counts()
        .to_string()
    )

    print()
    print("Holdout query libraries")
    print("-----------------------")
    print(
        split.holdout[
            "query_library"
        ]
        .value_counts()
        .to_string()
    )

    overlap = set(
        split.dev["inchikey14"]
    ) & set(
        split.holdout["inchikey14"]
    )

    if overlap:
        raise RuntimeError(
            "Dev and holdout structures overlap"
        )

    print()
    print(
        f"Wrote {DEV_PATH}"
    )
    print(
        f"Wrote {HOLDOUT_PATH}"
    )


if __name__ == "__main__":
    main()