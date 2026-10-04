from __future__ import annotations

from collections import Counter
from pathlib import Path

import pandas as pd


COCONUT_PATH = Path(
    "data/external/coconut/"
    "coconut_csv-10-2026.csv"
)

CHUNK_SIZE = 100_000


def main() -> None:
    collection_counts: Counter[str] = Counter()

    rows = 0
    rows_without_collections = 0
    collection_combinations: Counter[
        tuple[str, ...]
    ] = Counter()

    for chunk in pd.read_csv(
        COCONUT_PATH,
        usecols=[
            "identifier",
            "collections",
        ],
        chunksize=CHUNK_SIZE,
    ):
        rows += len(chunk)

        for value in chunk["collections"]:
            if pd.isna(value):
                rows_without_collections += 1
                continue

            collections = tuple(
                sorted(
                    {
                        item.strip()
                        for item in str(value).split("|")
                        if item.strip()
                    }
                )
            )

            if not collections:
                rows_without_collections += 1
                continue

            collection_combinations[
                collections
            ] += 1

            for collection in collections:
                collection_counts[
                    collection
                ] += 1

        print(
            f"\rRows processed: {rows:,}",
            end="",
            flush=True,
        )

    print()
    print()

    print("COCONUT collection inventory")
    print("=" * 70)

    print(
        f"Rows:                     "
        f"{rows:,}"
    )

    print(
        f"Rows without collections: "
        f"{rows_without_collections:,}"
    )

    print(
        f"Unique collections:       "
        f"{len(collection_counts):,}"
    )

    print(
        f"Unique combinations:      "
        f"{len(collection_combinations):,}"
    )

    print()
    print("Collections")
    print("-" * 70)

    for collection, count in (
        collection_counts.most_common()
    ):
        print(
            f"{count:>10,}  "
            f"{count / rows:>8.3%}  "
            f"{collection}"
        )

    print()
    print("Most common collection combinations")
    print("-" * 70)

    for combination, count in (
        collection_combinations.most_common(30)
    ):
        print(
            f"{count:>10,}  "
            f"{count / rows:>8.3%}  "
            f"{' | '.join(combination)}"
        )


if __name__ == "__main__":
    main()