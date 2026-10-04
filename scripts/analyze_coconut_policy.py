from __future__ import annotations

from collections import Counter
from pathlib import Path

import pandas as pd


COCONUT_PATH = Path(
    "data/external/coconut/"
    "coconut_csv-10-2026.csv"
)

CHUNK_SIZE = 100_000


# These are deliberately conservative.
#
# APPROVED means we have affirmative evidence that the source's
# licensing permits the kind of reuse we need.
#
# EXCLUDED means we've found restrictions incompatible with our
# competition policy.
#
# Everything else remains UNKNOWN until reviewed.

APPROVED = {
    "Wikidata Natural Products",
    "ChEBI NPs",
    "GNPS (Global Natural Products Social Molecular Networking)",
}

EXCLUDED = {
    "FooDB",
    "CMNPD",
    "Supernatural3",
    "NPASS",
    "NPAtlas",
    "DrugBankNP",
}

REVIEW_REQUIRED = {
    "Super Natural II",
    "ZINC NP",
    "InterBioScreen Ltd",
    "TCMDB-Taiwan (Traditional Chinese Medicine database)",
    "NPEdia",
    "UNPD (Universal Natural Products Database)",
    "CMAUP (cCollective molecular activities of useful plants)",
    "KNApSaCK",
    "PubChem NPs",
    "ChEMBL NPs",
}


def parse_collections(value: object) -> set[str]:
    if pd.isna(value):
        return set()

    return {
        item.strip()
        for item in str(value).split("|")
        if item.strip()
    }


def main() -> None:
    rows = 0

    disposition_counts: Counter[str] = Counter()
    unknown_counts: Counter[str] = Counter()

    for chunk in pd.read_csv(
        COCONUT_PATH,
        usecols=["identifier", "collections"],
        chunksize=CHUNK_SIZE,
    ):
        rows += len(chunk)

        for value in chunk["collections"]:
            collections = parse_collections(value)

            approved = collections & APPROVED
            excluded = collections & EXCLUDED
            review_required = collections & REVIEW_REQUIRED

            unresolved = (
                    collections
                    - APPROVED
                    - EXCLUDED
                    - REVIEW_REQUIRED
            )

            if approved:
                disposition = "approved"
            elif unresolved:
                disposition = "needs_review"

                for collection in unresolved:
                    unknown_counts[collection] += 1
            elif review_required:
                disposition = "review_required"
            elif excluded:
                disposition = "excluded"
            else:
                disposition = "no_provenance"

            disposition_counts[disposition] += 1

        print(
            f"\rRows processed: {rows:,}",
            end="",
            flush=True,
        )

    print("\n")
    print("COCONUT provisional policy")
    print("=" * 70)

    for disposition in [
        "approved",
        "needs_review",
        "excluded",
        "no_provenance",
    ]:
        count = disposition_counts[disposition]

        print(
            f"{disposition:20s}"
            f"{count:>10,}  "
            f"{count / rows:>8.3%}"
        )

    print()
    print("Most important unresolved collections")
    print("-" * 70)

    for collection, count in unknown_counts.most_common():
        print(
            f"{count:>10,}  "
            f"{count / rows:>8.3%}  "
            f"{collection}"
        )


if __name__ == "__main__":
    main()