from pathlib import Path

import numpy as np
import pandas as pd

from casmi26.retrieval.index import (
    load_retrieval_index,
)
from casmi26.retrieval.indexed import (
    search_structure_blockwise,
)


INDEX_DIR = Path(
    "data/processed/retrieval_index"
)

MANIFEST_PATH = Path(
    "data/processed/splits/retrieval_dev.parquet"
)

structure_count = 10
BLOCK_SIZE = 25_000

def main() -> None:
    print("Loading retrieval index...")

    metadata, vectors = load_retrieval_index(
        INDEX_DIR
    )

    manifest = pd.read_parquet(
        MANIFEST_PATH
    )

    print(
        f"Index spectra:       {len(metadata):,}"
    )
    print(
        f"Dev structures:      {len(manifest):,}"
    )

    # ---------------------------------------------------------
    # Build the reference pool for the COMPLETE dev benchmark.
    #
    # For each dev structure, remove spectra from its assigned
    # query library. Other libraries for that structure remain
    # available as legitimate retrieval references.
    # ---------------------------------------------------------

    query_pairs = set(
        zip(
            manifest["inchikey14"].astype(str),
            manifest["query_library"].astype(str),
            strict=True,
        )
    )

    print()
    print(
        "Building dev reference mask..."
    )

    reference_mask = np.fromiter(
        (
            (
                str(key),
                str(library),
            )
            not in query_pairs
            for key, library in zip(
                metadata["inchikey14"],
                metadata["ingest_lib"],
                strict=True,
            )
        ),
        dtype=bool,
        count=len(metadata),
    )

    reference_indices = np.flatnonzero(
        reference_mask
    )

    reference_metadata = (
        metadata.iloc[
            reference_indices
        ]
        .reset_index(drop=True)
    )

    reference_vectors = vectors[
        reference_indices
    ]

    print(
        f"Reference spectra:   "
        f"{len(reference_metadata):,}"
    )

    removed = (
        len(metadata)
        - len(reference_metadata)
    )

    print(
        f"Query spectra removed: {removed:,}"
    )

    # ---------------------------------------------------------
    # Score only the first 10 structures.
    # ---------------------------------------------------------

    smoke_manifest = (
        manifest
        .groupby(
            "query_library",
            sort=True,
            group_keys=False,
        )
        .head(2)
        .reset_index(drop=True)
    )

    structure_count = len(
        smoke_manifest
    )

    print()
    print("Smoke-test query libraries")
    print("--------------------------")
    print(
        smoke_manifest[
            "query_library"
        ]
        .value_counts()
        .sort_index()
    )

    reciprocal_ranks = []

    print()
    print(
        "Running retrieval smoke test..."
    )
    print()

    for position, row in enumerate(
        smoke_manifest.itertuples(
            index=False
        ),
        start=1,
    ):
        truth_key = str(
            row.inchikey14
        )

        query_library = str(
            row.query_library
        )

        query_mask = (
            (
                metadata[
                    "inchikey14"
                ].astype(str)
                == truth_key
            )
            & (
                metadata[
                    "ingest_lib"
                ].astype(str)
                == query_library
            )
        ).to_numpy()

        query_indices = np.flatnonzero(
            query_mask
        )

        if len(query_indices) == 0:
            raise RuntimeError(
                f"No query spectra found for "
                f"{truth_key} / {query_library}"
            )

        query_vectors = vectors[
            query_indices
        ]

        query_modes = (
            metadata.iloc[
                query_indices
            ][
                "ionization_mode"
            ]
            .astype(str)
            .to_numpy()
        )

        scores = search_structure_blockwise(
            query_vectors=query_vectors,
            query_modes=query_modes,
            reference_vectors=reference_vectors,
            reference_metadata=reference_metadata,
            block_size=BLOCK_SIZE,
        )

        ranked = sorted(
            scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )

        rank = next(
            (
                rank
                for rank, (key, _)
                in enumerate(
                    ranked,
                    start=1,
                )
                if key == truth_key
            ),
            None,
        )

        if (
            rank is not None
            and rank <= 25
        ):
            reciprocal_rank = (
                1.0 / rank
            )
        else:
            reciprocal_rank = 0.0

        reciprocal_ranks.append(
            reciprocal_rank
        )

        top_key = (
            ranked[0][0]
            if ranked
            else None
        )

        print(
            f"{position:>2}/"
            f"{structure_count}  "
            f"{truth_key}  "
            f"library={query_library:<12} "
            f"spectra={len(query_indices):>3}  "
            f"rank={str(rank):>5}  "
            f"top={top_key}"
        )

    mrr = float(
        np.mean(reciprocal_ranks)
    )

    top1 = sum(
        rr == 1.0
        for rr in reciprocal_ranks
    )

    top25 = sum(
        rr > 0.0
        for rr in reciprocal_ranks
    )

    print(
        f"Structures: {structure_count}"
    )
    print(
        f"MRR@25:     {mrr:.4f}"
    )
    print(
        f"Top-1:      "
        f"{top1}/{structure_count}"
    )
    print(
        f"Top-25:     "
        f"{top25}/{structure_count}"
    )

if __name__ == "__main__":
    main()