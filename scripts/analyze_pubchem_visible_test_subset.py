from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent

HYPOTHESES_PATH = (
    PROJECT_ROOT
    / "data/processed/test/"
    "multispectrum_formula_hypotheses.parquet"
)

BULK_ROOT = (
    PROJECT_ROOT
    / "data/processed/pubchem_bulk"
)

CHUNK_DIR = (
    BULK_ROOT
    / "chunks"
)

SUMMARY_PATH = (
    PROJECT_ROOT
    / "data/processed/diagnostics/"
    "pubchem_visible_test_subset_summary.json"
)

SUBSET_PATH = (
    PROJECT_ROOT
    / "data/processed/diagnostics/"
    "pubchem_visible_test_candidates.parquet"
)


def parquet_bytes(
    path: Path,
) -> int:
    return (
        path.stat().st_size
        if path.exists()
        else 0
    )


def human_bytes(
    value: int,
) -> str:
    units = [
        "B",
        "KB",
        "MB",
        "GB",
        "TB",
    ]

    size = float(
        value
    )

    for unit in units:
        if size < 1024.0 or unit == units[-1]:
            return f"{size:.2f} {unit}"

        size /= 1024.0

    return f"{size:.2f} TB"


def main() -> None:
    if not HYPOTHESES_PATH.exists():
        raise FileNotFoundError(
            "Visible-test formula hypotheses are missing. "
            "Run scripts/build_test_multispectrum_formula_hypotheses.py "
            "first."
        )

    chunk_paths = sorted(
        CHUNK_DIR.glob(
            "Compound_*.parquet"
        )
    )

    if not chunk_paths:
        raise FileNotFoundError(
            "No PubChem bulk chunk results found under "
            f"{CHUNK_DIR}"
        )

    hypotheses = pd.read_parquet(
        HYPOTHESES_PATH
    )

    if (
        "candidate_formula"
        not in hypotheses.columns
    ):
        raise ValueError(
            "Test hypotheses are missing candidate_formula"
        )

    visible_formulas = {
        str(value)
        for value in hypotheses[
            "candidate_formula"
        ].dropna()
    }

    if not visible_formulas:
        raise RuntimeError(
            "Visible test generated no candidate formulas"
        )

    full_rows = 0
    full_structures = set()

    subset_frames = []

    chunk_bytes = 0

    print(
        "Scanning PubChem bulk chunk outputs..."
    )

    for number, path in enumerate(
        chunk_paths,
        start=1,
    ):
        frame = pd.read_parquet(
            path
        )

        chunk_bytes += parquet_bytes(
            path
        )

        full_rows += len(
            frame
        )

        if (
            "inchikey14"
            in frame.columns
        ):
            full_structures.update(
                frame[
                    "inchikey14"
                ]
                .dropna()
                .astype(str)
            )

        selected = frame.loc[
            frame[
                "molecular_formula"
            ]
            .astype(str)
            .isin(
                visible_formulas
            )
        ]

        if not selected.empty:
            subset_frames.append(
                selected
            )

        if (
            number % 25 == 0
            or number == len(
                chunk_paths
            )
        ):
            print(
                f"  {number}/{len(chunk_paths)} chunks"
            )

    if subset_frames:
        subset = pd.concat(
            subset_frames,
            ignore_index=True,
        )

        subset = (
            subset
            .drop_duplicates(
                subset=[
                    "molecular_formula",
                    "inchikey14",
                ]
            )
            .sort_values(
                [
                    "molecular_formula",
                    "inchikey14",
                ]
            )
            .reset_index(
                drop=True
            )
        )

    else:
        subset = pd.DataFrame(
            columns=[
                "molecular_formula",
                "pubchem_cid",
                "full_inchikey",
                "inchikey14",
                "smiles",
            ]
        )

    SUBSET_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    subset.to_parquet(
        SUBSET_PATH,
        index=False,
        compression="zstd",
    )

    subset_bytes = parquet_bytes(
        SUBSET_PATH
    )

    per_formula = (
        subset.groupby(
            "molecular_formula"
        )[
            "inchikey14"
        ]
        .nunique()
        .rename(
            "candidate_count"
        )
        .reset_index()
    )

    requested = pd.DataFrame(
        {
            "molecular_formula":
                sorted(
                    visible_formulas
                ),
        }
    )

    per_formula = requested.merge(
        per_formula,
        on="molecular_formula",
        how="left",
        validate="one_to_one",
    )

    per_formula[
        "candidate_count"
    ] = (
        per_formula[
            "candidate_count"
        ]
        .fillna(0)
        .astype(int)
    )

    counts = per_formula[
        "candidate_count"
    ]

    summary = {
        "bulk_chunk_count":
            len(chunk_paths),
        "bulk_candidate_rows":
            int(full_rows),
        "bulk_unique_structures":
            len(full_structures),
        "bulk_chunk_parquet_bytes":
            int(chunk_bytes),
        "bulk_chunk_parquet_human":
            human_bytes(
                chunk_bytes
            ),
        "visible_test_molecules":
            int(
                hypotheses[
                    "molecule_id"
                ].nunique()
            )
            if "molecule_id"
            in hypotheses.columns
            else None,
        "visible_test_hypotheses":
            len(
                hypotheses
            ),
        "visible_test_unique_formulas":
            len(
                visible_formulas
            ),
        "visible_test_candidate_rows":
            len(
                subset
            ),
        "visible_test_unique_structures":
            int(
                subset[
                    "inchikey14"
                ].nunique()
            )
            if not subset.empty
            else 0,
        "visible_test_subset_parquet_bytes":
            int(
                subset_bytes
            ),
        "visible_test_subset_parquet_human":
            human_bytes(
                subset_bytes
            ),
        "formulas_with_zero_candidates":
            int(
                (
                    counts
                    == 0
                ).sum()
            ),
        "candidate_count_mean":
            float(
                counts.mean()
            )
            if len(counts)
            else 0.0,
        "candidate_count_median":
            float(
                counts.median()
            )
            if len(counts)
            else 0.0,
        "candidate_count_p90":
            float(
                counts.quantile(
                    0.90
                )
            )
            if len(counts)
            else 0.0,
        "candidate_count_p95":
            float(
                counts.quantile(
                    0.95
                )
            )
            if len(counts)
            else 0.0,
        "candidate_count_p99":
            float(
                counts.quantile(
                    0.99
                )
            )
            if len(counts)
            else 0.0,
        "candidate_count_max":
            int(
                counts.max()
            )
            if len(counts)
            else 0,
    }

    SUMMARY_PATH.write_text(
        json.dumps(
            summary,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    print()
    print(
        "PubChem visible-test subset"
    )
    print(
        "==========================="
    )
    print(
        "Bulk candidate rows:",
        f"{summary['bulk_candidate_rows']:,}",
    )
    print(
        "Bulk unique structures:",
        f"{summary['bulk_unique_structures']:,}",
    )
    print(
        "Bulk retained parquet size:",
        summary[
            "bulk_chunk_parquet_human"
        ],
    )

    print()
    print(
        "Visible-test molecules:",
        summary[
            "visible_test_molecules"
        ],
    )
    print(
        "Visible-test formula hypotheses:",
        f"{summary['visible_test_hypotheses']:,}",
    )
    print(
        "Visible-test unique formulas:",
        f"{summary['visible_test_unique_formulas']:,}",
    )
    print(
        "Visible-test PubChem candidate rows:",
        f"{summary['visible_test_candidate_rows']:,}",
    )
    print(
        "Visible-test unique structures:",
        f"{summary['visible_test_unique_structures']:,}",
    )
    print(
        "Visible-test subset parquet size:",
        summary[
            "visible_test_subset_parquet_human"
        ],
    )

    print()
    print(
        "Formulas with zero PubChem candidates:",
        summary[
            "formulas_with_zero_candidates"
        ],
    )
    print(
        "Candidates/formula mean:",
        f"{summary['candidate_count_mean']:.1f}",
    )
    print(
        "Candidates/formula median:",
        f"{summary['candidate_count_median']:.1f}",
    )
    print(
        "Candidates/formula p90:",
        f"{summary['candidate_count_p90']:.1f}",
    )
    print(
        "Candidates/formula p95:",
        f"{summary['candidate_count_p95']:.1f}",
    )
    print(
        "Candidates/formula p99:",
        f"{summary['candidate_count_p99']:.1f}",
    )
    print(
        "Candidates/formula max:",
        f"{summary['candidate_count_max']:,}",
    )

    print()
    print(
        f"Wrote {SUBSET_PATH}"
    )
    print(
        f"Wrote {SUMMARY_PATH}"
    )


if __name__ == "__main__":
    main()
