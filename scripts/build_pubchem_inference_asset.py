from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent

SOURCE_CANDIDATES = (
    PROJECT_ROOT
    / "data/processed/diagnostics/"
    "pubchem_formula_index_candidates.parquet"
)

SOURCE_BURDEN = (
    PROJECT_ROOT
    / "data/processed/diagnostics/"
    "pubchem_formula_index_candidates_burden.parquet"
)

SOURCE_SUMMARY = (
    PROJECT_ROOT
    / "data/processed/diagnostics/"
    "pubchem_formula_index_candidates_summary.json"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "data/processed/pubchem_formula_index"
)

CANDIDATE_DIR = OUTPUT_ROOT / "candidates"
BURDEN_PATH = OUTPUT_ROOT / "burden.parquet"
FORMULA_BUCKET_PATH = OUTPUT_ROOT / "formula_buckets.parquet"
MANIFEST_PATH = OUTPUT_ROOT / "manifest.json"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Convert the full PubChem formula-index catalog into a "
            "partitioned offline inference asset."
        )
    )

    parser.add_argument(
        "--buckets",
        type=int,
        default=256,
    )

    return parser.parse_args()


def bucket_for_formula(
    formula: str,
    bucket_count: int,
) -> int:
    digest = hashlib.sha256(
        str(formula).encode("utf-8")
    ).digest()

    value = int.from_bytes(
        digest[:8],
        byteorder="big",
        signed=False,
    )

    return value % bucket_count


def main() -> None:
    args = parse_args()

    if args.buckets <= 0:
        raise ValueError(
            "--buckets must be positive"
        )

    required = [
        SOURCE_CANDIDATES,
        SOURCE_BURDEN,
        SOURCE_SUMMARY,
    ]

    missing = [
        path
        for path in required
        if not path.exists()
    ]

    if missing:
        raise FileNotFoundError(
            "Missing PubChem formula-index diagnostics. "
            "Run scripts/build_pubchem_formula_candidates.py "
            "--source formula-index first. Missing: "
            + ", ".join(
                str(path)
                for path in missing
            )
        )

    burden = pd.read_parquet(
        SOURCE_BURDEN
    )

    required_burden = {
        "molecular_formula",
        "candidate_count",
        "hit_max_records",
        "fetch_failed",
    }

    missing_burden = (
        required_burden
        - set(burden.columns)
    )

    if missing_burden:
        raise ValueError(
            "PubChem burden file is missing columns: "
            f"{sorted(missing_burden)}"
        )

    failed = burden[
        "fetch_failed"
    ].fillna(False).astype(bool)

    if failed.any():
        examples = burden.loc[
            failed,
            "molecular_formula",
        ].astype(str).head(20).tolist()

        raise RuntimeError(
            "PubChem formula-index fetch is incomplete. "
            f"{int(failed.sum())} formulas failed. "
            f"Examples: {examples}"
        )

    formula_buckets = (
        burden[
            [
                "molecular_formula",
            ]
        ]
        .drop_duplicates()
        .copy()
    )

    formula_buckets[
        "bucket"
    ] = [
        bucket_for_formula(
            formula,
            args.buckets,
        )
        for formula in formula_buckets[
            "molecular_formula"
        ].astype(str)
    ]

    bucket_lookup = dict(
        zip(
            formula_buckets[
                "molecular_formula"
            ].astype(str),
            formula_buckets[
                "bucket"
            ].astype(int),
            strict=True,
        )
    )

    if OUTPUT_ROOT.exists():
        shutil.rmtree(
            OUTPUT_ROOT
        )

    CANDIDATE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "Loading PubChem formula-index candidates..."
    )

    candidates = pd.read_parquet(
        SOURCE_CANDIDATES
    )

    required_candidates = {
        "molecular_formula",
        "inchikey14",
        "smiles",
    }

    missing_candidates = (
        required_candidates
        - set(candidates.columns)
    )

    if missing_candidates:
        raise ValueError(
            "PubChem candidate file is missing columns: "
            f"{sorted(missing_candidates)}"
        )

    candidates[
        "bucket"
    ] = (
        candidates[
            "molecular_formula"
        ]
        .astype(str)
        .map(bucket_lookup)
    )

    if candidates[
        "bucket"
    ].isna().any():
        raise RuntimeError(
            "Some candidate formulas are absent from the burden index"
        )

    candidates[
        "bucket"
    ] = candidates[
        "bucket"
    ].astype(int)

    print(
        "Candidate rows:",
        f"{len(candidates):,}",
    )

    for bucket in range(
        args.buckets
    ):
        part = candidates.loc[
            candidates[
                "bucket"
            ]
            == bucket
        ].drop(
            columns=[
                "bucket",
            ]
        )

        path = (
            CANDIDATE_DIR
            / f"bucket-{bucket:03d}.parquet"
        )

        part.to_parquet(
            path,
            index=False,
            compression="zstd",
        )

        if (
            bucket % 16 == 0
            or bucket
            == args.buckets - 1
        ):
            print(
                f"  wrote bucket "
                f"{bucket + 1}/{args.buckets}"
            )

    burden.to_parquet(
        BURDEN_PATH,
        index=False,
        compression="zstd",
    )

    formula_buckets.to_parquet(
        FORMULA_BUCKET_PATH,
        index=False,
        compression="zstd",
    )

    source_summary = json.loads(
        SOURCE_SUMMARY.read_text()
    )

    manifest = {
        "bucket_count":
            args.buckets,
        "formula_count":
            int(
                formula_buckets[
                    "molecular_formula"
                ].nunique()
            ),
        "candidate_rows":
            int(
                len(candidates)
            ),
        "unique_candidate_structures":
            int(
                candidates[
                    "inchikey14"
                ].nunique()
            ),
        "source_max_records":
            source_summary.get(
                "max_records"
            ),
        "source_formulas_hitting_max_records":
            source_summary.get(
                "formulas_hitting_max_records"
            ),
        "source_formula_fetch_failures":
            source_summary.get(
                "formula_fetch_failures"
            ),
        "partition_method":
            "sha256-first-8-bytes-mod-bucket-count",
    }

    MANIFEST_PATH.write_text(
        json.dumps(
            manifest,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    print()
    print(
        "Offline PubChem inference asset"
    )
    print(
        "==============================="
    )
    print(
        "Buckets:",
        args.buckets,
    )
    print(
        "Formulas:",
        f"{manifest['formula_count']:,}",
    )
    print(
        "Candidate rows:",
        f"{manifest['candidate_rows']:,}",
    )
    print(
        "Unique structures:",
        f"{manifest['unique_candidate_structures']:,}",
    )
    print(
        "Capped formulas:",
        manifest[
            "source_formulas_hitting_max_records"
        ],
    )
    print(
        "Fetch failures:",
        manifest[
            "source_formula_fetch_failures"
        ],
    )
    print()
    print(
        f"Wrote {OUTPUT_ROOT}"
    )


if __name__ == "__main__":
    main()
