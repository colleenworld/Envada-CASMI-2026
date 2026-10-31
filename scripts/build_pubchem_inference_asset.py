from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent

RUN_NAME = os.environ.get(
    "CASMI26_PUBCHEM_RUN_NAME",
    "",
).strip()

if RUN_NAME:
    SOURCE_ROOT = (
        PROJECT_ROOT
        / "data/processed/pubchem_bulk_runs"
        / RUN_NAME
    )

    SOURCE_CHUNK_DIR = (
        SOURCE_ROOT
        / "chunks"
    )

    SOURCE_BURDEN = (
        SOURCE_ROOT
        / "burden.parquet"
    )

    SOURCE_SUMMARY = (
        SOURCE_ROOT
        / "summary.json"
    )

    OUTPUT_ROOT = (
        PROJECT_ROOT
        / "data/processed/pubchem_formula_index_runs"
        / RUN_NAME
    )

else:
    SOURCE_CHUNK_DIR = (
        PROJECT_ROOT
        / "data/processed/pubchem_bulk/chunks"
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
        SOURCE_BURDEN,
        SOURCE_SUMMARY,
    ]

    missing = [
        path
        for path in required
        if not path.exists()
    ]

    chunk_paths = sorted(
        SOURCE_CHUNK_DIR.glob(
            "Compound_*.parquet"
        )
    )

    if not chunk_paths:
        missing.append(
            SOURCE_CHUNK_DIR
        )

    if missing:
        raise FileNotFoundError(
            "Missing PubChem bulk-import outputs. "
            "Run scripts/import_pubchem_bulk_sdf.py first. Missing: "
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

    temporary_dir = (
        OUTPUT_ROOT
        / "_bucket_fragments"
    )

    temporary_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "Partitioning PubChem bulk chunk results..."
    )
    print(
        "Named run:",
        RUN_NAME or "(default)",
    )
    print(
        "Output root:",
        OUTPUT_ROOT,
    )
    print(
        "Source chunks:",
        len(chunk_paths),
    )

    asset_columns = [
        "molecular_formula",
        "inchikey14",
        "smiles",
    ]

    required_candidates = set(
        asset_columns
    )

    fragment_counts = {
        bucket: 0
        for bucket in range(
            args.buckets
        )
    }

    for number, source_path in enumerate(
        chunk_paths,
        start=1,
    ):
        frame = pd.read_parquet(
            source_path
        )

        missing_candidates = (
            required_candidates
            - set(frame.columns)
        )

        if missing_candidates:
            raise ValueError(
                f"{source_path} is missing candidate columns: "
                f"{sorted(missing_candidates)}"
            )

        if frame.empty:
            continue

        frame = frame[
            asset_columns
        ].copy()

        frame[
            "bucket"
        ] = (
            frame[
                "molecular_formula"
            ]
            .astype(str)
            .map(bucket_lookup)
        )

        if frame[
            "bucket"
        ].isna().any():
            examples = (
                frame.loc[
                    frame[
                        "bucket"
                    ].isna(),
                    "molecular_formula",
                ]
                .astype(str)
                .drop_duplicates()
                .head(20)
                .tolist()
            )

            raise RuntimeError(
                "Some candidate formulas are absent from the burden "
                f"index. Examples: {examples}"
            )

        frame[
            "bucket"
        ] = frame[
            "bucket"
        ].astype(int)

        for bucket, part in frame.groupby(
            "bucket",
            sort=False,
        ):
            bucket = int(
                bucket
            )

            fragment_number = (
                fragment_counts[
                    bucket
                ]
            )

            fragment_counts[
                bucket
            ] += 1

            fragment_path = (
                temporary_dir
                / (
                    f"bucket-{bucket:03d}-"
                    f"{fragment_number:04d}.parquet"
                )
            )

            part.drop(
                columns=[
                    "bucket",
                ]
            ).to_parquet(
                fragment_path,
                index=False,
                compression="zstd",
            )

        if (
            number % 25 == 0
            or number == len(
                chunk_paths
            )
        ):
            print(
                f"  partitioned "
                f"{number}/{len(chunk_paths)} source chunks"
            )

    print()
    print(
        "Consolidating formula buckets..."
    )

    candidate_rows = 0

    for bucket in range(
        args.buckets
    ):
        fragments = sorted(
            temporary_dir.glob(
                f"bucket-{bucket:03d}-*.parquet"
            )
        )

        if fragments:
            parts = [
                pd.read_parquet(
                    fragment
                )
                for fragment in fragments
            ]

            part = pd.concat(
                parts,
                ignore_index=True,
            )

            part = (
                part
                .sort_values(
                    [
                        "molecular_formula",
                        "inchikey14",
                        "smiles",
                    ],
                    na_position="last",
                )
                .drop_duplicates(
                    subset=[
                        "molecular_formula",
                        "inchikey14",
                    ],
                    keep="first",
                )
                .reset_index(
                    drop=True
                )
            )

        else:
            part = pd.DataFrame(
                columns=asset_columns
            )

        candidate_rows += len(
            part
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

        for fragment in fragments:
            fragment.unlink()

        if (
            bucket % 16 == 0
            or bucket
            == args.buckets - 1
        ):
            print(
                f"  wrote bucket "
                f"{bucket + 1}/{args.buckets} "
                f"| total unique rows "
                f"{candidate_rows:,}"
            )

    shutil.rmtree(
        temporary_dir
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
                candidate_rows
            ),
        "unique_candidate_structures":
            int(
                source_summary.get(
                    "unique_candidate_structures",
                    candidate_rows,
                )
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
        "candidate_columns":
            asset_columns,
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
