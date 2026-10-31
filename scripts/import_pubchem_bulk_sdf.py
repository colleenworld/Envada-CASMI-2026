from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from pathlib import Path
from urllib import request

import pandas as pd

from casmi26.pubchem.bulk_sdf import (
    extract_matching_candidates,
)
from casmi26.pubchem.formula_candidates import (
    CANDIDATE_COLUMNS,
    candidate_burden,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_FORMULA_SET = (
    PROJECT_ROOT
    / "data/processed/formula_index/"
    "training_coconut_formulas.parquet"
)

RUN_NAME = os.environ.get(
    "CASMI26_PUBCHEM_RUN_NAME",
    "",
).strip()

if RUN_NAME:
    OUTPUT_ROOT = (
        PROJECT_ROOT
        / "data/processed/pubchem_bulk_runs"
        / RUN_NAME
    )

    CHUNK_RESULT_DIR = (
        OUTPUT_ROOT
        / "chunks"
    )

    STATE_PATH = (
        OUTPUT_ROOT
        / "state.json"
    )

    FINAL_CANDIDATES_PATH = (
        OUTPUT_ROOT
        / "candidates.parquet"
    )

    FINAL_BURDEN_PATH = (
        OUTPUT_ROOT
        / "burden.parquet"
    )

    FINAL_SUMMARY_PATH = (
        OUTPUT_ROOT
        / "summary.json"
    )

else:
    OUTPUT_ROOT = (
        PROJECT_ROOT
        / "data/processed/pubchem_bulk"
    )

    CHUNK_RESULT_DIR = (
        OUTPUT_ROOT
        / "chunks"
    )

    STATE_PATH = (
        OUTPUT_ROOT
        / "state.json"
    )

    FINAL_CANDIDATES_PATH = (
        PROJECT_ROOT
        / "data/processed/diagnostics/"
        "pubchem_formula_index_candidates.parquet"
    )

    FINAL_BURDEN_PATH = (
        PROJECT_ROOT
        / "data/processed/diagnostics/"
        "pubchem_formula_index_candidates_burden.parquet"
    )

    FINAL_SUMMARY_PATH = (
        PROJECT_ROOT
        / "data/processed/diagnostics/"
        "pubchem_formula_index_candidates_summary.json"
    )

PUBCHEM_SDF_INDEX = (
    "https://ftp.ncbi.nlm.nih.gov/pubchem/"
    "Compound/CURRENT-Full/SDF/"
)

CHUNK_PATTERN = re.compile(
    r"Compound_\d{9}_\d{9}\.sdf\.gz"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Incrementally download PubChem SDF chunks, retain only "
            "compounds matching the inference formula set, and delete "
            "each source chunk after successful extraction."
        )
    )

    parser.add_argument(
        "--formula-set",
        type=Path,
        default=DEFAULT_FORMULA_SET,
    )

    parser.add_argument(
        "--download-dir",
        type=Path,
        default=(
            Path.home()
            / "pubchem-sdf"
        ),
    )

    parser.add_argument(
        "--max-chunks",
        type=int,
        default=None,
        help=(
            "Optional development limit. Omit to process all chunks."
        ),
    )

    parser.add_argument(
        "--keep-downloads",
        action="store_true",
        help=(
            "Keep downloaded .sdf.gz chunks instead of deleting them "
            "after successful extraction."
        ),
    )

    return parser.parse_args()


def load_target_formulas(
    path: Path,
) -> set[str]:
    frame = pd.read_parquet(
        path
    )

    if (
        "molecular_formula"
        not in frame.columns
    ):
        raise ValueError(
            "Formula set must contain molecular_formula"
        )

    formulas = {
        str(value).strip()
        for value in frame[
            "molecular_formula"
        ].dropna()
        if str(value).strip()
    }

    if not formulas:
        raise RuntimeError(
            "Formula set is empty"
        )

    return formulas


def discover_chunks() -> list[str]:
    with request.urlopen(
        PUBCHEM_SDF_INDEX,
        timeout=120,
    ) as response:
        html = response.read().decode(
            "utf-8",
            errors="replace",
        )

    return sorted(
        set(
            CHUNK_PATTERN.findall(
                html
            )
        )
    )


def load_state() -> dict:
    if not STATE_PATH.exists():
        return {
            "completed_chunks": {},
        }

    return json.loads(
        STATE_PATH.read_text()
    )


def save_state(
    state: dict,
) -> None:
    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = (
        STATE_PATH
        .with_suffix(
            ".json.tmp"
        )
    )

    temporary.write_text(
        json.dumps(
            state,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    temporary.replace(
        STATE_PATH
    )


def download_chunk(
    name: str,
    download_dir: Path,
) -> Path:
    download_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    path = (
        download_dir
        / name
    )

    url = (
        PUBCHEM_SDF_INDEX
        + name
    )

    print(
        "Downloading:",
        name,
    )

    subprocess.run(
        [
            "wget",
            "--continue",
            "--output-document",
            str(path),
            url,
        ],
        check=True,
    )

    return path


def chunk_output_path(
    name: str,
) -> Path:
    stem = name.removesuffix(
        ".sdf.gz"
    )

    return (
        CHUNK_RESULT_DIR
        / f"{stem}.parquet"
    )


def finalize(
    formulas: set[str],
) -> None:
    chunk_paths = sorted(
        CHUNK_RESULT_DIR.glob(
            "Compound_*.parquet"
        )
    )

    frames = [
        pd.read_parquet(
            path
        )
        for path
        in chunk_paths
    ]

    nonempty = [
        frame
        for frame in frames
        if not frame.empty
    ]

    if nonempty:
        candidates = pd.concat(
            nonempty,
            ignore_index=True,
        )

        candidates = (
            candidates
            .sort_values(
                [
                    "molecular_formula",
                    "inchikey14",
                    "pubchem_cid",
                    "full_inchikey",
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
        candidates = pd.DataFrame(
            columns=CANDIDATE_COLUMNS
        )

    FINAL_CANDIDATES_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    candidates.to_parquet(
        FINAL_CANDIDATES_PATH,
        index=False,
        compression="zstd",
    )

    burden = candidate_burden(
        candidates,
        sorted(
            formulas
        ),
    )

    burden[
        "raw_record_count"
    ] = pd.NA

    burden[
        "hit_max_records"
    ] = False

    burden[
        "fetch_failed"
    ] = False

    burden[
        "fetch_error"
    ] = pd.NA

    burden.to_parquet(
        FINAL_BURDEN_PATH,
        index=False,
        compression="zstd",
    )

    state = load_state()

    stats = list(
        state.get(
            "completed_chunks",
            {}
        ).values()
    )

    summary = {
        "source":
            "pubchem-bulk-sdf",
        "formula_count":
            len(formulas),
        "candidate_rows":
            len(candidates),
        "unique_candidate_structures":
            int(
                candidates[
                    "inchikey14"
                ].nunique()
            )
            if not candidates.empty
            else 0,
        "max_records":
            None,
        "formulas_hitting_max_records":
            0,
        "formula_fetch_failures":
            0,
        "chunks_completed":
            len(stats),
        "records_scanned":
            sum(
                int(
                    item.get(
                        "records_seen",
                        0,
                    )
                )
                for item in stats
            ),
        "invalid_records":
            sum(
                int(
                    item.get(
                        "invalid_records",
                        0,
                    )
                )
                for item in stats
            ),
        "matched_records":
            sum(
                int(
                    item.get(
                        "matched_records",
                        0,
                    )
                )
                for item in stats
            ),
    }

    FINAL_SUMMARY_PATH.write_text(
        json.dumps(
            summary,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    print()
    print(
        "Final PubChem bulk candidate catalog"
    )
    print(
        "===================================="
    )
    print(
        "Target formulas:",
        f"{len(formulas):,}",
    )
    print(
        "Candidate rows:",
        f"{len(candidates):,}",
    )
    print(
        "Unique structures:",
        f"{summary['unique_candidate_structures']:,}",
    )
    print(
        "Chunks completed:",
        f"{summary['chunks_completed']:,}",
    )
    print(
        "Records scanned:",
        f"{summary['records_scanned']:,}",
    )
    print()
    print(
        f"Wrote {FINAL_CANDIDATES_PATH}"
    )
    print(
        f"Wrote {FINAL_BURDEN_PATH}"
    )
    print(
        f"Wrote {FINAL_SUMMARY_PATH}"
    )


def main() -> None:
    args = parse_args()

    formulas = load_target_formulas(
        args.formula_set
    )

    print(
        "Target formula file:",
        args.formula_set,
    )

    print(
        "Named run:",
        RUN_NAME or "(default)",
    )

    print(
        "Run output root:",
        OUTPUT_ROOT,
    )
    print(
        "Target formulas:",
        f"{len(formulas):,}",
    )
    print(
        "Download directory:",
        args.download_dir,
    )

    CHUNK_RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    state = load_state()

    completed = state.setdefault(
        "completed_chunks",
        {},
    )

    chunks = discover_chunks()

    if args.max_chunks is not None:
        if args.max_chunks <= 0:
            raise ValueError(
                "--max-chunks must be positive"
            )

        chunks = chunks[
            :args.max_chunks
        ]

    print(
        "PubChem SDF chunks:",
        len(chunks),
    )
    print(
        "Already completed:",
        len(
            set(chunks)
            & set(completed)
        ),
    )

    for number, name in enumerate(
        chunks,
        start=1,
    ):
        if name in completed:
            print(
                f"[{number}/{len(chunks)}] "
                f"Skipping completed {name}"
            )
            continue

        local_path = (
            args.download_dir
            / name
        )

        if local_path.exists():
            print(
                f"[{number}/{len(chunks)}] "
                f"Using existing {local_path}"
            )
        else:
            print(
                f"[{number}/{len(chunks)}]"
            )

            local_path = (
                download_chunk(
                    name,
                    args.download_dir,
                )
            )

        print(
            "Extracting matches:",
            name,
        )

        frame, stats = (
            extract_matching_candidates(
                local_path,
                formulas,
            )
        )

        output_path = (
            chunk_output_path(
                name
            )
        )

        frame.to_parquet(
            output_path,
            index=False,
            compression="zstd",
        )

        completed[
            name
        ] = stats

        save_state(
            state
        )

        print(
            "  records:",
            f"{stats['records_seen']:,}",
            "| matches:",
            f"{stats['matched_records']:,}",
            "| unique retained:",
            f"{stats['unique_candidates']:,}",
        )
        print(
            "  wrote:",
            output_path,
        )

        if not args.keep_downloads:
            local_path.unlink()

            print(
                "  deleted:",
                local_path,
            )

    expected = set(
        chunks
    )

    finished = (
        expected
        <= set(
            completed
        )
    )

    if finished:
        finalize(
            formulas
        )
    else:
        print()
        print(
            "Not all selected chunks are complete; "
            "final catalog was not written."
        )


if __name__ == "__main__":
    main()
