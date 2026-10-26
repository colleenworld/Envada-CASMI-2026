from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys


PROJECT_ROOT = Path(
    os.environ.get(
        "CASMI26_PROJECT_ROOT",
        Path(__file__).resolve().parent.parent,
    )
)

DEFAULT_ROOT = (
    PROJECT_ROOT
    / "data/processed/blind_pseudotest"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create and run a leakage-safe blind pseudo-test using "
            "the exact learned-ranker Kaggle inference path."
        )
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=1000,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20261026,
    )
    parser.add_argument(
        "--mist-root",
        type=Path,
        default=Path(
            os.environ.get(
                "MIST_ROOT",
                "/home/colleen/PycharmProjects/MIST",
            )
        ),
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
    )
    return parser.parse_args()


def run(
    command: list[str],
    *,
    env: dict[str, str],
) -> None:
    print()
    print(
        "+",
        " ".join(command),
        flush=True,
    )

    subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        env=env,
        check=True,
    )


def backup_file(
    source: Path,
    backup_dir: Path,
) -> Path | None:
    if not source.exists():
        return None

    destination = (
        backup_dir
        / source.name
    )

    shutil.copy2(
        source,
        destination,
    )

    return destination


def restore_file(
    backup: Path | None,
    destination: Path,
) -> None:
    if backup is None:
        if destination.exists():
            destination.unlink()
        return

    shutil.copy2(
        backup,
        destination,
    )


def main() -> None:
    args = parse_args()

    # Preserve the virtual-environment interpreter path.
    # Resolving it follows the venv symlink back to /usr/bin/python
    # and loses the environment's installed packages.
    python = sys.executable

    args.root.mkdir(
        parents=True,
        exist_ok=True,
    )

    env = os.environ.copy()

    env[
        "CASMI26_PROJECT_ROOT"
    ] = str(
        PROJECT_ROOT
    )

    env[
        "MIST_ROOT"
    ] = str(
        args.mist_root
        .expanduser()
        .resolve()
    )

    strict_retrieval = (
        PROJECT_ROOT
        / "data/processed/retrieval_index_strict_validation"
    )

    training_structures = (
        PROJECT_ROOT
        / "data/processed/structure_index/training_structures.parquet"
    )

    for path in [
        strict_retrieval
        / "metadata.parquet",
        strict_retrieval
        / "vectors.npz",
        training_structures,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                "Blind validation requires leakage-safe asset: "
                f"{path}"
            )

    env[
        "CASMI26_RETRIEVAL_INDEX_DIR"
    ] = str(
        strict_retrieval
    )

    env[
        "CASMI26_STRUCTURES_PATH"
    ] = str(
        training_structures
    )

    run(
        [
            python,
            "scripts/create_blind_pseudotest.py",
            "--sample-size",
            str(
                args.sample_size
            ),
            "--seed",
            str(
                args.seed
            ),
            "--output-root",
            str(
                args.root
            ),
        ],
        env=env,
    )

    pseudo_test = (
        args.root
        / "test.parquet"
    )

    pseudo_sample = (
        args.root
        / "sample_submission.csv"
    )

    raw_test = (
        PROJECT_ROOT
        / "data/raw/test.parquet"
    )

    raw_sample = (
        PROJECT_ROOT
        / "data/raw/sample_submission.csv"
    )

    submission = (
        PROJECT_ROOT
        / "submission.csv"
    )

    backup_dir = (
        args.root
        / "_backup"
    )

    if backup_dir.exists():
        shutil.rmtree(
            backup_dir
        )

    backup_dir.mkdir(
        parents=True
    )

    test_backup = backup_file(
        raw_test,
        backup_dir,
    )

    sample_backup = backup_file(
        raw_sample,
        backup_dir,
    )

    submission_backup = backup_file(
        submission,
        backup_dir,
    )

    pseudo_submission = (
        args.root
        / "submission.csv"
    )

    try:
        run(
            [
                python,
                "scripts/run_kaggle_guard5_inference.py",
                "--project-root",
                str(
                    PROJECT_ROOT
                ),
                "--mist-root",
                str(
                    args.mist_root
                    .expanduser()
                    .resolve()
                ),
                "--test-path",
                str(
                    pseudo_test
                ),
                "--sample-submission-path",
                str(
                    pseudo_sample
                ),
            ],
            env=env,
        )

        if not submission.exists():
            raise RuntimeError(
                "Inference completed without submission.csv"
            )

        shutil.copy2(
            submission,
            pseudo_submission,
        )

        run(
            [
                python,
                "scripts/evaluate_blind_pseudotest.py",
                "--root",
                str(
                    args.root
                ),
                "--submission",
                str(
                    pseudo_submission
                ),
            ],
            env=env,
        )

    finally:
        restore_file(
            test_backup,
            raw_test,
        )

        restore_file(
            sample_backup,
            raw_sample,
        )

        restore_file(
            submission_backup,
            submission,
        )

        shutil.rmtree(
            backup_dir,
            ignore_errors=True,
        )

    print()
    print("Blind pseudo-test complete")
    print("=" * 76)
    print(
        "Pseudo-test artifacts:",
        args.root,
    )
    print(
        "Original Kaggle raw test/sample files restored."
    )
    print(
        "Inference artifacts remain in data/processed/test for diagnosis."
    )


if __name__ == "__main__":
    main()
