from __future__ import annotations

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

ROOT = (
    PROJECT_ROOT
    / "data/processed/domain_pseudotest"
)


def run(
    command: list[str],
    *,
    env: dict[str, str],
) -> None:
    print()
    print(
        "+",
        " ".join(
            command
        ),
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
    python = sys.executable

    env = os.environ.copy()

    env[
        "CASMI26_PROJECT_ROOT"
    ] = str(
        PROJECT_ROOT
    )

    run(
        [
            python,
            "scripts/create_domain_pseudotest.py",
            "--output-root",
            str(
                ROOT
            ),
        ],
        env=env,
    )

    exclusion_path = (
        ROOT
        / "excluded_structure_keys.parquet"
    )

    env[
        "CASMI26_EXCLUDED_STRUCTURE_KEYS_PATH"
    ] = str(
        exclusion_path
    )

    pseudo_test = (
        ROOT
        / "test.parquet"
    )

    pseudo_sample = (
        ROOT
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
        ROOT
        / "_backup"
    )

    if backup_dir.exists():
        shutil.rmtree(
            backup_dir
        )

    backup_dir.mkdir(
        parents=True,
        exist_ok=True,
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
        ROOT
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
                    ROOT
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
    print(
        "Domain pseudo-test complete"
    )
    print(
        "=" * 76
    )
    print(
        "Artifacts:",
        ROOT,
    )
    print(
        "Original Kaggle raw test/sample files restored."
    )


if __name__ == "__main__":
    main()
