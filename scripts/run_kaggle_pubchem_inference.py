from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run the complete CASMI26 PubChem inference pipeline "
            "inside a writable Kaggle working copy."
        )
    )

    parser.add_argument(
        "--project-root",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--mist-root",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--test-path",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--sample-submission-path",
        type=Path,
        required=True,
    )

    return parser.parse_args()


def run(
    command: list[str],
    *,
    cwd: Path,
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
        cwd=cwd,
        env=env,
        check=True,
    )


def main() -> None:
    args = parse_args()

    project_root = (
        args.project_root
        .expanduser()
        .resolve()
    )

    mist_root = (
        args.mist_root
        .expanduser()
        .resolve()
    )

    test_path = (
        args.test_path
        .expanduser()
        .resolve()
    )

    sample_path = (
        args.sample_submission_path
        .expanduser()
        .resolve()
    )

    required = [
        project_root / "scripts",
        project_root / "src",
        project_root
        / "data/processed/formula_index",
        project_root
        / "data/processed/structure_index",
        project_root
        / "data/processed/retrieval_index",
        project_root
        / "data/processed/pubchem_formula_index",
        project_root
        / "data/external/coconut",
        mist_root / "src",
        mist_root
        / "pretrained_models/"
        "mist_contrastive_canopus_pretrain.ckpt",
        test_path,
        sample_path,
    ]

    missing = [
        path
        for path in required
        if not path.exists()
    ]

    if missing:
        raise FileNotFoundError(
            "Missing Kaggle inference inputs: "
            + ", ".join(
                str(path)
                for path in missing
            )
        )

    raw_dir = (
        project_root
        / "data/raw"
    )

    raw_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    shutil.copy2(
        test_path,
        raw_dir / "test.parquet",
    )

    shutil.copy2(
        sample_path,
        raw_dir / "sample_submission.csv",
    )

    test_output = (
        project_root
        / "data/processed/test"
    )

    if test_output.exists():
        shutil.rmtree(
            test_output
        )

    submission = (
        project_root
        / "submission.csv"
    )

    if submission.exists():
        submission.unlink()

    env = os.environ.copy()

    env[
        "CASMI26_PROJECT_ROOT"
    ] = str(
        project_root
    )

    env[
        "MIST_ROOT"
    ] = str(
        mist_root
    )

    env[
        "MIST_PYTHON"
    ] = str(
        Path(
            sys.executable
        ).resolve()
    )

    pythonpath = str(
        project_root
        / "src"
    )

    existing_pythonpath = env.get(
        "PYTHONPATH"
    )

    if existing_pythonpath:
        pythonpath = (
            pythonpath
            + os.pathsep
            + existing_pythonpath
        )

    env[
        "PYTHONPATH"
    ] = pythonpath

    python = str(
        Path(
            sys.executable
        ).resolve()
    )

    stages = [
        [
            python,
            "scripts/build_test_multispectrum_formula_hypotheses.py",
        ],
        [
            python,
            "scripts/export_test_mist_inputs.py",
        ],
        [
            python,
            "scripts/prepare_test_pubchem_candidates.py",
        ],
        [
            python,
            "scripts/run_test_pubchem_mist.py",
            "--mist-root",
            str(
                mist_root
            ),
            "--mist-python",
            python,
        ],
        [
            python,
            "scripts/rank_test_pubchem_positive.py",
        ],
        [
            python,
            "scripts/rank_test_negative_retrieval.py",
        ],
        [
            python,
            "scripts/build_submission.py",
        ],
    ]

    print(
        "Project root:",
        project_root,
    )

    print(
        "MIST root:",
        mist_root,
    )

    print(
        "Python:",
        python,
    )

    print(
        "Test:",
        test_path,
    )

    print(
        "Sample submission:",
        sample_path,
    )

    for command in stages:
        run(
            command,
            cwd=project_root,
            env=env,
        )

    if not submission.exists():
        raise RuntimeError(
            "Inference completed without creating submission.csv"
        )

    print()
    print(
        "Kaggle inference complete"
    )
    print(
        "========================="
    )
    print(
        "Submission:",
        submission,
    )


if __name__ == "__main__":
    main()
