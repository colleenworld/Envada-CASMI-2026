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
            "Run the top-3 formula + full PubChem + guard-5 "
            "CASMI26 inference pipeline on Kaggle."
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
    parser.add_argument(
        "--hdf-formulas-per-batch",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--hdf-workers",
        type=int,
        default=2,
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
        project_root
        / "data/processed/spectrum_fingerprint/prototype/"
        "spectrum_to_morgan2048.pt",
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
            "Missing guard-5 Kaggle inference inputs: "
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

    python = str(
        Path(
            sys.executable
        ).resolve()
    )

    env[
        "MIST_PYTHON"
    ] = python

    pythonpath = str(
        project_root
        / "src"
    )

    if env.get(
        "PYTHONPATH"
    ):
        pythonpath += (
            os.pathsep
            + env[
                "PYTHONPATH"
            ]
        )

    env[
        "PYTHONPATH"
    ] = pythonpath

    #
    # Kaggle inference assets include an offline wheelhouse. The guard-5
    # candidate stage requires RDKit, which is not guaranteed to be in
    # the base runtime. Install packaged wheels before running stages.
    #
    wheelhouse = (
        mist_root.parent
        / "wheelhouse"
    )

    if wheelhouse.exists():
        wheels = sorted(
            wheelhouse.glob(
                "*.whl"
            )
        )

        if wheels:
            run(
                [
                    python,
                    "-m",
                    "pip",
                    "install",
                    "--no-index",
                    "--find-links",
                    str(
                        wheelhouse
                    ),
                    *[
                        str(
                            wheel
                        )
                        for wheel in wheels
                    ],
                ],
                cwd=project_root,
                env=env,
            )

    try:
        subprocess.run(
            [
                python,
                "-c",
                "import rdkit; print('RDKit:', rdkit.__version__)",
            ],
            cwd=project_root,
            env=env,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            "RDKit is required for guard-5 inference and was not "
            "available after offline wheelhouse setup."
        ) from exc

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
            "scripts/prepare_test_guard5_candidates.py",
        ],
        [
            python,
            "scripts/score_test_guard5_spectrum_fingerprint.py",
        ],
        [
            python,
            "scripts/run_test_guard5_mist.py",
            "--mist-root",
            str(
                mist_root
            ),
            "--mist-python",
            python,
            "--hdf-formulas-per-batch",
            str(
                args.hdf_formulas_per_batch
            ),
            "--hdf-workers",
            str(
                args.hdf_workers
            ),
        ],
        [
            python,
            "scripts/rank_test_guard5_positive.py",
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
        "Guard-5 Kaggle inference"
    )
    print(
        "========================"
    )
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

    for command in stages:
        run(
            command,
            cwd=project_root,
            env=env,
        )

    if not submission.exists():
        raise RuntimeError(
            "Guard-5 inference completed without submission.csv"
        )

    print()
    print(
        "Guard-5 Kaggle inference complete"
    )
    print(
        "================================"
    )
    print(
        "Submission:",
        submission,
    )


if __name__ == "__main__":
    main()
