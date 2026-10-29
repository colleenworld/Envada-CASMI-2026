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

BLIND_ROOT = (
    PROJECT_ROOT
    / "data/processed/blind_pseudotest"
)

EVAL_ROOT = (
    BLIND_ROOT
    / "fused_formula_e2e"
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

    pseudo_test = (
        BLIND_ROOT
        / "test.parquet"
    )

    pseudo_sample = (
        BLIND_ROOT
        / "sample_submission.csv"
    )

    fused_formulas = (
        BLIND_ROOT
        / "production_msbuddy_fused_formulas.parquet"
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
        pseudo_test,
        pseudo_sample,
        fused_formulas,
        strict_retrieval
        / "metadata.parquet",
        strict_retrieval
        / "vectors.npz",
        training_structures,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"Missing required artifact: {path}"
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
        EVAL_ROOT
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

    EVAL_ROOT.mkdir(
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
        "CASMI26_RETRIEVAL_INDEX_DIR"
    ] = str(
        strict_retrieval
    )
    env[
        "CASMI26_STRUCTURES_PATH"
    ] = str(
        training_structures
    )

    # Keep the benchmark honest: use the same offline PubChem asset as
    # production and allow MSBuddy-only formulas that are not covered by
    # that asset to fall through instead of querying/building a
    # blind-specific expansion.
    env[
        "CASMI26_ALLOW_PARTIAL_PUBCHEM"
    ] = "1"

    test_output = (
        PROJECT_ROOT
        / "data/processed/test"
    )

    if test_output.exists():
        shutil.rmtree(
            test_output
        )

    if submission.exists():
        submission.unlink()

    try:
        shutil.copy2(
            pseudo_test,
            raw_test,
        )

        shutil.copy2(
            pseudo_sample,
            raw_sample,
        )

        stages = [
            "scripts/build_test_multispectrum_formula_hypotheses.py",
            "scripts/apply_blind_fused_formula_hypotheses.py",
            "scripts/prepare_test_guard5_candidates.py",
            "scripts/score_test_guard5_spectrum_fingerprint.py",
            "scripts/build_test_learned_ranker_features.py",
            "scripts/rank_test_learned_ranker.py",
            "scripts/rank_test_negative_retrieval.py",
            "scripts/build_submission.py",
        ]

        for stage in stages:
            run(
                [
                    python,
                    stage,
                ],
                env=env,
            )

        if not submission.exists():
            raise RuntimeError(
                "Fused formula inference completed without submission.csv"
            )

        fused_submission = (
            EVAL_ROOT
            / "submission.csv"
        )

        shutil.copy2(
            submission,
            fused_submission,
        )

        run(
            [
                python,
                "scripts/evaluate_blind_pseudotest.py",
                "--root",
                str(
                    BLIND_ROOT
                ),
                "--submission",
                str(
                    fused_submission
                ),
            ],
            env=env,
        )

        for name in [
            "evaluation.parquet",
            "failure_summary.csv",
        ]:
            source = (
                BLIND_ROOT
                / name
            )

            if source.exists():
                shutil.copy2(
                    source,
                    EVAL_ROOT
                    / name,
                )

        print()
        print(
            "Fused formula blind end-to-end evaluation complete"
        )
        print(
            "=" * 76
        )
        print(
            "Evaluation root:",
            EVAL_ROOT,
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


if __name__ == "__main__":
    main()
