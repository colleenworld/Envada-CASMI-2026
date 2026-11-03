from __future__ import annotations

import argparse
import importlib.metadata
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Phase 2 on the original CASMI26 MIST/Morgan-4096 "
            "submission pipeline, adding frozen MSBuddy formula fusion "
            "and PubChem mass-tier candidates."
        )
    )
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--mist-root", type=Path, required=True)
    parser.add_argument("--test-path", type=Path, required=True)
    parser.add_argument("--sample-submission-path", type=Path, required=True)
    parser.add_argument("--pubchem-mass-tier-root", type=Path, required=True)
    parser.add_argument("--wheelhouse", type=Path)
    parser.add_argument("--msbuddy-wheel", type=Path)
    return parser.parse_args()


def run(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
) -> None:
    print()
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, env=env, check=True)


def installed_distribution_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def ensure_rdkit(python: str, wheelhouse: Path | None, cwd: Path) -> None:
    check = subprocess.run(
        [python, "-c", "import rdkit; print(rdkit.__version__)"],
        cwd=cwd,
        text=True,
        capture_output=True,
    )
    if check.returncode == 0:
        print("RDKit:", check.stdout.strip())
        return

    if wheelhouse is None or not wheelhouse.is_dir():
        raise RuntimeError(
            "RDKit is not installed and no offline wheelhouse was supplied."
        )

    run(
        [
            python,
            "-m",
            "pip",
            "install",
            "--no-index",
            "--find-links",
            str(wheelhouse),
            "rdkit",
        ],
        cwd=cwd,
        env=os.environ.copy(),
    )


def ensure_msbuddy_import(python: str, cwd: Path) -> None:
    import_check = subprocess.run(
        [
            python,
            "-c",
            "from msbuddy import Msbuddy, MsbuddyConfig; "
            "from msbuddy.base import MetaFeature, Spectrum",
        ],
        cwd=cwd,
        text=True,
        capture_output=True,
    )
    if import_check.returncode != 0:
        raise RuntimeError(
            "MSBuddy 0.3.16 is installed but its runtime imports failed. "
            "Check that all MSBuddy dependencies are available in the offline "
            "wheelhouse.\n"
            + import_check.stderr
        )


def ensure_msbuddy(
    python: str,
    wheelhouse: Path | None,
    explicit_wheel: Path | None,
    cwd: Path,
) -> None:
    actual = installed_distribution_version("msbuddy")
    if actual is not None:
        if actual != "0.3.16":
            raise RuntimeError(
                f"Phase 2 requires msbuddy==0.3.16; found {actual}."
            )
        print("MSBuddy:", actual)
        ensure_msbuddy_import(python, cwd)
        return

    wheel = explicit_wheel
    if wheel is None and wheelhouse is not None and wheelhouse.is_dir():
        matches = sorted(wheelhouse.glob("msbuddy-0.3.16*.whl"))
        if matches:
            wheel = matches[0]

    if wheel is None or not wheel.is_file():
        raise FileNotFoundError(
            "Phase 2 requires msbuddy==0.3.16, but it is neither installed "
            "nor present as a wheel. Add msbuddy-0.3.16*.whl to the offline "
            "wheelhouse at the root of the casmi26-inference-assets dataset."
        )

    command = [
        python,
        "-m",
        "pip",
        "install",
        "--no-index",
        "--find-links",
        str(wheel.parent),
    ]
    if wheelhouse is not None and wheelhouse.is_dir():
        command.extend(["--find-links", str(wheelhouse)])
    command.append(str(wheel))

    run(command, cwd=cwd, env=os.environ.copy())

    actual = installed_distribution_version("msbuddy")
    if actual != "0.3.16":
        raise RuntimeError(
            f"MSBuddy wheel install did not produce version 0.3.16; found {actual!r}."
        )
    print("MSBuddy:", actual)
    ensure_msbuddy_import(python, cwd)


def main() -> None:
    args = parse_args()

    project_root = args.project_root.expanduser().resolve()
    mist_root = args.mist_root.expanduser().resolve()
    test_path = args.test_path.expanduser().resolve()
    sample_path = args.sample_submission_path.expanduser().resolve()
    mass_tier_root = args.pubchem_mass_tier_root.expanduser().resolve()
    wheelhouse = (
        args.wheelhouse.expanduser().resolve()
        if args.wheelhouse is not None
        else mist_root.parent / "wheelhouse"
    )
    msbuddy_wheel = (
        args.msbuddy_wheel.expanduser().resolve()
        if args.msbuddy_wheel is not None
        else None
    )

    test_root = project_root / "data/processed/test"
    raw_root = project_root / "data/raw"
    fused_hypotheses = (
        test_root / "multispectrum_formula_hypotheses_fused.parquet"
    )
    positive_labels_path = (
        test_root / "mist/test_multiformula_labels_mist_supported.tsv"
    )
    subformula_root = test_root / "mist/subforms_positive"
    predictions_root = test_root / "contrastive_predictions_positive"
    checkpoint = (
        mist_root
        / "pretrained_models/mist_contrastive_canopus_pretrain.ckpt"
    )

    required = [
        project_root / "scripts/build_test_multispectrum_formula_hypotheses.py",
        project_root / "scripts/build_test_msbuddy_fused_formula_hypotheses.py",
        project_root / "scripts/export_test_mist_inputs.py",
        project_root / "scripts/filter_test_mist_supported.py",
        project_root / "scripts/build_test_contrastive_candidates.py",
        project_root / "scripts/rank_test_positive_contrastive.py",
        project_root / "scripts/rank_test_negative_retrieval.py",
        project_root / "scripts/build_submission.py",
        project_root / "src/casmi26/chemistry/inference.py",
        project_root / "src/casmi26/pubchem/mass_tier.py",
        project_root / "src/casmi26/pubchem/mass_tier_candidates.py",
        project_root / "data/processed/formula_index/training_coconut_formulas.parquet",
        project_root / "data/processed/structure_index/structures.parquet",
        project_root / "data/processed/retrieval_index/metadata.parquet",
        project_root / "data/processed/retrieval_index/vectors.npz",
        project_root / "data/external/coconut/coconut_approved.parquet",
        mist_root / "src/mist/subformulae/assign_subformulae.py",
        mist_root / "src/mist/retrieval_contrast.py",
        checkpoint,
        test_path,
        sample_path,
        mass_tier_root / "pubchem_tier__pc_mass.npy",
        mass_tier_root / "pubchem_tier__pc_off.npy",
        mass_tier_root / "pubchem_tier__pc_smiles.npy",
    ]
    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Phase 2 preflight found missing inputs:\n  "
            + "\n  ".join(str(path) for path in missing)
        )

    # Validate dependencies before clearing any previous test artifacts.
    ensure_rdkit(sys.executable, wheelhouse, project_root)
    ensure_msbuddy(sys.executable, wheelhouse, msbuddy_wheel, project_root)

    raw_root.mkdir(parents=True, exist_ok=True)
    shutil.copy2(test_path, raw_root / "test.parquet")
    shutil.copy2(sample_path, raw_root / "sample_submission.csv")

    if test_root.exists():
        shutil.rmtree(test_root)
    test_root.mkdir(parents=True, exist_ok=True)

    submission = project_root / "submission.csv"
    if submission.exists():
        submission.unlink()

    env = os.environ.copy()
    env["CASMI26_PROJECT_ROOT"] = str(project_root)
    env["CASMI26_FORMULA_HYPOTHESES_PATH"] = str(fused_hypotheses)
    env["CASMI26_USE_PUBCHEM_MASS_TIER"] = "1"
    env["CASMI26_PUBCHEM_MASS_TIER_ROOT"] = str(mass_tier_root)
    env["MIST_ROOT"] = str(mist_root)
    env["MIST_PYTHON"] = sys.executable

    pythonpath = [
        str(project_root / "src"),
        str(mist_root / "src"),
        env.get("PYTHONPATH", ""),
    ]
    env["PYTHONPATH"] = os.pathsep.join(value for value in pythonpath if value)

    python = sys.executable

    print("CASMI26 Phase 2 — original MIST/Morgan-4096 pipeline")
    print("=" * 66)
    print("Project root:", project_root)
    print("MIST root:", mist_root)
    print("PubChem mass-tier root:", mass_tier_root)
    print("Test rows:", len(pd.read_parquet(test_path)))
    print("MSBuddy: 0.3.16")
    print("Ranking: original MIST Morgan-4096; weights unchanged")

    run(
        [python, "scripts/build_test_multispectrum_formula_hypotheses.py"],
        cwd=project_root,
        env=env,
    )
    run(
        [python, "scripts/build_test_msbuddy_fused_formula_hypotheses.py"],
        cwd=project_root,
        env=env,
    )
    run(
        [python, "scripts/export_test_mist_inputs.py"],
        cwd=project_root,
        env=env,
    )
    run(
        [python, "scripts/filter_test_mist_supported.py"],
        cwd=project_root,
        env=env,
    )
    run(
        [python, "scripts/build_test_contrastive_candidates.py"],
        cwd=project_root,
        env=env,
    )

    positive_labels = pd.read_csv(positive_labels_path, sep="\t")
    if not positive_labels.empty:
        if subformula_root.exists():
            shutil.rmtree(subformula_root)
        run(
            [
                python,
                str(mist_root / "src/mist/subformulae/assign_subformulae.py"),
                "--spec-files",
                str(test_root / "mist/test_multiformula.mgf"),
                "--labels-file",
                str(positive_labels_path),
                "--output-dir",
                str(subformula_root),
                "--feature-id",
                "FEATURE_ID",
                "--num-workers",
                "0",
            ],
            cwd=mist_root,
            env=env,
        )
        json_count = len(list(subformula_root.glob("*.json")))
        if json_count != len(positive_labels):
            raise RuntimeError(
                "MIST subformula output count mismatch: "
                f"{json_count} JSON files for {len(positive_labels)} labels."
            )

        hdf_files = sorted(
            (test_root / "contrastive_retrieval").glob(
                "*morgan4096*retrieval_db.h5"
            )
        )
        if len(hdf_files) != 1:
            raise RuntimeError(
                "Expected exactly one Morgan-4096 retrieval HDF; found: "
                + ", ".join(str(path) for path in hdf_files)
            )

        if predictions_root.exists():
            shutil.rmtree(predictions_root)
        run(
            [
                python,
                str(mist_root / "src/mist/retrieval_contrast.py"),
                "--model-ckpt",
                str(checkpoint),
                "--dataset-name",
                "casmi_test_positive_contrastive",
                "--labels-file",
                str(positive_labels_path),
                "--subform-folder",
                str(subformula_root),
                "--hdf-file",
                str(hdf_files[0]),
                "--save-dir",
                str(predictions_root),
                "--dist-name",
                "cosine",
                "--num-workers",
                "0",
                "--top-k",
                "500",
            ],
            cwd=mist_root,
            env=env,
        )

    run(
        [python, "scripts/rank_test_positive_contrastive.py"],
        cwd=project_root,
        env=env,
    )
    run(
        [python, "scripts/rank_test_negative_retrieval.py"],
        cwd=project_root,
        env=env,
    )
    run(
        [python, "scripts/build_submission.py"],
        cwd=project_root,
        env=env,
    )

    if not submission.is_file():
        raise RuntimeError(
            "Phase 2 pipeline finished without creating submission.csv."
        )

    print()
    print("Phase 2 complete:", submission)
    print("Submission rows:", len(pd.read_csv(submission)))


if __name__ == "__main__":
    main()
