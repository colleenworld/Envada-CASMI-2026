from __future__ import annotations

import argparse
import json
import os
import pickle
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(
    os.environ.get(
        "CASMI26_PROJECT_ROOT",
        Path(__file__).resolve().parent.parent,
    )
)

TEST_ROOT = PROJECT_ROOT / "data/processed/test"

SOURCE_MGF_PATH = (
    TEST_ROOT
    / "mist/test_multiformula.mgf"
)

SOURCE_LABELS_PATH = (
    TEST_ROOT
    / "mist/test_multiformula_labels.tsv"
)

CANDIDATES_PATH = (
    TEST_ROOT
    / "guard5_neighbor_top500.parquet"
)

FORM_TO_SMI_PATH = (
    TEST_ROOT
    / "guard5_full_form_to_smi.pkl"
)

WORK_DIR = (
    TEST_ROOT
    / "guard5_mist"
)

MGF_PATH = WORK_DIR / "test_positive.mgf"
LABELS_PATH = WORK_DIR / "test_positive_labels.tsv"

SANITIZED_FORM_TO_SMI_PATH = (
    WORK_DIR
    / "form_to_smi_sanitized.pkl"
)

SUBFORMS_DIR = WORK_DIR / "subforms"
HDF_DIR = WORK_DIR / "retrieval"
PREDICTION_DIR = WORK_DIR / "predictions"

DATABASE_NAME = "casmi_test_guard5_full_morgan4096"
DATASET_NAME = "casmi_test_guard5_positive"
DIST_NAME = "cosine"


def parse_args():
    parser = argparse.ArgumentParser()

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
        "--mist-python",
        type=Path,
        default=(
            Path(os.environ["MIST_PYTHON"])
            if os.environ.get("MIST_PYTHON")
            else None
        ),
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=500,
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


def resolve_mist_python(
    explicit: Path | None,
) -> Path:
    if explicit is not None:
        path = explicit.expanduser().resolve()

        if not path.exists():
            raise FileNotFoundError(
                f"MIST Python does not exist: {path}"
            )

        return path

    conda = shutil.which("conda")

    if conda:
        completed = subprocess.run(
            [conda, "env", "list", "--json"],
            check=True,
            capture_output=True,
            text=True,
        )

        payload = json.loads(
            completed.stdout
        )

        for env_path in payload.get("envs", []):
            env_path = Path(env_path)

            if env_path.name == "mist-cpu":
                python_path = env_path / "bin/python"

                if python_path.exists():
                    return python_path

    for candidate in [
        Path.home()
        / "miniconda3/envs/mist-cpu/bin/python",
        Path.home()
        / "anaconda3/envs/mist-cpu/bin/python",
    ]:
        if candidate.exists():
            return candidate

    #
    # Kaggle runs MIST in the notebook's existing Python environment.
    # The packaged MIST source is selected separately with --mist-root.
    #
    return Path(sys.executable).resolve()


def run(
    command: list[str],
    *,
    cwd: Path,
) -> None:
    print()
    print(
        "+",
        " ".join(command),
    )

    subprocess.run(
        command,
        cwd=cwd,
        check=True,
    )


def filter_mgf(
    source: Path,
    destination: Path,
    selected_specs: set[str],
) -> None:
    text = source.read_text()

    blocks = [
        block.strip()
        for block in text.split(
            "BEGIN IONS"
        )
        if block.strip()
    ]

    kept = []
    found_specs = set()

    for body in blocks:
        lines = body.splitlines()

        title = None

        for line in lines:
            if line.startswith("TITLE="):
                title = line[
                    len("TITLE="):
                ].strip()
                break

        if (
            title is not None
            and title in selected_specs
        ):
            found_specs.add(title)
            kept.append(
                "BEGIN IONS\n"
                + body
            )

    missing = selected_specs - found_specs

    if missing:
        raise ValueError(
            "Selected PubChem hypotheses are missing from the "
            "exported MGF. Examples: "
            f"{sorted(missing)[:20]}"
        )

    destination.write_text(
        "\n\n".join(kept)
        + "\n"
    )


def main() -> None:
    args = parse_args()

    mist_root = args.mist_root.resolve()
    mist_python = str(
        resolve_mist_python(
            args.mist_python
        )
    )

    assign_script = (
        mist_root
        / "src/mist/subformulae/"
        "assign_subformulae.py"
    )

    retrieval_script = (
        mist_root
        / "src/mist/retrieval_contrast.py"
    )

    checkpoint = (
        mist_root
        / "pretrained_models/"
        "mist_contrastive_canopus_pretrain.ckpt"
    )

    required = [
        SOURCE_MGF_PATH,
        SOURCE_LABELS_PATH,
        CANDIDATES_PATH,
        FORM_TO_SMI_PATH,
        assign_script,
        retrieval_script,
        checkpoint,
    ]

    missing = [
        path
        for path in required
        if not path.exists()
    ]

    if missing:
        raise FileNotFoundError(
            "Missing PubChem MIST inputs: "
            + ", ".join(
                str(path)
                for path in missing
            )
        )

    WORK_DIR.mkdir(parents=True, exist_ok=True)
    SUBFORMS_DIR.mkdir(parents=True, exist_ok=True)
    HDF_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTION_DIR.mkdir(parents=True, exist_ok=True)

    candidates = pd.read_parquet(
        CANDIDATES_PATH,
        columns=[
            "hypothesis_spec",
        ],
    )

    selected_specs = set(
        candidates[
            "hypothesis_spec"
        ].astype(str)
    )

    labels = pd.read_csv(
        SOURCE_LABELS_PATH,
        sep="\t",
    )

    labels[
        "spec"
    ] = labels[
        "spec"
    ].astype(str)

    filtered_labels = labels.loc[
        labels[
            "spec"
        ].isin(
            selected_specs
        )
    ].copy()

    #
    # MIST's ion vocabulary uses the explicit ammonia composition
    # for ammonium adducts. Keep our project-facing adduct label
    # unchanged everywhere else, but translate it in the labels
    # passed into MIST.
    #
    mist_ionization_map = {
        "[M+NH4]+": "[M+H3N+H]+",
    }

    filtered_labels[
        "ionization"
    ] = (
        filtered_labels[
            "ionization"
        ]
        .astype(str)
        .replace(
            mist_ionization_map
        )
    )

    label_specs = set(
        filtered_labels[
            "spec"
        ]
    )

    if label_specs != selected_specs:
        raise ValueError(
            "Pruned candidate specs and filtered label specs differ. "
            f"Missing labels: {len(selected_specs - label_specs)}; "
            f"unexpected labels: {len(label_specs - selected_specs)}"
        )

    filtered_labels.to_csv(
        LABELS_PATH,
        sep="\t",
        index=False,
    )

    filter_mgf(
        SOURCE_MGF_PATH,
        MGF_PATH,
        selected_specs,
    )

    print(
        "PubChem MIST hypotheses:",
        len(selected_specs),
    )
    print(
        "MIST Python:",
        mist_python,
    )
    print(
        "Top-k:",
        args.top_k,
    )

    for directory in [
        SUBFORMS_DIR,
        HDF_DIR,
        PREDICTION_DIR,
    ]:
        for path in directory.glob("*"):
            if path.is_file():
                path.unlink()

    run(
        [
            mist_python,
            str(assign_script),
            "--spec-files",
            str(MGF_PATH),
            "--labels-file",
            str(LABELS_PATH),
            "--output-dir",
            str(SUBFORMS_DIR),
            "--feature-id",
            "TITLE",
            "--num-workers",
            "0",
        ],
        cwd=mist_root,
    )

    print()
    print(
        "Sanitizing guard-5 full candidate SMILES "
        "inside the MIST environment..."
    )

    sanitizer = (
        PROJECT_ROOT
        / "scripts/sanitize_mist_candidates_resumable.py"
    )
    sanitize_cache_dir = (
        WORK_DIR
        / "sanitize_cache"
    )

    run(
        [
            mist_python,
            str(sanitizer),
            "--input",
            str(FORM_TO_SMI_PATH),
            "--output",
            str(SANITIZED_FORM_TO_SMI_PATH),
            "--cache-dir",
            str(sanitize_cache_dir),
        ],
        cwd=mist_root,
    )

    hdf_builder = (
        PROJECT_ROOT
        / "scripts/build_mist_hdf_bounded.py"
    )

    run(
        [
            mist_python,
            str(hdf_builder),
            "--mist-root",
            str(mist_root),
            "--labels-file",
            str(LABELS_PATH),
            "--form-to-smi",
            str(SANITIZED_FORM_TO_SMI_PATH),
            "--output-dir",
            str(HDF_DIR),
            "--database-name",
            DATABASE_NAME,
            "--formulas-per-batch",
            str(args.hdf_formulas_per_batch),
            "--workers",
            str(args.hdf_workers),
        ],
        cwd=mist_root,
    )

    hdf_files = sorted(
        HDF_DIR.glob(
            "*morgan4096*retrieval_db.h5"
        )
    )

    if len(hdf_files) != 1:
        raise RuntimeError(
            "Expected exactly one PubChem MIST retrieval HDF, "
            f"found {len(hdf_files)}: {hdf_files}"
        )

    hdf_path = hdf_files[0]

    run(
        [
            mist_python,
            str(retrieval_script),
            "--model-ckpt",
            str(checkpoint),
            "--dataset-name",
            DATASET_NAME,
            "--save-dir",
            str(PREDICTION_DIR),
            "--subform-folder",
            str(SUBFORMS_DIR),
            "--hdf-file",
            str(hdf_path),
            "--dist-name",
            DIST_NAME,
            "--num-workers",
            "0",
            "--labels-file",
            str(LABELS_PATH),
            "--top-k",
            str(args.top_k),
        ],
        cwd=mist_root,
    )

    predictions = sorted(
        PREDICTION_DIR.glob(
            "retrieval_contrastive_*.p"
        )
    )

    if len(predictions) != 1:
        raise RuntimeError(
            "Expected exactly one PubChem MIST prediction artifact, "
            f"found {len(predictions)}: {predictions}"
        )

    print()
    print(
        "PubChem MIST inference complete"
    )
    print(
        "==============================="
    )
    print(
        "Prediction artifact:",
        predictions[0],
    )
    print()
    print(
        "Next:"
    )
    print(
        "python scripts/rank_test_guard5_positive.py"
    )


if __name__ == "__main__":
    main()
