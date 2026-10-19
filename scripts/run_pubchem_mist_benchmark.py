from __future__ import annotations

import argparse
import os
import pickle
import subprocess
import sys
import time
from pathlib import Path


PROJECT_ROOT = (
    Path(__file__).resolve().parent.parent
)

BENCHMARK_DIR = (
    PROJECT_ROOT
    / "data/processed/diagnostics/"
    "pubchem_mist_benchmark"
)

MGF_PATH = (
    BENCHMARK_DIR
    / "benchmark.mgf"
)

LABELS_PATH = (
    BENCHMARK_DIR
    / "benchmark_labels.tsv"
)

FORM_TO_SMI_PATH = (
    BENCHMARK_DIR
    / "form_to_smi.pkl"
)

SUBFORMS_DIR = (
    BENCHMARK_DIR
    / "subforms"
)

HDF_DIR = (
    BENCHMARK_DIR
    / "retrieval"
)

PREDICTION_DIR = (
    BENCHMARK_DIR
    / "predictions"
)

DATABASE_NAME = (
    "pubchem_benchmark_morgan4096"
)

DATASET_NAME = (
    "pubchem_benchmark_positive"
)

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
        default=Path(
            os.environ.get(
                "MIST_PYTHON",
                sys.executable,
            )
        ),
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=500,
    )

    return parser.parse_args()


def run(
    command: list[str],
    *,
    cwd: Path,
) -> None:
    print()
    print(
        "+",
        " ".join(
            command
        ),
    )

    subprocess.run(
        command,
        cwd=cwd,
        check=True,
    )


def main() -> None:
    args = parse_args()

    mist_root = (
        args.mist_root.resolve()
    )

    mist_python = str(
        args.mist_python
    )

    assign_script = (
        mist_root
        / "src/mist/subformulae/"
        "assign_subformulae.py"
    )

    retrieval_script = (
        mist_root
        / "src/mist/"
        "retrieval_contrast.py"
    )

    checkpoint = (
        mist_root
        / "pretrained_models/"
        "mist_contrastive_canopus_pretrain.ckpt"
    )

    required = [
        MGF_PATH,
        LABELS_PATH,
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
            "Missing benchmark/MIST inputs: "
            + ", ".join(
                str(path)
                for path in missing
            )
        )

    SUBFORMS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    HDF_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    PREDICTION_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    started = time.perf_counter()

    print(
        "MIST root:",
        mist_root,
    )
    print(
        "MIST Python:",
        mist_python,
    )
    print(
        "Top-k:",
        args.top_k,
    )

    #
    # 1. Assign subformulae for the sampled validation spectra.
    #
    run(
        [
            mist_python,
            str(
                assign_script
            ),
            "--spec-files",
            str(
                MGF_PATH
            ),
            "--labels-file",
            str(
                LABELS_PATH
            ),
            "--output-dir",
            str(
                SUBFORMS_DIR
            ),
            "--feature-id",
            "TITLE",
            "--num-workers",
            "0",
        ],
        cwd=mist_root,
    )

    #
    # 2. Build the Morgan-4096 retrieval database from the selected
    #    PubChem formula candidates.
    #
    sys.path.insert(
        0,
        str(
            mist_root
            / "src"
        ),
    )

    from mist.retrieval_lib.make_hdf5 import (
        make_retrieval_hdf5,
    )

    with FORM_TO_SMI_PATH.open(
        "rb"
    ) as handle:
        form_to_smi = pickle.load(
            handle
        )

    print()
    print(
        "Building MIST Morgan-4096 retrieval HDF..."
    )

    make_retrieval_hdf5(
        labels_file=str(
            LABELS_PATH
        ),
        form_to_smi=form_to_smi,
        output_dir=str(
            HDF_DIR
        ),
        database_name=DATABASE_NAME,
        fp_names=[
            "morgan4096",
        ],
    )

    hdf_files = sorted(
        HDF_DIR.glob(
            "*morgan4096*retrieval_db.h5"
        )
    )

    if len(hdf_files) != 1:
        raise RuntimeError(
            "Expected exactly one benchmark retrieval HDF, "
            f"found {len(hdf_files)}: {hdf_files}"
        )

    hdf_path = hdf_files[0]

    print(
        "HDF:",
        hdf_path,
    )

    #
    # 3. Run the pretrained positive-ion contrastive checkpoint.
    #
    run(
        [
            mist_python,
            str(
                retrieval_script
            ),
            "--model-ckpt",
            str(
                checkpoint
            ),
            "--dataset-name",
            DATASET_NAME,
            "--save-dir",
            str(
                PREDICTION_DIR
            ),
            "--subform-folder",
            str(
                SUBFORMS_DIR
            ),
            "--hdf-file",
            str(
                hdf_path
            ),
            "--dist-name",
            DIST_NAME,
            "--num-workers",
            "0",
            "--labels-file",
            str(
                LABELS_PATH
            ),
            "--top-k",
            str(
                args.top_k
            ),
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
            "Expected exactly one MIST prediction artifact, "
            f"found {len(predictions)}: {predictions}"
        )

    prediction_path = (
        predictions[0]
    )

    elapsed = (
        time.perf_counter()
        - started
    )

    print()
    print(
        "MIST benchmark inference complete"
    )
    print(
        "================================="
    )
    print(
        "Elapsed seconds:",
        f"{elapsed:.1f}",
    )
    print(
        "Prediction artifact:",
        prediction_path,
    )
    print()
    print(
        "Evaluate with:"
    )
    print(
        f"{sys.executable} "
        "scripts/evaluate_pubchem_mist_benchmark.py "
        f"{prediction_path}"
    )


if __name__ == "__main__":
    main()
