from __future__ import annotations

import argparse
import os
import json
import shutil
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

SANITIZED_FORM_TO_SMI_PATH = (
    BENCHMARK_DIR
    / "form_to_smi_sanitized.pkl"
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
        default=(
            Path(
                os.environ[
                    "MIST_PYTHON"
                ]
            )
            if os.environ.get(
                "MIST_PYTHON"
            )
            else None
        ),
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=500,
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

    conda = shutil.which(
        "conda"
    )

    if conda:
        completed = subprocess.run(
            [
                conda,
                "env",
                "list",
                "--json",
            ],
            check=True,
            capture_output=True,
            text=True,
        )

        payload = json.loads(
            completed.stdout
        )

        for env_path in payload.get(
            "envs",
            [],
        ):
            env_path = Path(
                env_path
            )

            if env_path.name == "mist-cpu":
                python_path = (
                    env_path
                    / "bin/python"
                )

                if python_path.exists():
                    return python_path

    candidates = [
        Path.home()
        / "miniconda3/envs/mist-cpu/bin/python",
        Path.home()
        / "anaconda3/envs/mist-cpu/bin/python",
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "Could not locate the mist-cpu Python interpreter. "
        "Pass --mist-python /path/to/mist-cpu/bin/python "
        "or set MIST_PYTHON."
    )


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

    mist_python_path = (
        resolve_mist_python(
            args.mist_python
        )
    )

    mist_python = str(
        mist_python_path
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
    # 2. Sanitize PubChem candidate SMILES with the exact RDKit
    #    version used by the MIST environment. PubChem occasionally
    #    returns structures that this older RDKit cannot parse; MIST's
    #    Morgan featurizer otherwise crashes on those entries.
    #
    print()
    print(
        "Sanitizing candidate SMILES in MIST environment..."
    )

    sanitize_code = (
        "import pickle; "
        "from rdkit import Chem, RDLogger; "
        "RDLogger.DisableLog('rdApp.*'); "
        f"src={str(FORM_TO_SMI_PATH)!r}; "
        f"dst={str(SANITIZED_FORM_TO_SMI_PATH)!r}; "
        "data=pickle.load(open(src,'rb')); "
        "out={}; total=0; kept=0; removed=0; "
        "\nfor formula, entries in data.items():"
        "\n    clean=[]"
        "\n    seen=set()"
        "\n    for entry in entries:"
        "\n        total += 1"
        "\n        smi=str(entry['smi'])"
        "\n        mol=Chem.MolFromSmiles(smi)"
        "\n        if mol is None:"
        "\n            removed += 1"
        "\n            continue"
        "\n        canonical=Chem.MolToSmiles(mol,isomericSmiles=False)"
        "\n        ikey=str(entry['ikey'])"
        "\n        key=(ikey,canonical)"
        "\n        if key in seen:"
        "\n            continue"
        "\n        seen.add(key)"
        "\n        clean.append({'smi':canonical,'ikey':ikey})"
        "\n        kept += 1"
        "\n    if clean:"
        "\n        out[formula]=clean"
        "\npickle.dump(out,open(dst,'wb'))"
        "\nprint(f'Candidate SMILES total: {total:,}')"
        "\nprint(f'Candidate SMILES kept: {kept:,}')"
        "\nprint(f'Candidate SMILES removed as invalid: {removed:,}')"
        "\nprint(f'Formulas retained: {len(out):,}')"
    )

    run(
        [
            mist_python,
            "-c",
            sanitize_code,
        ],
        cwd=mist_root,
    )

    #
    # 3. Build the Morgan-4096 retrieval database inside the MIST
    #    environment as well. Importing MIST from the project's
    #    Python 3.14 environment is not supported.
    #
    print()
    print(
        "Building MIST Morgan-4096 retrieval HDF..."
    )

    hdf_builder_code = (
        "import pickle, sys; "
        f"sys.path.insert(0, {str(mist_root / 'src')!r}); "
        "from mist.retrieval_lib.make_hdf5 import make_retrieval_hdf5; "
        f"form_to_smi=pickle.load(open({str(SANITIZED_FORM_TO_SMI_PATH)!r}, 'rb')); "
        "make_retrieval_hdf5("
        f"labels_file={str(LABELS_PATH)!r}, "
        "form_to_smi=form_to_smi, "
        f"output_dir={str(HDF_DIR)!r}, "
        f"database_name={DATABASE_NAME!r}, "
        "fp_names=['morgan4096'])"
    )

    run(
        [
            mist_python,
            "-c",
            hdf_builder_code,
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
