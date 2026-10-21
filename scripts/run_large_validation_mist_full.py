from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

BENCHMARK_DIR = (
    PROJECT_ROOT
    / "data/processed/diagnostics/large_validation_audit/"
    "true_formula_neighbor_benchmark/mist_full"
)

MGF_PATH = BENCHMARK_DIR / "benchmark.mgf"
LABELS_PATH = BENCHMARK_DIR / "labels.tsv"
FORM_TO_SMI_PATH = BENCHMARK_DIR / "form_to_smi.pkl"
SANITIZED_PATH = BENCHMARK_DIR / "form_to_smi_sanitized.pkl"
SUBFORMS_DIR = BENCHMARK_DIR / "subforms"
HDF_DIR = BENCHMARK_DIR / "retrieval"
PREDICTION_DIR = BENCHMARK_DIR / "predictions"

DATABASE_NAME = "large_validation_full_pubchem_morgan4096"
DATASET_NAME = "large_validation_full_pubchem_positive"


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
        return explicit.expanduser().resolve()

    conda = shutil.which("conda")

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

        payload = json.loads(completed.stdout)

        for env_path in payload.get("envs", []):
            env_path = Path(env_path)

            if env_path.name == "mist-cpu":
                candidate = env_path / "bin/python"

                if candidate.exists():
                    return candidate

    for candidate in [
        Path.home()
        / "miniconda3/envs/mist-cpu/bin/python",
        Path.home()
        / "anaconda3/envs/mist-cpu/bin/python",
    ]:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "Could not locate mist-cpu Python. "
        "Pass --mist-python explicitly."
    )


def run(command: list[str], cwd: Path) -> None:
    print()
    print("+", " ".join(command))
    subprocess.run(
        command,
        cwd=cwd,
        check=True,
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
        / "src/mist/subformulae/assign_subformulae.py"
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
            "Missing inputs: "
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

    for path in HDF_DIR.glob("*"):
        if path.is_file():
            path.unlink()

    for path in PREDICTION_DIR.glob(
        "retrieval_contrastive_*.p"
    ):
        path.unlink()

    started = time.perf_counter()

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
        mist_root,
    )

    sanitizer = (
        PROJECT_ROOT
        / "scripts/sanitize_mist_candidates_resumable.py"
    )
    sanitize_cache_dir = (
        BENCHMARK_DIR
        / "sanitize_cache"
    )

    if not sanitizer.exists():
        raise FileNotFoundError(
            f"Missing sanitizer: {sanitizer}"
        )

    run(
        [
            mist_python,
            str(sanitizer),
            "--input",
            str(FORM_TO_SMI_PATH),
            "--output",
            str(SANITIZED_PATH),
            "--cache-dir",
            str(sanitize_cache_dir),
        ],
        mist_root,
    )

    build_hdf = (
        "import pickle,sys; "
        f"sys.path.insert(0,{str(mist_root / 'src')!r}); "
        "import mist.retrieval_lib.make_hdf5 as mh; "
        "from mist import utils; "
        f"mh.CHUNKSIZE={args.hdf_formulas_per_batch!r}; "
        "_orig=utils.chunked_parallel; "
        f"_workers={args.hdf_workers!r}; "
        "def _limited(items, fn, *a, **kw): "
        "    kw['max_cpu']=_workers; "
        "    return _orig(items, fn, *a, **kw); "
        "utils.chunked_parallel=_limited; "
        f"m=pickle.load(open({str(SANITIZED_PATH)!r},'rb')); "
        "mh.make_retrieval_hdf5("
        f"labels_file={str(LABELS_PATH)!r},"
        "form_to_smi=m,"
        f"output_dir={str(HDF_DIR)!r},"
        f"database_name={DATABASE_NAME!r},"
        "fp_names=['morgan4096'])"
    )

    run(
        [
            mist_python,
            "-c",
            build_hdf,
        ],
        mist_root,
    )

    hdf_files = sorted(
        HDF_DIR.glob(
            "*morgan4096*retrieval_db.h5"
        )
    )

    if len(hdf_files) != 1:
        raise RuntimeError(
            f"Expected one HDF, found {hdf_files}"
        )

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
            str(hdf_files[0]),
            "--dist-name",
            "cosine",
            "--num-workers",
            "0",
            "--labels-file",
            str(LABELS_PATH),
            "--top-k",
            str(args.top_k),
        ],
        mist_root,
    )

    predictions = sorted(
        PREDICTION_DIR.glob(
            "retrieval_contrastive_*.p"
        )
    )

    if len(predictions) != 1:
        raise RuntimeError(
            f"Expected one prediction artifact, found {predictions}"
        )

    print()
    print(
        "Full-candidate MIST benchmark complete"
    )
    print(
        "======================================"
    )
    print(
        "Elapsed seconds:",
        f"{time.perf_counter() - started:.1f}",
    )
    print(
        "Prediction artifact:",
        predictions[0],
    )


if __name__ == "__main__":
    main()
