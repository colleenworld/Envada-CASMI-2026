# %% [markdown]
# # CASMI 2026 — Kaggle inference submission
#
# This is the offline Kaggle entry point for the CASMI26 inference project.
#
# By default it runs the Phase 2 production pipeline:
# 1. frozen production + MSBuddy formula fusion;
# 2. formula-specific candidate retrieval augmented with the PubChem mass-tier asset;
# 3. Morgan-4096 structural-neighbor and spectrum-fingerprint scoring;
# 4. the frozen learned ranker and negative-mode retrieval fallback;
# 5. submission assembly and Kaggle-format validation.
#
# The original notebook workflow that produced the 0.403 baseline remains available as a fallback. Set `USE_PHASE2_GUARD5 = False` to run that original sequence instead.
#
# The competition reruns this notebook against a hidden replacement for `test.parquet`, so predictions must be recomputed from the attached inference assets.
#

# %% [markdown]
# ## Required Kaggle inputs
#
# Attach the competition data plus one dataset containing the frozen inference assets.
#
# The project snapshot must include the Phase 2 pipeline files and these assets:
#
# ```text
# project/
#   scripts/
#     run_kaggle_guard5_inference.py
#     build_test_multispectrum_formula_hypotheses.py
#     build_test_msbuddy_fused_formula_hypotheses.py
#     prepare_test_guard5_candidates.py
#     score_test_guard5_spectrum_fingerprint.py
#     build_test_learned_ranker_features.py
#     rank_test_learned_ranker.py
#     rank_test_negative_retrieval.py
#     build_submission.py
#   src/casmi26/
#   data/processed/
#     formula_index/
#     structure_index/
#     retrieval_index/
#     pubchem_formula_index/
#     spectrum_fingerprint/prototype/spectrum_to_morgan2048.pt
#     diagnostics/large_validation_audit/true_formula_neighbor_benchmark/
#       learned_ranker_top500/lgbm_ranker.txt
#   data/external/
#     coconut/
#     casmi26-public-assets/
#       pubchem_tier__pc_mass.npy
#       pubchem_tier__pc_off.npy
#       pubchem_tier__pc_smiles.npy
#
# MIST/
#   src/mist/
#   pretrained_models/mist_contrastive_canopus_pretrain.ckpt
#   ../wheelhouse/
# ```
#
# The offline wheelhouse beside the MIST folder must contain `msbuddy==0.3.16` and the required runtime wheels (including RDKit if it is not already available).
#
# The notebook retains the original 0.403 workflow as a reproducible fallback. The Phase 2 path is selected by default below.
#

# %%
from pathlib import Path
import os
import shutil
import subprocess
import sys
import time
import pandas as pd

KAGGLE_INPUT = Path("/kaggle/input")
WORKING = Path("/kaggle/working")
WORK_ROOT = WORKING / "casmi26"

print("Python:", sys.version)
print("Executable:", sys.executable)
print("Internet must remain disabled.")

# %%
def find_file(name, must_contain=None):
    matches = []
    for path in KAGGLE_INPUT.rglob(name):
        text = str(path)
        if must_contain is None or must_contain.lower() in text.lower():
            matches.append(path)
    if not matches:
        raise FileNotFoundError(f"Could not find {name!r} under {KAGGLE_INPUT}")
    if len(matches) > 1:
        print(f"Multiple matches for {name}:")
        for p in matches[:20]:
            print(" ", p)
    return matches[0]

TEST_SOURCE = find_file("test.parquet", "enveda")
SAMPLE_SOURCE = find_file("sample_submission.csv", "enveda")
TRAIN_SOURCE = find_file("train.parquet", "enveda")

PROJECT_SCRIPT = find_file("build_test_multispectrum_formula_hypotheses.py")
PROJECT_SOURCE = PROJECT_SCRIPT.parent.parent

MIST_ENTRY = find_file("retrieval_contrast.py")
MIST_SOURCE = MIST_ENTRY.parents[2]

print("Competition test:", TEST_SOURCE)
print("Competition sample:", SAMPLE_SOURCE)
print("Competition train:", TRAIN_SOURCE)
print("Project source:", PROJECT_SOURCE)
print("MIST source:", MIST_SOURCE)

# %%
if WORK_ROOT.exists():
    shutil.rmtree(WORK_ROOT)

shutil.copytree(PROJECT_SOURCE, WORK_ROOT, dirs_exist_ok=True)

raw_dir = WORK_ROOT / "data/raw"
raw_dir.mkdir(parents=True, exist_ok=True)

shutil.copy2(TEST_SOURCE, raw_dir / "test.parquet")
shutil.copy2(SAMPLE_SOURCE, raw_dir / "sample_submission.csv")

train_target = raw_dir / "train.parquet"
if train_target.exists() or train_target.is_symlink():
    train_target.unlink()
train_target.symlink_to(TRAIN_SOURCE)

print("Writable project:", WORK_ROOT)
print("Hidden test rows:", len(pd.read_parquet(raw_dir / "test.parquet")))
print("Sample rows:", len(pd.read_csv(raw_dir / "sample_submission.csv")))

# %% [markdown]
# ## Offline dependency install
#
# Keep internet disabled. The Phase 2 runner uses the wheelhouse beside the discovered MIST folder and requires **msbuddy==0.3.16**. This cell installs the local wheelhouse for the legacy baseline path as well; the Phase 2 runner also performs its own offline dependency setup and version check.
#

# %%
WHEELHOUSE = MIST_SOURCE.parent / "wheelhouse"

if WHEELHOUSE.is_dir():
    print("Using offline wheelhouse:", WHEELHOUSE)
    wheels = sorted(WHEELHOUSE.glob("*.whl"))
    print("Wheel count:", len(wheels))
    if wheels:
        subprocess.run(
            [
                sys.executable, "-m", "pip", "install",
                "--no-index", "--find-links", str(WHEELHOUSE),
                *[str(w) for w in wheels],
            ],
            check=True,
        )
else:
    print("No offline wheelhouse found beside MIST; the selected pipeline may fail if dependencies are absent.")

# %%
os.environ["PYTHONPATH"] = os.pathsep.join(
    [
        str(WORK_ROOT / "src"),
        str(MIST_SOURCE / "src"),
        os.environ.get("PYTHONPATH", ""),
    ]
)

sys.path.insert(0, str(WORK_ROOT / "src"))
sys.path.insert(0, str(MIST_SOURCE / "src"))

from casmi26.spectra.binning import bin_spectrum
from casmi26.chemistry.adducts import neutral_mass
print("CASMI imports: OK")

import torch
print("torch:", torch.__version__)

from mist import utils as mist_utils
print("MIST import: OK")

# %%
def run(cmd, cwd=WORK_ROOT, env=None):
    started = time.time()
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)

    print("\n$", " ".join(map(str, cmd)))
    subprocess.run(
        [str(x) for x in cmd],
        cwd=str(cwd),
        env=merged_env,
        check=True,
    )
    print(f"Finished in {(time.time() - started):.1f}s")

PYTHON = sys.executable
MIST_PYTHON = sys.executable

PIPELINE_ENV = {
    "CASMI26_PROJECT_ROOT": str(WORK_ROOT),
}

# %% [markdown]
# ## Phase 2: production formula fusion and expanded candidates
#
# This runs the repository's production inference runner. It applies the frozen MSBuddy formula fusion settings and PubChem mass-tier candidate source, then uses the existing Morgan-4096/MIST/COCONUT ranking pipeline. It does not tune on the Kaggle test set.
#
# Set `USE_PHASE2_GUARD5 = False` to reproduce the original notebook workflow that produced the 0.403 baseline.
#

# %%
USE_PHASE2_GUARD5 = True

if USE_PHASE2_GUARD5:
    command = [
        PYTHON,
        "scripts/run_kaggle_guard5_inference.py",
        "--project-root", str(WORK_ROOT),
        "--mist-root", str(MIST_SOURCE),
        "--test-path", str(TEST_SOURCE),
        "--sample-submission-path", str(SAMPLE_SOURCE),
        "--phase2",
    ]
    run(command, env=PIPELINE_ENV)

    project_submission = WORK_ROOT / "submission.csv"
    final_submission = WORKING / "submission.csv"
    if not project_submission.exists():
        raise FileNotFoundError(project_submission)
    shutil.copy2(project_submission, final_submission)
    print("Phase 2 submission copied to:", final_submission)

# %% [markdown]
# ## Legacy baseline pipeline: original 0.403 workflow
#
# These cells are retained unchanged in behavior for reproduction. They run only when `USE_PHASE2_GUARD5 = False`.
#

# %%
if not USE_PHASE2_GUARD5:
    run(
        [PYTHON, "scripts/build_test_multispectrum_formula_hypotheses.py"],
        env=PIPELINE_ENV,
    )

# %% [markdown]
# ## Legacy baseline — 2. Export MIST test inputs

# %%
if not USE_PHASE2_GUARD5:
    run(
        [PYTHON, "scripts/export_test_mist_inputs.py"],
        env=PIPELINE_ENV,
    )

# %% [markdown]
# ## Legacy baseline — 3. Keep only MIST-supported positive-ion hypotheses
#
# The pretrained checkpoint is positive-ion only. `[M+NH4]+` is remapped to MIST's internal `[M+H3N+H]+`. Negative-only molecules are handled later by the frozen spectral-retrieval fallback.
#

# %%
if not USE_PHASE2_GUARD5:
    root = WORK_ROOT / "data/processed/test"
    
    hyp = pd.read_parquet(root / "multispectrum_formula_hypotheses.parquet")
    labels = pd.read_csv(root / "mist/test_multiformula_labels.tsv", sep="\t")
    
    supported_adducts = {
        "[M+H]+",
        "[M+Na]+",
        "[M+NH4]+",
        "[M+K]+",
    }
    
    supported_specs = set(
        hyp.loc[
            hyp["selected_adduct"].isin(supported_adducts),
            "hypothesis_spec",
        ].astype(str)
    )
    
    filtered = labels[
        labels["spec"].astype(str).isin(supported_specs)
    ].copy()
    
    filtered["ionization"] = filtered["ionization"].replace(
        {"[M+NH4]+": "[M+H3N+H]+"}
    )
    
    labels_positive = root / "mist/test_multiformula_labels_mist_supported.tsv"
    filtered.to_csv(labels_positive, sep="\t", index=False)
    
    supported_molecules = hyp.loc[
        hyp["hypothesis_spec"].isin(filtered["spec"]),
        "molecule_id",
    ].nunique()
    
    print("Original hypotheses:", len(labels))
    print("MIST-supported hypotheses:", len(filtered))
    print("Molecules represented:", supported_molecules)
    print(filtered["ionization"].value_counts().to_string())

# %% [markdown]
# ## Legacy baseline — 4. Generate positive-ion subformula assignments

# %%
if not USE_PHASE2_GUARD5:
    subforms_positive = root / "mist/subforms_positive"
    if subforms_positive.exists():
        shutil.rmtree(subforms_positive)
    
    run(
        [
            MIST_PYTHON,
            str(MIST_SOURCE / "src/mist/subformulae/assign_subformulae.py"),
            "--spec-files", str(root / "mist/test_multiformula.mgf"),
            "--labels-file", str(labels_positive),
            "--output-dir", str(subforms_positive),
            "--feature-id", "FEATURE_ID",
            "--num-workers", "0",
        ],
        cwd=MIST_SOURCE,
    )
    
    json_count = len(list(subforms_positive.glob("*.json")))
    print("Positive subformula JSON files:", json_count)
    assert json_count == len(filtered)

# %% [markdown]
# ## Legacy baseline — 5. Build the full test-time structure candidate universe

# %%
if not USE_PHASE2_GUARD5:
    run(
        [PYTHON, "scripts/build_test_contrastive_candidates.py"],
        env=PIPELINE_ENV,
    )
    
    h5_files = list(
        (root / "contrastive_retrieval").glob("*morgan4096*retrieval_db.h5")
    )
    if len(h5_files) != 1:
        raise RuntimeError(
            f"Expected one Morgan-4096 retrieval HDF, found {h5_files}"
        )
    
    HDF_PATH = h5_files[0]
    print("Contrastive HDF:", HDF_PATH)

# %% [markdown]
# ## Legacy baseline — 6. Positive-ion contrastive MIST retrieval

# %%
if not USE_PHASE2_GUARD5:
    positive_predictions = root / "contrastive_predictions_positive"
    if positive_predictions.exists():
        shutil.rmtree(positive_predictions)
    
    checkpoint = MIST_SOURCE / "pretrained_models/mist_contrastive_canopus_pretrain.ckpt"
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    
    run(
        [
            MIST_PYTHON,
            str(MIST_SOURCE / "src/mist/retrieval_contrast.py"),
            "--model-ckpt", str(checkpoint),
            "--dataset-name", "casmi_test_positive_contrastive",
            "--labels-file", str(labels_positive),
            "--subform-folder", str(subforms_positive),
            "--hdf-file", str(HDF_PATH),
            "--save-dir", str(positive_predictions),
            "--dist-name", "cosine",
            "--num-workers", "0",
            "--top-k", "500",
        ],
        cwd=MIST_SOURCE,
    )

# %% [markdown]
# ## Legacy baseline — 7. Rank positive-ion candidates

# %%
if not USE_PHASE2_GUARD5:
    run(
        [PYTHON, "scripts/rank_test_positive_contrastive.py"],
        env=PIPELINE_ENV,
    )

# %% [markdown]
# ## Legacy baseline — 8. Rank negative-only molecules with frozen spectral retrieval
#
# This branch uses same-polarity retrieval with neutral-mass filtering at ±0.01 Da, falling back to unrestricted same-negative-mode cosine only if no mass-filtered candidates exist.
#

# %%
if not USE_PHASE2_GUARD5:
    run(
        [PYTHON, "scripts/rank_test_negative_retrieval.py"],
        env=PIPELINE_ENV,
    )

# %% [markdown]
# ## Legacy baseline — 9. Assemble `submission.csv`

# %%
if not USE_PHASE2_GUARD5:
    run(
        [PYTHON, "scripts/build_submission.py"],
        env=PIPELINE_ENV,
    )
    
    project_submission = WORK_ROOT / "submission.csv"
    final_submission = WORKING / "submission.csv"
    shutil.copy2(project_submission, final_submission)
    
    print("Submission copied to:", final_submission)

# %% [markdown]
# ## 10. Final Kaggle-format validation

# %%
submission = pd.read_csv(WORKING / "submission.csv")
sample = pd.read_csv(raw_dir / "sample_submission.csv")

assert list(submission.columns) == ["molecule_id", "smiles"]
assert len(submission) == len(sample)
assert submission["molecule_id"].is_unique
assert submission["smiles"].notna().all()
assert submission["molecule_id"].tolist() == sample["molecule_id"].astype(str).tolist()

candidate_counts = submission["smiles"].str.split(";").str.len()
assert candidate_counts.between(1, 25).all()

print("Rows:", len(submission))
print("Unique molecule IDs:", submission["molecule_id"].nunique())
print("Candidate-count distribution:")
print(candidate_counts.describe().to_string())
print()
print("submission.csv is ready at /kaggle/working/submission.csv")

# %% [markdown]
# ## Commit / submit checklist
#
# Before the final Kaggle commit:
#
# - Internet: **Off**
# - Runtime: **under 9 hours**
# - Competition dataset attached
# - Updated inference-assets dataset attached (including the three PubChem mass-tier arrays and the learned-ranker/model assets listed above)
# - Offline wheelhouse beside MIST includes **msbuddy==0.3.16**
# - `USE_PHASE2_GUARD5 = True` for the Phase 2 pipeline; set it to `False` only to reproduce the original 0.403 notebook workflow
# - `/kaggle/working/submission.csv` produced and validated successfully
# - Do not use a precomputed visible-test submission: Kaggle replaces `test.parquet` during the hidden rerun
#
