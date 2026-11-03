# %% [markdown]
# # CASMI 2026 — Kaggle inference submission
#
# This is the offline Kaggle entry point for the CASMI26 inference project.
#
# By default it runs Phase 2 on the original 0.403 workflow:
#
# 1. Frozen production formula hypotheses plus MSBuddy formula fusion.
# 2. Candidate structures from training, approved COCONUT, and PubChem mass-tier.
# 3. The original MIST contrastive retrieval using Morgan-4096 fingerprints.
# 4. The unchanged positive ranking and same-polarity retrieval fallback.
# 5. Submission assembly and Kaggle-format validation.
#
# The original notebook workflow that produced 0.403 remains available as a fallback. Set `USE_PHASE2 = False` to reproduce it.
#
# The competition reruns this notebook against a hidden replacement for `test.parquet`, so predictions are recomputed from the attached inference assets.

# %% [markdown]
# ## Required Kaggle inputs
#
# Attach these three Kaggle inputs:
#
# - The Enveda CASMI26 competition dataset.
# - `casmi26-inference-assets`, containing the project snapshot and MIST folder.
# - `casmi26-pubchem-tier`, containing the three PubChem mass-tier arrays.
#
# The project snapshot must be refreshed from branch `phase2-production-msbuddy-mass-tier` and contain:
#
# ```text
# project/scripts/
#   build_test_multispectrum_formula_hypotheses.py
#   build_test_msbuddy_fused_formula_hypotheses.py
#   export_test_mist_inputs.py
#   filter_test_mist_supported.py
#   build_test_contrastive_candidates.py
#   rank_test_positive_contrastive.py
#   rank_test_negative_retrieval.py
#   build_submission.py
#   run_kaggle_phase2_inference.py
# project/src/casmi26/
#   chemistry/inference.py
#   pubchem/mass_tier.py
#   pubchem/mass_tier_candidates.py
# project/data/processed/
#   formula_index/training_coconut_formulas.parquet
#   structure_index/structures.parquet
#   retrieval_index/metadata.parquet
#   retrieval_index/vectors.npz
# project/data/external/coconut/coconut_approved.parquet
# MIST/src/mist/
# MIST/pretrained_models/mist_contrastive_canopus_pretrain.ckpt
# wheelhouse/rdkit-*.whl
# wheelhouse/msbuddy-0.3.16-*.whl
# ```
#
# The PubChem mass-tier arrays remain in the separate `casmi26-pubchem-tier` dataset; the notebook passes that mounted directory directly to the runner.
#
# The runner requires exactly `msbuddy==0.3.16`. If it is not already installed, its wheel must be present in the offline wheelhouse. No internet access is required.

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
USE_PHASE2 = True

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

PROJECT_SCRIPT = find_file(
    "build_test_multispectrum_formula_hypotheses.py",
    "casmi26-inference-assets",
)
PROJECT_SOURCE = PROJECT_SCRIPT.parent.parent

MIST_SOURCE = PROJECT_SOURCE.parent / "MIST"
if not MIST_SOURCE.is_dir():
    raise FileNotFoundError(f"Expected MIST beside project snapshot: {MIST_SOURCE}")

MASS_TIER_ENTRY = find_file("pubchem_tier__pc_mass.npy")
PUBCHEM_MASS_TIER_ROOT = MASS_TIER_ENTRY.parent
for asset_name in ("pubchem_tier__pc_off.npy", "pubchem_tier__pc_smiles.npy"):
    if not (PUBCHEM_MASS_TIER_ROOT / asset_name).is_file():
        raise FileNotFoundError(PUBCHEM_MASS_TIER_ROOT / asset_name)

PHASE2_REQUIRED_PROJECT_FILES = [
    "scripts/run_kaggle_phase2_inference.py",
    "scripts/build_test_msbuddy_fused_formula_hypotheses.py",
    "scripts/filter_test_mist_supported.py",
    "scripts/build_test_contrastive_candidates.py",
    "scripts/export_test_mist_inputs.py",
    "scripts/rank_test_positive_contrastive.py",
    "src/casmi26/chemistry/inference.py",
    "src/casmi26/pubchem/mass_tier.py",
    "src/casmi26/pubchem/mass_tier_candidates.py",
]
missing_phase2_code = [
    PROJECT_SOURCE / relative
    for relative in PHASE2_REQUIRED_PROJECT_FILES
    if not (PROJECT_SOURCE / relative).is_file()
]
if missing_phase2_code:
    raise RuntimeError(
        "The attached casmi26-inference-assets project snapshot is older "
        "than Phase 2. Publish a new version with project/scripts and "
        "project/src from branch phase2-production-msbuddy-mass-tier. "
        "Missing files:\n"
        + "\n".join(str(path) for path in missing_phase2_code)
    )

MSBUDDY_WHEELS = sorted(KAGGLE_INPUT.rglob("msbuddy-0.3.16*.whl"))

print("Competition test:", TEST_SOURCE)
print("Competition sample:", SAMPLE_SOURCE)
print("Competition train:", TRAIN_SOURCE)
print("Project source:", PROJECT_SOURCE)
print("MIST source:", MIST_SOURCE)
print("PubChem mass-tier root:", PUBCHEM_MASS_TIER_ROOT)
print("MSBuddy wheels found:", len(MSBUDDY_WHEELS))

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
print("PubChem mass-tier assets will be read from:", PUBCHEM_MASS_TIER_ROOT)

# %% [markdown]
# ## Offline dependency install
#
# Keep internet disabled. The Phase 2 runner installs only the required offline wheels immediately before inference. The wheelhouse beside MIST must include `msbuddy==0.3.16`; RDKit is also required. The legacy fallback retains its original wheelhouse setup.

# %%
WHEELHOUSE = MIST_SOURCE.parent / "wheelhouse"

if WHEELHOUSE.is_dir():
    print("Offline wheelhouse:", WHEELHOUSE)
    wheels = sorted(WHEELHOUSE.glob("*.whl"))
    print("Wheel count:", len(wheels))
    if USE_PHASE2:
        print("The Phase 2 runner installs/checks the required offline dependencies before inference.")
        print("MSBuddy wheel found:", bool(MSBUDDY_WHEELS))
    elif wheels:
        subprocess.run(
            [
                sys.executable, "-m", "pip", "install",
                "--no-index", "--find-links", str(WHEELHOUSE),
                *[str(w) for w in wheels],
            ],
            check=True,
        )
else:
    print("No offline wheelhouse found beside MIST; required packages must already be installed.")

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

print("Project source path:", WORK_ROOT / "src")
print("MIST source path:", MIST_SOURCE / "src")
print("Dependency installation and runtime imports are handled by the selected pipeline.")

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
# ## Phase 2: frozen formula fusion + original MIST/Morgan-4096 retrieval
#
# This runs the original 0.403 pipeline with two additions: frozen MSBuddy formula fusion and PubChem mass-tier structures in the existing formula-specific candidate universe.
#
# The MIST checkpoint, Morgan-4096 retrieval, ranking weights, and negative-mode fallback are unchanged.
#
# Set `USE_PHASE2 = False` to reproduce the original 0.403 workflow.

# %%
if USE_PHASE2:
    command = [
        PYTHON,
        "scripts/run_kaggle_phase2_inference.py",
        "--project-root", str(WORK_ROOT),
        "--mist-root", str(MIST_SOURCE),
        "--test-path", str(TEST_SOURCE),
        "--sample-submission-path", str(SAMPLE_SOURCE),
        "--pubchem-mass-tier-root", str(PUBCHEM_MASS_TIER_ROOT),
        "--wheelhouse", str(WHEELHOUSE),
    ]
    if MSBUDDY_WHEELS:
        command.extend(["--msbuddy-wheel", str(MSBUDDY_WHEELS[0])])
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
# These cells retain the original 0.403 workflow. They run only when `USE_PHASE2 = False`.

# %%
if not USE_PHASE2:
    run(
        [PYTHON, "scripts/build_test_multispectrum_formula_hypotheses.py"],
        env=PIPELINE_ENV,
    )

# %% [markdown]
# ## Legacy baseline — 2. Export MIST test inputs

# %%
if not USE_PHASE2:
    run(
        [PYTHON, "scripts/export_test_mist_inputs.py"],
        env=PIPELINE_ENV,
    )

# %% [markdown]
# ## Legacy baseline — 3. Keep only MIST-supported positive-ion hypotheses
#
# The pretrained checkpoint is positive-ion only. `[M+NH4]+` is remapped to MIST's internal `[M+H3N+H]+`. Negative-only molecules are handled later by the frozen spectral-retrieval fallback.

# %%
if not USE_PHASE2:
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
if not USE_PHASE2:
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
if not USE_PHASE2:
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
if not USE_PHASE2:
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
if not USE_PHASE2:
    run(
        [PYTHON, "scripts/rank_test_positive_contrastive.py"],
        env=PIPELINE_ENV,
    )

# %% [markdown]
# ## Legacy baseline — 8. Rank negative-only molecules with frozen spectral retrieval
#
# This branch uses same-polarity retrieval with neutral-mass filtering at ±0.01 Da, falling back to unrestricted same-negative-mode cosine only if no mass-filtered candidates exist.

# %%
if not USE_PHASE2:
    run(
        [PYTHON, "scripts/rank_test_negative_retrieval.py"],
        env=PIPELINE_ENV,
    )

# %% [markdown]
# ## Legacy baseline — 9. Assemble `submission.csv`

# %%
if not USE_PHASE2:
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
# Before the final Kaggle commit, confirm the refreshed inference-assets dataset includes the Phase 2 source files and the offline MSBuddy 0.3.16 wheel, and that the separate PubChem mass-tier dataset is attached:
#
# - Internet: **Off**
# - Runtime: **under 9 hours**
# - Competition dataset attached
# - Updated inference-assets dataset attached (including the three PubChem mass-tier arrays and the learned-ranker/model assets listed above)
# - Offline wheelhouse beside MIST includes **msbuddy==0.3.16**
# - `USE_PHASE2 = True` for the Phase 2 pipeline; set it to `False` only to reproduce the original 0.403 notebook workflow
# - `/kaggle/working/submission.csv` produced and validated successfully
# - Do not use a precomputed visible-test submission: Kaggle replaces `test.parquet` during the hidden rerun

