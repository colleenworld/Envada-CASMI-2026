#!/usr/bin/env bash
set -euo pipefail

# Refresh the local Kaggle casmi26-inference-assets dataset without replacing
# the existing MIST checkout/checkpoint, retrieval indexes, COCONUT data, or
# wheelhouse. By default this only prepares and validates the dataset.
#
# Usage:
#   scripts/refresh_kaggle_inference_assets.sh [--publish]
#       [--repo PATH] [--assets PATH]
#
# Defaults:
#   repo   = this script's Git repository
#   assets = ~/casmi26-inference-assets
#
# Kaggle's current asset wheelhouse contains a CPython 3.13 RDKit wheel, so
# CPython 3.13 / manylinux x86_64 is the default target for offline wheels.
# Override with KAGGLE_PYTHON_VERSION, KAGGLE_PYTHON_ABI, and
# KAGGLE_MANYLINUX_PLATFORM if the Kaggle notebook runtime changes.

DATASET_ID="colleendavis/casmi26-inference-assets"
PUBLISH=0
REPO_ARG=""
ASSETS_ARG=""

while (($#)); do
  case "$1" in
    --publish)
      PUBLISH=1
      shift
      ;;
    --repo)
      REPO_ARG="${2:?Missing path after --repo}"
      shift 2
      ;;
    --assets)
      ASSETS_ARG="${2:?Missing path after --assets}"
      shift 2
      ;;
    -h|--help)
      sed -n '1,34p' "$0"
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ARG:-$(cd "$SCRIPT_DIR/.." && pwd)}"
ASSET_ROOT="${ASSETS_ARG:-$HOME/casmi26-inference-assets}"
REPO_ROOT="$(cd "$REPO_ROOT" && pwd)"
ASSET_ROOT="$(cd "$ASSET_ROOT" && pwd)"

PROJECT_ROOT="$ASSET_ROOT/project"
MIST_ROOT="$ASSET_ROOT/MIST"
WHEELHOUSE="$ASSET_ROOT/wheelhouse"
PYTHON="${PYTHON:-python3}"
KAGGLE_PYTHON_VERSION="${KAGGLE_PYTHON_VERSION:-3.13}"
KAGGLE_PYTHON_ABI="${KAGGLE_PYTHON_ABI:-cp313}"
KAGGLE_MANYLINUX_PLATFORM="${KAGGLE_MANYLINUX_PLATFORM:-manylinux_2_28_x86_64}"
WHEEL_BUILD_IMAGE="${WHEEL_BUILD_IMAGE:-quay.io/pypa/manylinux_2_28_x86_64}"
MESSAGE="Refresh CASMI26 Phase 2 inference assets (MSBuddy 0.3.16)"

fail() {
  echo "ERROR: $*" >&2
  exit 1
}

echo "Repository:  $REPO_ROOT"
echo "Assets:      $ASSET_ROOT"
echo "Dataset ID:  $DATASET_ID"

[[ -d "$PROJECT_ROOT" ]] || fail "Missing existing project snapshot: $PROJECT_ROOT"
[[ -d "$MIST_ROOT" ]] || fail "Missing existing MIST directory: $MIST_ROOT"
[[ -d "$WHEELHOUSE" ]] || fail "Missing existing wheelhouse: $WHEELHOUSE"
[[ -f "$ASSET_ROOT/dataset-metadata.json" ]] || fail "Missing dataset-metadata.json in $ASSET_ROOT"

"$PYTHON" - "$ASSET_ROOT/dataset-metadata.json" "$DATASET_ID" <<'PY'
import json
import sys
from pathlib import Path

metadata_path = Path(sys.argv[1])
expected_id = sys.argv[2]
metadata = json.loads(metadata_path.read_text())
actual_id = metadata.get("id")
if actual_id != expected_id:
    raise SystemExit(
        f"Dataset metadata id is {actual_id!r}; expected {expected_id!r}. "
        "Refusing to publish to a different dataset."
    )
print(f"Dataset metadata verified: {actual_id}")
PY

# Preserve and verify the large assets already in the dataset.
PRESERVED_FILES=(
  "data/processed/formula_index/training_coconut_formulas.parquet"
  "data/processed/structure_index/structures.parquet"
  "data/processed/retrieval_index/metadata.parquet"
  "data/processed/retrieval_index/vectors.npz"
  "data/external/coconut/coconut_approved.parquet"
)
for relative in "${PRESERVED_FILES[@]}"; do
  [[ -f "$PROJECT_ROOT/$relative" ]] || fail "Existing project asset is missing: project/$relative"
done

[[ -f "$MIST_ROOT/src/mist/subformulae/assign_subformulae.py" ]] || fail "MIST source is missing"
[[ -f "$MIST_ROOT/src/mist/retrieval_contrast.py" ]] || fail "MIST contrastive retrieval source is missing"
[[ -f "$MIST_ROOT/pretrained_models/mist_contrastive_canopus_pretrain.ckpt" ]] || fail "MIST checkpoint is missing"

# Copy the current repository code into the project snapshot. This deliberately
# does not use --delete and does not touch project/data, MIST, or existing wheels.
[[ -d "$REPO_ROOT/scripts" ]] || fail "Repository scripts directory is missing"
[[ -d "$REPO_ROOT/src/casmi26" ]] || fail "Repository src/casmi26 directory is missing"
mkdir -p "$PROJECT_ROOT/scripts" "$PROJECT_ROOT/src/casmi26"

if command -v rsync >/dev/null 2>&1; then
  rsync -a "$REPO_ROOT/scripts/" "$PROJECT_ROOT/scripts/"
  rsync -a "$REPO_ROOT/src/casmi26/" "$PROJECT_ROOT/src/casmi26/"
else
  cp -a "$REPO_ROOT/scripts/." "$PROJECT_ROOT/scripts/"
  cp -a "$REPO_ROOT/src/casmi26/." "$PROJECT_ROOT/src/casmi26/"
fi

PHASE2_FILES=(
  "scripts/build_test_multispectrum_formula_hypotheses.py"
  "scripts/build_test_msbuddy_fused_formula_hypotheses.py"
  "scripts/export_test_mist_inputs.py"
  "scripts/filter_test_mist_supported.py"
  "scripts/build_test_contrastive_candidates.py"
  "scripts/rank_test_positive_contrastive.py"
  "scripts/rank_test_negative_retrieval.py"
  "scripts/build_submission.py"
  "scripts/run_kaggle_phase2_inference.py"
  "src/casmi26/chemistry/inference.py"
  "src/casmi26/pubchem/mass_tier.py"
  "src/casmi26/pubchem/mass_tier_candidates.py"
)
for relative in "${PHASE2_FILES[@]}"; do
  [[ -f "$PROJECT_ROOT/$relative" ]] || fail "Phase 2 file did not copy: project/$relative"
done

echo
echo "Phase 2 project code copied and verified."

# brain-isotopic-distribution has no CPython 3.13 Linux x86_64 wheel.
# Build a wheel for the Kaggle ABI in the official manylinux image, then let
# pip resolve MSBuddy and its remaining dependencies using the completed wheelhouse.
BRAIN_WHEEL="$(find "$WHEELHOUSE" -maxdepth 1 -type f \
  -name "brain_isotopic_distribution-*-${KAGGLE_PYTHON_ABI}-${KAGGLE_PYTHON_ABI}-${KAGGLE_MANYLINUX_PLATFORM}.whl" \
  -print -quit)"
if [[ -z "$BRAIN_WHEEL" ]]; then
  command -v docker >/dev/null 2>&1 || fail "Docker is required to build the CPython ${KAGGLE_PYTHON_ABI} brain-isotopic-distribution wheel"
  echo "Downloading brain-isotopic-distribution source distribution..."
  "$PYTHON" -m pip download \
    --dest "$WHEELHOUSE" \
    --no-deps \
    --no-binary=:all: \
    brain-isotopic-distribution

  BRAIN_SDIST="$(find "$WHEELHOUSE" -maxdepth 1 -type f -name 'brain_isotopic_distribution-*.tar.gz' -print -quit)"
  [[ -n "$BRAIN_SDIST" ]] || fail "brain-isotopic-distribution source archive was not downloaded"

  echo "Building brain-isotopic-distribution for ${KAGGLE_PYTHON_ABI} in $WHEEL_BUILD_IMAGE..."
  docker run --rm --platform linux/amd64 \
    -v "$WHEELHOUSE:/wheelhouse" \
    "$WHEEL_BUILD_IMAGE" \
    bash -lc "
      set -euo pipefail
      TARGET_PYTHON='/opt/python/${KAGGLE_PYTHON_ABI}-${KAGGLE_PYTHON_ABI}/bin/python'
      [[ -x \"\$TARGET_PYTHON\" ]] || { echo \"Missing target interpreter: \$TARGET_PYTHON\" >&2; exit 1; }
      mkdir -p /tmp/brain-wheel
      \"\$TARGET_PYTHON\" -m pip wheel --no-deps --wheel-dir /tmp/brain-wheel /wheelhouse/$(basename "$BRAIN_SDIST")
      auditwheel repair --plat '${KAGGLE_MANYLINUX_PLATFORM}' --wheel-dir /wheelhouse /tmp/brain-wheel/brain_isotopic_distribution-*.whl
      \"\$TARGET_PYTHON\" -m pip install --no-deps --no-index --find-links /wheelhouse brain-isotopic-distribution
      \"\$TARGET_PYTHON\" -c 'from brainpy import isotopic_variants; peaks = isotopic_variants({\"C\": 6, \"H\": 12, \"O\": 6}, npeaks=3); assert len(peaks) == 3; print(\"brainpy smoke test passed\")'
    "
  rm -f "$BRAIN_SDIST"
fi

BRAIN_WHEEL="$(find "$WHEELHOUSE" -maxdepth 1 -type f \
  -name "brain_isotopic_distribution-*-${KAGGLE_PYTHON_ABI}-${KAGGLE_PYTHON_ABI}-${KAGGLE_MANYLINUX_PLATFORM}.whl" \
  -print -quit)"
[[ -n "$BRAIN_WHEEL" ]] || fail "No brain-isotopic-distribution wheel was built for ${KAGGLE_PYTHON_ABI} / ${KAGGLE_MANYLINUX_PLATFORM}"

# pip download appends wheels; it does not remove or replace the existing
# RDKit/MIST wheelhouse contents. The local brainpy wheel satisfies MSBuddy's
# dependency while the remaining packages are resolved from binary wheels.
"$PYTHON" -m pip download \
  --dest "$WHEELHOUSE" \
  --find-links "$WHEELHOUSE" \
  --only-binary=:all: \
  --python-version "$KAGGLE_PYTHON_VERSION" \
  --implementation cp \
  --abi "$KAGGLE_PYTHON_ABI" \
  --abi abi3 \
  --abi none \
  --platform "$KAGGLE_MANYLINUX_PLATFORM" \
  --platform manylinux2014_x86_64 \
  "msbuddy==0.3.16" \
  lightgbm

MSBUDDY_WHEEL="$(find "$WHEELHOUSE" -maxdepth 1 -type f -name 'msbuddy-0.3.16-*.whl' -print -quit)"
[[ -n "$MSBUDDY_WHEEL" ]] || fail "MSBuddy 0.3.16 wheel was not downloaded"

# MSBuddy's package page lists Python <3.13 as its supported range. Kaggle's
# current image is Python 3.13, so verify the exact imports used by the runner
# in the matching target interpreter before allowing publication.
echo "Checking MSBuddy 0.3.16 imports under ${KAGGLE_PYTHON_ABI}..."
docker run --rm --platform linux/amd64 \
  -v "$WHEELHOUSE:/wheelhouse" \
  "$WHEEL_BUILD_IMAGE" \
  bash -lc "
    set -euo pipefail
    TARGET_PYTHON='/opt/python/${KAGGLE_PYTHON_ABI}-${KAGGLE_PYTHON_ABI}/bin/python'
    [[ -x \"\$TARGET_PYTHON\" ]] || { echo \"Missing target interpreter: \$TARGET_PYTHON\" >&2; exit 1; }
    \"\$TARGET_PYTHON\" -m pip install --no-index --find-links /wheelhouse 'msbuddy==0.3.16' lightgbm
    \"\$TARGET_PYTHON\" -c 'from msbuddy import Msbuddy, MsbuddyConfig; from msbuddy.base import MetaFeature, Spectrum; assert all((Msbuddy, MsbuddyConfig, MetaFeature, Spectrum))'
  "
echo "MSBuddy 0.3.16 imports verified under ${KAGGLE_PYTHON_ABI}."

echo
echo "MSBuddy wheel: $MSBUDDY_WHEEL"
echo "Wheelhouse now contains $(find "$WHEELHOUSE" -maxdepth 1 -type f -name '*.whl' | wc -l) wheels."

# Final asset preflight mirrors the inputs required by the Phase 2 runner.
for relative in "${PHASE2_FILES[@]}"; do
  [[ -f "$PROJECT_ROOT/$relative" ]] || fail "Final preflight failed: project/$relative"
done
for filename in \
  pubchem_tier__pc_mass.npy \
  pubchem_tier__pc_off.npy \
  pubchem_tier__pc_smiles.npy; do
  echo "Separate PubChem input required in Kaggle: $filename"
done

echo
echo "Prepared dataset successfully."
echo "Existing MIST, checkpoint, indexes, COCONUT data, and wheelhouse files were preserved."
echo "Repository commit: $(git -C "$REPO_ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown)"

if ((PUBLISH)); then
  command -v kaggle >/dev/null 2>&1 || fail "Kaggle CLI is not installed"
  echo
  echo "Publishing a new Kaggle dataset version..."
  kaggle datasets version -p "$ASSET_ROOT" -m "$MESSAGE" -r zip
else
  echo
  echo "Not published. Review the prepared directory, then run:"
  printf '  kaggle datasets version -p %q -m %q -r zip\n' "$ASSET_ROOT" "$MESSAGE"
  echo "Or rerun this script with --publish to publish after preparation."
fi
