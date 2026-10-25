#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${CASMI26_PROJECT_ROOT:-/home/colleen/PycharmProjects/Envada-CASMI-2026}"
MIST_ROOT="${MIST_ROOT:-/home/colleen/PycharmProjects/MIST}"
ASSET_ROOT="$PROJECT_ROOT/kaggle/casmi26-inference-assets"

cd "$PROJECT_ROOT"

rm -rf "$ASSET_ROOT/project" "$ASSET_ROOT/MIST" "$ASSET_ROOT/wheelhouse"

mkdir -p "$ASSET_ROOT/project"
mkdir -p "$ASSET_ROOT/MIST"
mkdir -p "$ASSET_ROOT/wheelhouse"

cp -a scripts "$ASSET_ROOT/project/"
cp -a src "$ASSET_ROOT/project/"

mkdir -p "$ASSET_ROOT/project/data/processed"

for required_dir in \
  data/processed/formula_index \
  data/processed/structure_index \
  data/processed/retrieval_index \
  data/processed/pubchem_formula_index
do
  if [ ! -d "$required_dir" ]; then
    echo "Missing $required_dir" >&2
    exit 1
  fi
done

cp -a data/processed/formula_index \
      "$ASSET_ROOT/project/data/processed/"

cp -a data/processed/structure_index \
      "$ASSET_ROOT/project/data/processed/"

cp -a data/processed/retrieval_index \
      "$ASSET_ROOT/project/data/processed/"

cp -a data/processed/pubchem_formula_index \
      "$ASSET_ROOT/project/data/processed/"

FINGERPRINT_MODEL="data/processed/spectrum_fingerprint/prototype/spectrum_to_morgan2048.pt"

if [ ! -f "$FINGERPRINT_MODEL" ]; then
  echo "Missing $FINGERPRINT_MODEL" >&2
  exit 1
fi

mkdir -p "$ASSET_ROOT/project/data/processed/spectrum_fingerprint/prototype"

cp "$FINGERPRINT_MODEL" \
   "$ASSET_ROOT/project/data/processed/spectrum_fingerprint/prototype/"

mkdir -p "$ASSET_ROOT/project/data/external"

if [ ! -d data/external/coconut ]; then
  echo "Missing data/external/coconut" >&2
  exit 1
fi

cp -a data/external/coconut \
      "$ASSET_ROOT/project/data/external/"

if [ ! -d "$MIST_ROOT/src" ]; then
  echo "Missing MIST source: $MIST_ROOT/src" >&2
  exit 1
fi

if [ ! -f "$MIST_ROOT/pretrained_models/mist_contrastive_canopus_pretrain.ckpt" ]; then
  echo "Missing MIST checkpoint" >&2
  exit 1
fi

cp -a "$MIST_ROOT/src" "$ASSET_ROOT/MIST/"

# Patch the packaged MIST copy for Kaggle's current PyTorch Lightning API.
# The local MIST checkout is left untouched.
python scripts/patch_mist_kaggle_compat.py "$ASSET_ROOT/MIST"

mkdir -p "$ASSET_ROOT/MIST/pretrained_models"

cp "$MIST_ROOT/pretrained_models/mist_contrastive_canopus_pretrain.ckpt" \
   "$ASSET_ROOT/MIST/pretrained_models/"

echo
echo "Downloading Kaggle CPython 3.13 RDKit wheel..."

uvx --from pip pip download \
  "rdkit==2026.9.1" \
  --only-binary=:all: \
  --platform manylinux_2_28_x86_64 \
  --python-version 313 \
  --implementation cp \
  --abi cp313 \
  --no-deps \
  --dest "$ASSET_ROOT/wheelhouse"

RDKIT_WHEEL_COUNT="$(find "$ASSET_ROOT/wheelhouse" -maxdepth 1 -type f -name 'rdkit-*-cp313-*.whl' | wc -l)"

if [ "$RDKIT_WHEEL_COUNT" -ne 1 ]; then
  echo "Expected exactly one CPython 3.13 RDKit wheel; found $RDKIT_WHEEL_COUNT" >&2
  exit 1
fi

echo
echo "Packaged asset sizes:"
du -sh "$ASSET_ROOT/project/data/processed/pubchem_formula_index"
du -sh "$ASSET_ROOT/project/data/processed/retrieval_index"
du -sh "$ASSET_ROOT/project/data/processed/spectrum_fingerprint"
du -sh "$ASSET_ROOT/MIST"
du -sh "$ASSET_ROOT/wheelhouse"

echo
echo "Uploading Kaggle dataset version..."

kaggle datasets version \
  -m "Guard-5 + spectrum fingerprint inference assets" \
  -p "$ASSET_ROOT" \
  -t \
  -r zip
