#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VALIDATION_DIR="${CASMI_V4_VALIDATION_DIR:-$PROJECT_ROOT/data/processed/v4_validation}"
WORK_DIR="${CASMI_V4_VALIDATION_WORK:-$VALIDATION_DIR/run}"
SCRIPT="$PROJECT_ROOT/kaggle/public_v4/sota_top1_submission.py"

for f in train.parquet test.parquet sample_submission.csv truth.parquet; do
  if [[ ! -f "$VALIDATION_DIR/$f" ]]; then
    echo "Missing validation artifact: $VALIDATION_DIR/$f" >&2
    echo "Run: uv run python scripts/build_v4_validation_comp.py" >&2
    exit 1
  fi
done

if [[ ! -f "$SCRIPT" ]]; then
  echo "Missing v4 script: $SCRIPT" >&2
  exit 1
fi

mkdir -p "$WORK_DIR"

echo "Validation input: $VALIDATION_DIR"
echo "Validation work:  $WORK_DIR"
echo

# The v4 script discovers its model/pool assets recursively. Before launching,
# verify that at least the core reproduction assets are visible somewhere from
# the project tree or an explicitly supplied asset root.
ASSET_ROOT="${CASMI_V4_ASSET_ROOT:-$PROJECT_ROOT}"

required_patterns=(
  "casmi26-v4b-models"
  "casmi26-v3-models"
  "pool_meta.parquet"
  "pc_smiles.npy"
)

missing=0
for pattern in "${required_patterns[@]}"; do
  if ! find "$ASSET_ROOT" -name "$pattern" -o -path "*/$pattern/*" 2>/dev/null | head -n 1 | grep -q .; then
    echo "Missing local v4 asset: $pattern" >&2
    missing=1
  fi
done

if [[ "$missing" -ne 0 ]]; then
  cat >&2 <<EOF

The validation split is ready, but the full v4 model/pool assets are not
available under:
  $ASSET_ROOT

Set CASMI_V4_ASSET_ROOT to a directory containing the downloaded Kaggle
reproduction assets, or run this validation on Kaggle instead.
EOF
  exit 2
fi

cd "$ASSET_ROOT"

export CASMI_COMP_DIR="$VALIDATION_DIR"
export CASMI_WORK_DIR="$WORK_DIR"
export CASMI_LOCAL=1
export KAGGLE_IS_COMPETITION_RERUN=1

uv run python "$SCRIPT"

echo
echo "Validation run complete."
echo "Manifest:   $WORK_DIR/run_manifest.json"
echo "Submission: $WORK_DIR/submission.csv"
echo "Truth:      $VALIDATION_DIR/truth.parquet"
echo
echo "Evaluate shield thresholds with:"
echo "uv run python scripts/evaluate_v4_shield_thresholds.py \\"
echo "  --run-manifest $WORK_DIR/run_manifest.json \\"
echo "  --truth-file $VALIDATION_DIR/truth.parquet"
