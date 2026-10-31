#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${CASMI26_PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SOURCE_DIR="${CASMI_V4_VALIDATION_DIR:-$PROJECT_ROOT/data/processed/v4_validation}"
STAGE_DIR="${CASMI_V4_VALIDATION_DATASET_STAGE:-$PROJECT_ROOT/.kaggle/v4_validation_dataset}"
KAGGLE_USER="${KAGGLE_USERNAME:-ColleenDavis}"
DATASET_SLUG="${CASMI_V4_VALIDATION_DATASET_SLUG:-casmi26-v4-validation-split}"
DATASET_REF="$KAGGLE_USER/$DATASET_SLUG"

for f in train.parquet test.parquet sample_submission.csv truth.parquet; do
  if [[ ! -f "$SOURCE_DIR/$f" ]]; then
    echo "Missing validation artifact: $SOURCE_DIR/$f" >&2
    exit 1
  fi
done

rm -rf "$STAGE_DIR"
mkdir -p "$STAGE_DIR"

for f in train.parquet test.parquet sample_submission.csv truth.parquet; do
  ln -s "$SOURCE_DIR/$f" "$STAGE_DIR/$f"
done

cat > "$STAGE_DIR/dataset-metadata.json" <<EOF
{
  "title": "CASMI26 V4 Validation Split",
  "id": "$DATASET_REF",
  "licenses": [{"name": "CC0-1.0"}]
}
EOF

echo "Uploading validation dataset: $DATASET_REF"
if kaggle datasets status "$DATASET_REF" >/dev/null 2>&1; then
  kaggle datasets version -p "$STAGE_DIR" -m "Refresh structure-disjoint v4 validation split"
else
  kaggle datasets create -p "$STAGE_DIR" -r zip
fi

echo
echo "Validation dataset ready: $DATASET_REF"
