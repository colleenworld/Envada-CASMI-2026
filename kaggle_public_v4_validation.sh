#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${CASMI26_PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
KERNEL_DIR="$PROJECT_ROOT/kaggle/public_v4_validation"
META="$KERNEL_DIR/kernel-metadata.json"
KAGGLE_USER="${KAGGLE_USERNAME:-ColleenDavis}"
VALIDATION_DATASET="${CASMI_V4_VALIDATION_DATASET:-$KAGGLE_USER/casmi26-v4-validation-split}"
POPULARITY_DATASET="${CASMI26_POPULARITY_DATASET:-}"

if [[ -z "$POPULARITY_DATASET" ]]; then
  POPULARITY_DATASET="$(
    kaggle datasets list -s "CASMI26 PubChem Popularity Prior" --csv 2>/dev/null       | python -c 'import csv,sys
rows=list(csv.DictReader(sys.stdin))
for row in rows:
    ref=(row.get("ref") or row.get("Ref") or "").strip()
    title=(row.get("title") or row.get("Title") or "").strip().lower()
    if ref and ("pubchem" in title and "popularity" in title and "prior" in title):
        print(ref)
        break'
  )"
fi

if [[ -z "$POPULARITY_DATASET" ]]; then
  echo "Could not resolve popularity-prior dataset." >&2
  exit 1
fi

python - "$META" "$KAGGLE_USER" "$VALIDATION_DATASET" "$POPULARITY_DATASET" <<'PY'
import json, sys
from pathlib import Path

path=Path(sys.argv[1])
user=sys.argv[2]
validation=sys.argv[3]
popularity=sys.argv[4]

meta=json.loads(path.read_text())
meta["id"]=f"{user}/casmi26-public-v4-validation"
sources=list(meta.get("dataset_sources", []))
for ref in (validation, popularity):
    if ref not in sources:
        sources.append(ref)
meta["dataset_sources"]=sources
path.write_text(json.dumps(meta, indent=2)+"\n")

print("Kernel ID:", meta["id"])
print("Validation dataset:", validation)
print("Popularity prior:", popularity)
PY

kaggle config view >/dev/null
kaggle kernels push -p "$KERNEL_DIR"

echo
echo "Validation kernel pushed."
echo "Check with:"
echo "  kaggle kernels status $KAGGLE_USER/casmi26-public-v4-validation"
echo
echo "When complete, download with:"
echo "  mkdir -p data/processed/v4_validation/kaggle_run"
echo "  kaggle kernels output $KAGGLE_USER/casmi26-public-v4-validation -p data/processed/v4_validation/kaggle_run"
