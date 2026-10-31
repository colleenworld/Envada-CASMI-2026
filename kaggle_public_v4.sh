#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${CASMI26_PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
KERNEL_DIR="$PROJECT_ROOT/kaggle/public_v4"
META="$KERNEL_DIR/kernel-metadata.json"

if ! command -v kaggle >/dev/null 2>&1; then
  echo "kaggle CLI is not installed or not on PATH" >&2
  exit 1
fi

if [ ! -f "$META" ]; then
  echo "Missing $META" >&2
  exit 1
fi

KAGGLE_USER="${KAGGLE_USERNAME:-}"

if [ -z "$KAGGLE_USER" ] && [ -f "${HOME}/.kaggle/kaggle.json" ]; then
  KAGGLE_USER="$(
    python - <<'PY'
import json
from pathlib import Path

path = Path.home() / ".kaggle" / "kaggle.json"
print(json.loads(path.read_text())["username"])
PY
  )"
fi

if [ -z "$KAGGLE_USER" ]; then
  echo "Could not determine Kaggle username." >&2
  echo "Set KAGGLE_USERNAME or configure ~/.kaggle/kaggle.json." >&2
  exit 1
fi

POPULARITY_DATASET="${CASMI26_POPULARITY_DATASET:-}"

if [ -z "$POPULARITY_DATASET" ]; then
  POPULARITY_DATASET="$(
    kaggle datasets list -s "CASMI26 PubChem Popularity Prior" --csv 2>/dev/null \
      | python -c 'import csv,sys
rows=list(csv.DictReader(sys.stdin))
for row in rows:
    ref=(row.get("ref") or row.get("Ref") or "").strip()
    title=(row.get("title") or row.get("Title") or "").strip().lower()
    if ref and ("pubchem" in title and "popularity" in title and "prior" in title):
        print(ref)
        break'
  )"
fi

if [ -z "$POPULARITY_DATASET" ]; then
  echo "Could not resolve the CASMI26 PubChem Popularity Prior dataset." >&2
  echo "Find it with: kaggle datasets list -s 'CASMI26 PubChem Popularity Prior'" >&2
  echo "Then rerun with CASMI26_POPULARITY_DATASET=owner/dataset" >&2
  exit 1
fi

python - "$META" "$KAGGLE_USER" "$POPULARITY_DATASET" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
username = sys.argv[2]
popularity_dataset = sys.argv[3]

meta = json.loads(path.read_text())
meta["id"] = f"{username}/casmi26-public-v4-test"

sources = list(meta.get("dataset_sources", []))
if popularity_dataset not in sources:
    sources.append(popularity_dataset)
meta["dataset_sources"] = sources

path.write_text(json.dumps(meta, indent=2) + "\n")

print("Kernel ID:", meta["id"])
print("Competition:", ", ".join(meta.get("competition_sources", [])))
print("Popularity prior:", popularity_dataset)
print("Datasets:")
for source in meta.get("dataset_sources", []):
    print(" -", source)
PY

echo
echo "Validating Kaggle credentials..."
kaggle config view >/dev/null

echo
echo "Pushing public-v4 Kaggle kernel..."
kaggle kernels push -p "$KERNEL_DIR"

echo
echo "Kernel pushed."
echo "Check status with:"
echo "  kaggle kernels status $KAGGLE_USER/casmi26-public-v4-test"
echo
echo "When the version succeeds, open it on Kaggle and use Submit to Competition."
