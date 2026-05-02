#!/bin/bash
# Usage: bash upload.sh YOUR_KAGGLE_USERNAME [--skip-data]
# --skip-data  skip re-uploading the processed dataset (saves time when only code/ckpt changed)
set -e
USERNAME=${1:-YOUR_KAGGLE_USERNAME}
SKIP_DATA=false
for arg in "$@"; do [ "$arg" = "--skip-data" ] && SKIP_DATA=true; done
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ML_DIR="$(dirname "$SCRIPT_DIR")"
# Stage in a path with NO spaces to avoid Kaggle CLI bug on Windows
STAGING="$HOME/kaggle_staging"

command -v kaggle &>/dev/null || { echo "Run: pip install kaggle"; exit 1; }
[ -f ~/.kaggle/kaggle.json ] || { echo "Put kaggle.json in ~/.kaggle/  (kaggle.com -> Settings -> API -> Create Token)"; exit 1; }

rm -rf "$STAGING"
mkdir -p "$STAGING/data" "$STAGING/checkpoints" "$STAGING/code"

# Dataset 1: processed data
cp -r "$ML_DIR/data/processed" "$STAGING/data/"
cat > "$STAGING/data/dataset-metadata.json" <<EOF
{"title":"Arabic Handwriting Processed Data","id":"$USERNAME/arabic-handwriting-data","licenses":[{"name":"CC0-1.0"}]}
EOF

# Dataset 2: latest checkpoint — wrap in subdir so --dir-mode zip zips it (avoids Windows CLI bug)
MAX_STEP=-1; LATEST=""
for f in "$ML_DIR/outputs/arabic_v1/checkpoints/model-"*; do
    STEP="${f##*model-}"
    [ "$STEP" -gt "$MAX_STEP" ] 2>/dev/null && { MAX_STEP=$STEP; LATEST=$f; }
done
[ -z "$LATEST" ] && { echo "No checkpoints found"; exit 1; }
echo "Uploading checkpoint: model-$MAX_STEP"
mkdir -p "$STAGING/checkpoints/files"
cp "$LATEST" "$STAGING/checkpoints/files/"
cp "$ML_DIR/outputs/arabic_v1/config/config.yaml" "$STAGING/checkpoints/files/"
cat > "$STAGING/checkpoints/dataset-metadata.json" <<EOF
{"title":"Arabic Handwriting Checkpoints","id":"$USERNAME/arabic-handwriting-checkpoints","licenses":[{"name":"CC0-1.0"}]}
EOF

# Dataset 3: source code + config — wrap in subdir for same reason
mkdir -p "$STAGING/code/files"
cp -r "$ML_DIR/src" "$STAGING/code/files/"
cp -r "$ML_DIR/config" "$STAGING/code/files/"
cat > "$STAGING/code/dataset-metadata.json" <<EOF
{"title":"Arabic Handwriting Code","id":"$USERNAME/arabic-handwriting-code","licenses":[{"name":"CC0-1.0"}]}
EOF

# Upload (create new or version existing)
for ds in data checkpoints code; do
    [ "$ds" = "data" ] && [ "$SKIP_DATA" = "true" ] && echo "--- Skipping data (--skip-data) ---" && continue
    echo "--- Uploading $ds ---"
    if kaggle datasets list --mine 2>/dev/null | grep -q "arabic-handwriting-$ds"; then
        kaggle datasets version -p "$STAGING/$ds" -m "update $(date +%Y-%m-%d)" --dir-mode zip
    else
        kaggle datasets create -p "$STAGING/$ds" --dir-mode zip
    fi
done

echo ""
echo "All datasets uploaded. Now:"
echo "  1. Go to kaggle.com/code -> New Notebook"
echo "  2. Add all 3 datasets as input (right panel -> Add data)"
echo "  3. Settings -> Accelerator -> GPU T4 x2"
echo "  4. Paste the contents of kaggle/notebook.ipynb and run all cells"
