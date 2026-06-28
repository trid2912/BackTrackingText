#!/bin/bash
#SBATCH --partition=public
#SBATCH --job-name=backtrack_text
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --gres=gpu:1

# Activate the environment that has SAM-2 + dependencies installed.
# source /path/to/your/venv/bin/activate

# Run from the repo root. tracking.py adds ./sam2 (the cloned SAM-2 repo) to
# sys.path, and SAM-2's Hydra configs are resolved from the importable package.
cd /path/to/BackTrackingText

SCRIPT="core/tracking.py"

SAM2_CFG="configs/sam2.1/sam2.1_hiera_l.yaml"
SAM2_CKPT="sam2/checkpoints/sam2.1_hiera_large.pt"
IMAGE_ROOT="/path/to/tracking_data"          # frames root: <video>/<clip>/<frame>.jpg
COTRACKER_ROOT="/path/to/cotracker_json"     # precomputed CoTracker3 results per sequence
GT_ROOT="/path/to/tracking_json"             # ground-truth / template boxes per sequence
OUT_ROOT="/path/to/output/hybrid"

mkdir -p "$OUT_ROOT"

for i in $(seq -w 1 50); do
    COTRACKER_JSON="${COTRACKER_ROOT}/${i}.json"
    GT_JSON="${GT_ROOT}/${i}.json"

    if [ ! -f "$COTRACKER_JSON" ] || [ ! -f "$GT_JSON" ]; then
        echo "=== Skipping sequence $i (missing input files) ==="
        continue
    fi

    echo "=== Processing sequence $i ==="
    python "$SCRIPT" \
        --json_a "$COTRACKER_JSON" \
        --gt_json "$GT_JSON" \
        --image_root "$IMAGE_ROOT" \
        --out_json "${OUT_ROOT}/${i}.json" \
        --score_thr 0.8 \
        --num_vertices 16 \
        --area_ratio_thr 4.0 \
        --sam2_cfg "$SAM2_CFG" \
        --sam2_ckpt "$SAM2_CKPT"
done

echo "=== All sequences done ==="
