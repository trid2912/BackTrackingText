# Back-Tracking from Clarity: Self-Learning to See Text from Afar

Official implementation of *"Back-Tracking from Clarity: Self-Learning to See Text from Afar"* (ECCV 2026).

## Overview

A self-supervised framework that enhances scene-text detectors to read text at a
distance. The key idea is to **back-track** confident large-text detections
backward through a video so that small, distant appearances of the same text get
pseudo-labels automatically — without any manual annotation.

This repository contains **our** code. The heavy components it builds on
(SAM-2, DeepSolo, CoTracker3, GroundingDINO) are external
repositories — run `bash setup_external.sh` to clone them.
They are `.gitignore`d, so they are never committed here.

## Repository structure

```
BackTrackingText/
├── core/
│   ├── core_polygon.py        # Core-polygon estimation (our algorithm)
│   └── tracking.py            # Back-tracking tracker: SAM-2 carrier + core-polygon homography
├── scripts/
│   ├── pseudo_label_deepsolo.py  # SAM-2 reverse tracking from a detection JSON -> pseudo-labels
│   ├── produce_synthetic_data.py # Synthetic-augmentation baseline (GroundingDINO + SAM-2)
│   └── evaluate.py               # COCO AP / AR evaluation
├── utils/
│   ├── filter_tracking.py     # Filter tracklets by birth-detection score bins
│   └── post_process.py        # IoU-based de-duplication / label merging
├── run_tracking.sh            # Example SLURM launcher for core/tracking.py
├── setup_external.sh          # Clone the external dependencies
└── README.md
```

## Dataset (SceneText50)

**SceneText50** — 50 outdoor ego-centric videos (~840 clips) from 17 countries,
with quadrilateral text annotations. The annotations are for **evaluation**; the
self-learning pipeline needs no manual labels for training.

Download from Hugging Face: https://huggingface.co/datasets/Tri1/SceneText50

```bash
# option 1 — huggingface-cli
pip install -U "huggingface_hub[cli]"
huggingface-cli download Tri1/SceneText50 --repo-type dataset --local-dir data/tracking_data

# option 2 — git + git-lfs
git lfs install
git clone https://huggingface.co/datasets/Tri1/SceneText50 data/tracking_data
```

Layout:

```
tracking_data/
├── 01/ 02/ ... 50/                 # one folder per video
│   ├── clip_10/                    # a clip = a short frame sequence
│   │   ├── clip_10_0000.jpg
│   │   ├── clip_10_0001.jpg
│   │   └── ...
│   ├── clip_10_0000.txt            # per-frame annotation (same basename as the frame)
│   ├── clip_10_0001.txt
│   └── ...
```

Each annotation line is one text instance:

```
instance_id x1 y1 x2 y2 x3 y3 x4 y4
```

`instance_id` is the track ID (consistent across frames of a clip) and the four
`(x, y)` pairs are the clockwise quadrilateral corners. Point `--image_root` at
this `tracking_data` folder when running the tracker / pipeline.

## Pipeline & data formats

Offline pseudo-label generation, then student training. Inference uses only the
trained detector (no future frames, no tracking).

1. **Large-text detection (external).** Run a reliable detector (DeepSolo in the
   paper; GroundingDINO also supported) per frame -> per-clip detection JSON
   (`masks` / `bbox` / `score`).
2. **Point back-tracking — CoTracker3 (external).** Track each detection's
   bounding points backward; keep results with confidence > 0.8. Save per-
   sequence JSON consumed by `core/tracking.py` via `--json_a`.
3. **Carrier back-tracking — `core/tracking.py` (ours).** SAM-2 + core-polygon +
   RANSAC homography, used as the fallback path. See `run_tracking.sh`.
4. **Merge / filter — `utils/post_process.py`, `utils/filter_tracking.py`.**
5. **Student training (external).** Train GroundingDINO on the pseudo-labels
   (freeze image encoder, train decoder, EMA 0.999).
6. **Evaluation — `scripts/evaluate.py`.** COCO AP/AR. (Tracking metrics —
   LaSOT P@20 / S@50 / AUC — and detection-delay / FAR are reported in the
   paper; those scripts are not included here.)

## Usage

```bash
# clone deps first
bash setup_external.sh
# download sam2/checkpoints/sam2.1_hiera_large.pt per the SAM-2 repo

# edit the /path/to/... variables, then:
bash run_tracking.sh
```

Or directly:

```bash
python core/tracking.py \
    --json_a   path/to/cotracker_or_detection.json \
    --gt_json  path/to/gt.json \
    --image_root path/to/frames \
    --out_json path/to/out.json \
    --score_thr 0.8 --num_vertices 16 --area_ratio_thr 4.0
```

## Acknowledgements

This project builds on these excellent open-source works:

- [SAM 2](https://github.com/facebookresearch/sam2)
- [CoTracker3](https://github.com/facebookresearch/co-tracker)
- [DeepSolo](https://github.com/ViTAE-Transformer/DeepSolo)
- [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO)
