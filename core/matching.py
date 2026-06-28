#!/usr/bin/env python3
"""
Reverse SAM2 tracking with homography propagation (mask_a → mask_a_prime)
- Works with both '1.jpg' and 'clip_10_12.jpg' naming styles
- Computes H from mask_b → mask_b_prime
- Applies H to mask_a → mask_a_prime
- Keeps only when |det(H)| < 1e4
"""

import os
import sys
import json
import cv2
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
from pycocotools import mask as mask_utils
from core_polygon import extract_white_pixels, monotone_chain, find_best_core_polygon
from sam2.build_sam import build_sam2_video_predictor
import argparse
import re


# =============== Utility functions ===============

def polygon_to_mask(polygon, height, width):
    """Convert polygon to binary mask."""
    mask = np.zeros((height, width), dtype=np.uint8)
    pts = np.array(polygon, np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(mask, [pts], 1)
    return mask


def encode_rle_from_mask(mask):
    """Encode binary mask as COCO RLE."""
    return mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))


def extract_frame_index(name):
    """Extract numeric suffix (e.g. clip_10_12 → 12)."""
    m = re.findall(r"\d+", Path(name).stem)
    return int(m[-1]) if m else -1


def ensure_frame_key(k):
    """Return normalized frame key as string."""
    nums = re.findall(r"\d+", k)
    return str(int(nums[-1])) if nums else k


# =============== Main ===============

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames_dir", required=True)
    parser.add_argument("--json_path", required=True)
    parser.add_argument("--sam2_cfg", required=True)
    parser.add_argument("--sam2_ckpt", required=True)
    parser.add_argument("--out_json", default="out.json")
    parser.add_argument("--out_viz_dir", default=None)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    predictor = build_sam2_video_predictor(args.sam2_cfg, args.sam2_ckpt, device=device)
    inference_state = predictor.init_state(args.frames_dir)

    # Load JSON
    with open(args.json_path, "r") as f:
        input_data = json.load(f)

    # Collect frame files and sort numerically
    frame_files = sorted(
        [f for f in os.listdir(args.frames_dir) if f.lower().endswith((".jpg", ".png"))],
        key=lambda x: extract_frame_index(x)
    )
    frame_filename_map = {extract_frame_index(f): f for f in frame_files}

    # Normalize JSON keys
    frame_ids = sorted([(extract_frame_index(k), k) for k in input_data.keys()], key=lambda x: x[0])

    results = {}

    for idx, (frame_num, frame_key) in enumerate(tqdm(frame_ids, desc="Processing frames")):
        if frame_num not in frame_filename_map:
            continue

        frame_name = frame_filename_map[frame_num]
        frame_path = os.path.join(args.frames_dir, frame_name)
        frame = cv2.imread(frame_path)
        if frame is None:
            continue
        h, w = frame.shape[:2]

        objs = input_data[frame_key]
        for obj in objs:
            if "segmentation" not in obj or not obj["segmentation"]:
                continue

            polygon = obj["segmentation"][0]
            mask_b = polygon_to_mask(polygon, h, w)

            # Try previous frame
            prev_num = frame_num - 1
            if prev_num not in frame_filename_map:
                continue

            prev_frame_name = frame_filename_map[prev_num]
            prev_frame_path = os.path.join(args.frames_dir, prev_frame_name)
            prev_frame = cv2.imread(prev_frame_path)
            if prev_frame is None:
                continue

            # --- Predict mask_b_prime using SAM2 (reverse propagation)
            predictor.reset_state(inference_state)
            predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=frame_num,
                obj_id=1,
                points=[[w // 2, h // 2]],
                labels=[1]
            )
            preds = predictor.propagate_in_video(
                inference_state=inference_state,
                start_frame_idx=frame_num,
                reverse=True,
                max_frame_skip=1
            )

            if prev_num not in preds or 1 not in preds[prev_num]:
                continue

            pred_logits = preds[prev_num][1]["logits"]
            mask_b_prime = (pred_logits > 0).astype(np.uint8)

            # --- Compute homography between mask_b and mask_b_prime
            pts_b = np.array(extract_white_pixels(mask_b))
            pts_b_prime = np.array(extract_white_pixels(mask_b_prime))
            if len(pts_b) < 4 or len(pts_b_prime) < 4:
                continue

            H, _ = cv2.findHomography(pts_b, pts_b_prime, cv2.RANSAC, 5.0)
            if H is None:
                continue

            detH = abs(np.linalg.det(H[:2, :2])) if H.shape == (3, 3) else 1e6
            if detH >= 1e4:
                continue  # skip unstable transform

            # --- Apply H to mask_a (existing mask from this obj) ---
            if "mask_a" not in obj:
                continue  # if no mask_a stored, skip
            mask_a = polygon_to_mask(obj["mask_a"], h, w)
            mask_a_prime = cv2.warpPerspective(mask_a.astype(np.uint8), H, (w, h))

            # --- Save result ---
            rle = encode_rle_from_mask(mask_a_prime)
            frame_out_key = ensure_frame_key(prev_frame_name)
            results.setdefault(frame_out_key, []).append({
                "obj_id": obj.get("ID", None),
                "mask": rle
            })

            # --- Optional visualization ---
            if args.out_viz_dir:
                os.makedirs(args.out_viz_dir, exist_ok=True)
                overlay = prev_frame.copy()
                contours, _ = cv2.findContours(mask_a_prime, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)
                cv2.imwrite(os.path.join(args.out_viz_dir, f"{frame_out_key}_{obj.get('ID','')}.jpg"), overlay)

    # Save JSON output
    with open(args.out_json, "w") as f:
        json.dump(results, f, indent=2)

    print(f"✅ Done. Results saved to: {args.out_json}")


if __name__ == "__main__":
    main()
