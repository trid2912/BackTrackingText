#!/usr/bin/env python3
import os
import json
import argparse
import numpy as np
import cv2
from collections import defaultdict
from pathlib import Path
from PIL import Image
import torch
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "sam2"))

# ------------------------------------------------------------
# SAM2
# ------------------------------------------------------------
from sam2.build_sam import build_sam2_video_predictor

# ------------------------------------------------------------
# Geometry utils (core polygon)
# ------------------------------------------------------------
from core.core_polygon import (
    extract_white_pixels,
    monotone_chain,
    find_best_core_polygon,
)

# ============================================================
# IO
# ============================================================

def load_json(p):
    with open(p, "r") as f:
        return json.load(f)

def save_json(obj, p):
    with open(p, "w") as f:
        json.dump(obj, f, indent=2)

# ============================================================
# Grouping helpers
# ============================================================

def group_by_tid(anns):
    d = defaultdict(list)
    for a in anns:
        d[a["tid"]].append(a)
    return d

def group_by_clip(anns, images_dict):
    clips = defaultdict(list)
    for a in anns:
        fname = images_dict[a["image_id"]]
        clip = os.path.dirname(fname)
        clips[clip].append(a)
    return clips

# ============================================================
# SAM2 frame index mapping (CRITICAL)
# ============================================================

def build_sam2_frame_mapping(gt_images, clip_folder):
    """
    Map image_id -> SAM2 frame_idx
    Must exactly match SAM2 internal ordering
    """
    items = []
    for im in gt_images:
        if im["file_name"].startswith(clip_folder + "/"):
            items.append((im["id"], os.path.basename(im["file_name"])))

    items.sort(key=lambda x: x[1])  # lexicographic
    return {img_id: idx for idx, (img_id, _) in enumerate(items)}

# ============================================================
# Rasterize bbox
# ============================================================

def rasterize_bbox(bbox, H, W):
    x, y, w, h = bbox
    x1 = int(max(0, x))
    y1 = int(max(0, y))
    x2 = int(min(W - 1, x + w))
    y2 = int(min(H - 1, y + h))

    mask = np.zeros((H, W), dtype=np.uint8)
    mask[y1:y2, x1:x2] = 1
    return mask

# ============================================================
# Main SAM2 repair per clip
# ============================================================

def repair_clip(
    predictor,
    anns,
    gt,
    images_dict,
    image_root,
    clip_folder,
    score_thr,
    num_vertices,
    range_search,
    area_ratio_thr,
):
    out = []

    sam2_map = build_sam2_frame_mapping(gt["images"], clip_folder)
    inv_map = {v: k for k, v in sam2_map.items()}

    tracks = group_by_tid(anns)

    frames_dir = os.path.join(image_root, clip_folder)
    inference_state = predictor.init_state(frames_dir)

    for tid, track in tracks.items():
        # ----------------------------------------------------
        # Sort by true time (SAM2 frame index)
        # ----------------------------------------------------
        track_sorted = sorted(
            track, key=lambda a: sam2_map[a["image_id"]]
        )

        # ----------------------------------------------------
        # Find EARLIEST reliable frame
        # ----------------------------------------------------
        init_ann = None
        for ann in track_sorted:
            if ann["score"] >= score_thr:
                init_ann = ann
                break

        # If no reliable frame → keep original track
        if init_ann is None:
            out.extend(track_sorted)
            continue

        init_frame_idx = sam2_map[init_ann["image_id"]]

        # ----------------------------------------------------
        # Prepare initialization geometry
        # ----------------------------------------------------
        img_path = os.path.join(image_root, images_dict[init_ann["image_id"]])
        frame0 = np.array(Image.open(img_path).convert("RGB"))
        H, W = frame0.shape[:2]

        mask_a = rasterize_bbox(init_ann["pred_bbox"], H, W)
        ys, xs = np.where(mask_a == 1)
        if len(xs) == 0:
            out.extend(track_sorted)
            continue

        centroid = np.array([[xs.mean(), ys.mean()]], dtype=np.float32)

        predictor.reset_state(inference_state)

        _, _, logits0 = predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=init_frame_idx,
            obj_id=tid,
            points=centroid,
            labels=np.array([1], dtype=np.int32),
        )

        mask_b_logits = logits0[0].cpu().numpy()[0]
        pts_b = extract_white_pixels((mask_b_logits > 0).astype(np.uint8))

        if len(pts_b) < 4:
            out.extend(track_sorted)
            continue

        hull_b = monotone_chain(pts_b)
        vertices_b = find_best_core_polygon(
            hull_b, num_vertices, range_search
        )

        # ----------------------------------------------------
        # Keep original predictions from init frame onward
        # ----------------------------------------------------
        for ann in track_sorted:
            if sam2_map[ann["image_id"]] >= init_frame_idx:
                out.append(ann)

        # ----------------------------------------------------
        # Backward SAM2 propagation
        # ----------------------------------------------------
        prev_area = len(pts_b)  # carrier mask area in the init frame
        for frame_idx, _, out_logits in predictor.propagate_in_video(
            inference_state, reverse=True
        ):
            frame_idx = int(frame_idx)
            if frame_idx >= init_frame_idx:
                continue

            mask_b_p = (out_logits[0].cpu().numpy()[0] > 0).astype(np.uint8)
            pts_b_p = extract_white_pixels(mask_b_p)
            if len(pts_b_p) < 4:
                continue

            # Reject abrupt carrier mask-area changes (SAM-2 drift / occlusion).
            area_p = len(pts_b_p)
            if area_ratio_thr and prev_area:
                ratio = max(area_p / prev_area, prev_area / max(area_p, 1))
                if ratio > area_ratio_thr:
                    continue
            prev_area = area_p

            hull_b_p = monotone_chain(pts_b_p)
            try:
                vertices_b_p = find_best_core_polygon(
                    hull_b_p, num_vertices, range_search
                )
            except Exception:
                continue

            Hm, _ = cv2.findHomography(vertices_b, vertices_b_p, cv2.RANSAC)
            if Hm is None:
                continue

            mask_a_p = cv2.warpPerspective(
                mask_a, Hm, (W, H), flags=cv2.INTER_NEAREST
            )

            pts_a_p = extract_white_pixels(mask_a_p)
            if len(pts_a_p) < 4:
                continue

            hull_a_p = monotone_chain(pts_a_p)
            try:
                quad = find_best_core_polygon(
                    hull_a_p, 4, range_search
                )
            except Exception:
                continue

            quad = quad.flatten().tolist()
            xs = quad[0::2]
            ys = quad[1::2]

            out.append(
                {
                    "tid": tid,
                    "image_id": inv_map[frame_idx],
                    "pred_bbox": [
                        min(xs),
                        min(ys),
                        max(xs) - min(xs),
                        max(ys) - min(ys),
                    ],
                    "score": init_ann["score"],  # propagated score
                }
            )

    return out

# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_a", required=True)
    parser.add_argument("--gt_json", required=True)
    parser.add_argument("--image_root", required=True)
    parser.add_argument("--out_json", required=True)

    parser.add_argument("--score_thr", type=float, default=0.5)
    parser.add_argument("--num_vertices", type=int, default=16)
    parser.add_argument("--range_search", type=float, default=np.pi / 30)
    parser.add_argument("--area_ratio_thr", type=float, default=4.0,
                        help="Reject a back-tracked frame if the carrier mask area changes by more "
                             "than this ratio vs. the previous frame (abrupt change => unreliable; 0 disables).")

    parser.add_argument("--sam2_cfg", default="configs/sam2.1/sam2.1_hiera_l.yaml")
    parser.add_argument("--sam2_ckpt", default="./checkpoints/sam2.1_hiera_large.pt")
    parser.add_argument("--device", default="cuda")

    args = parser.parse_args()

    gt = load_json(args.gt_json)
    images_dict = {im["id"]: im["file_name"] for im in gt["images"]}

    json_a = load_json(args.json_a)
    clips = group_by_clip(json_a, images_dict)

    predictor = build_sam2_video_predictor(
        args.sam2_cfg, args.sam2_ckpt, device=args.device
    )

    merged = []

    with torch.inference_mode():
        for clip, anns in clips.items():
            print(f"Processing clip: {clip}")
            merged.extend(
                repair_clip(
                    predictor,
                    anns,
                    gt,
                    images_dict,
                    args.image_root,
                    clip,
                    args.score_thr,
                    args.num_vertices,
                    args.range_search,
                    args.area_ratio_thr,
                )
            )

    save_json(merged, args.out_json)
    print("Saved:", args.out_json)

if __name__ == "__main__":
    main()
