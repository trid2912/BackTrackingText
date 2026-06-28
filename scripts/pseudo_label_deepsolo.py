import os
import json
import torch
import numpy as np
import supervision as sv
from pathlib import Path
from tqdm import tqdm
from pycocotools import mask as maskUtils
import argparse

from sam2.build_sam import build_sam2_video_predictor


# =========================
# Helpers
# =========================
def to_pylist(x):
    if isinstance(x, np.ndarray):
        return x.astype(float).tolist()
    if torch.is_tensor(x):
        return x.detach().cpu().numpy().astype(float).tolist()
    return float(x) if isinstance(x, (np.floating, np.float32, np.float64)) else x


def make_json_serializable(obj):
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.astype(float).tolist()
    elif torch.is_tensor(obj):
        return obj.detach().cpu().numpy().astype(float).tolist()
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    else:
        return obj


# =========================
# Process Clip
# =========================
def process_clip(det_json_path: str, frames_dir: str, output_json: str, video_predictor):
    """Run SAM2 reverse tracking from detection JSON and save tracking_results.json"""

    with open(det_json_path, "r") as f:
        detections = json.load(f)

    if len(detections) == 0:
        return

    inference_state = video_predictor.init_state(video_path=frames_dir)
    tracking_results = {i: {"detections": [], "tracking": []} for i in range(len(detections))}
    obj_id_counter = 0

    for frame_idx, det in enumerate(detections):
        masks = det.get("masks", [])
        if len(masks) == 0:
            continue

        video_predictor.reset_state(inference_state)

        for m in masks:
            mask = maskUtils.decode(m).astype(np.uint8)
            _, out_obj_ids, out_mask_logits = video_predictor.add_new_mask(
                inference_state=inference_state,
                frame_idx=frame_idx,
                obj_id=obj_id_counter,
                mask=torch.from_numpy(mask).to(dtype=torch.float32, device="cuda"),
            )
            tracking_results[frame_idx]["detections"].append({
                "mask": m,
                "obj_id": obj_id_counter
            })
            obj_id_counter += 1

        # reverse tracking only
        for out_f_idx, out_obj_ids, out_mask_logits in video_predictor.propagate_in_video(
            inference_state, reverse=True
        ):
            for obj_id, mask in zip(out_obj_ids, out_mask_logits):
                bbox = sv.mask_to_xyxy((mask > 0).cpu())
                tracking_results[out_f_idx]["tracking"].append({
                    "obj_id": int(obj_id),
                    "bbox": to_pylist(bbox)
                })

    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(make_json_serializable(tracking_results), f, indent=2)

    print(f"[Saved] {output_json}")


# =========================
# Main
# =========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames-root", required=True, help="Root folder of frames (video/clip/images)")
    parser.add_argument("--dets-root", required=True, help="Root folder of detection JSONs")
    parser.add_argument("--output-root", required=True, help="Output folder for tracking results")
    parser.add_argument("--sam2-checkpoint", default="./checkpoints/sam2.1_hiera_large.pt")
    parser.add_argument("--sam2-cfg", default="configs/sam2.1/sam2.1_hiera_l.yaml")
    args = parser.parse_args()

    os.makedirs(args.output_root, exist_ok=True)
    video_predictor = build_sam2_video_predictor(args.sam2_cfg, args.sam2_checkpoint)

    for vid in sorted(os.listdir(args.dets_root)):
        vid_path = os.path.join(args.dets_root, vid)
        if not os.path.isdir(vid_path):
            continue

        for det_json in sorted(os.listdir(vid_path)):
            if not det_json.endswith(".json"):
                continue

            det_json_path = os.path.join(vid_path, det_json)
            clip_name = Path(det_json).stem
            frames_dir = os.path.join(args.frames_root, vid, clip_name)

            if not os.path.isdir(frames_dir):
                print(f"⚠️ Skipping {frames_dir}, not found")
                continue

            output_json = os.path.join(args.output_root, vid, f"{clip_name}.json")
            process_clip(det_json_path, frames_dir, output_json, video_predictor)
