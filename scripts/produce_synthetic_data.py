import os
import cv2
import torch
import numpy as np
import supervision as sv
from torchvision.ops import box_convert
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import json

# import sys
# sys.path.insert(0,"..")

from sam2.build_sam import build_sam2_video_predictor, build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor 
from grounding_dino.groundingdino.util.inference import load_model, load_image, predict
from utils.track_utils import sample_points_from_masks
from utils.video_utils import create_video_from_images


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
# Config
# =========================
DATASET_DIR = os.environ.get("DATASET_DIR", "data/tracking_data")   # root containing 01, 02, ...
OUTPUT_DIR = "output/json_results"                      # mirrored structure
os.makedirs(OUTPUT_DIR, exist_ok=True)

GROUNDING_DINO_CONFIG = "grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CHECKPOINT = "gdino_checkpoints/groundingdino_swint_ogc.pth"
BOX_THRESHOLD = 0.35
TEXT_THRESHOLD = 0.25
TEXT_PROMPT = "word."

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =========================
# Init Models
# =========================
grounding_model = load_model(
    model_config_path=GROUNDING_DINO_CONFIG, 
    model_checkpoint_path=GROUNDING_DINO_CHECKPOINT,
    device=DEVICE
)

sam2_checkpoint = "./checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

video_predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)
sam2_image_model = build_sam2(model_cfg, sam2_checkpoint)
image_predictor = SAM2ImagePredictor(sam2_image_model)
import os


# =========================
# Processing Function
# =========================
def process_clip(frames_dir: str, output_json: str):
    """Process one clip folder and save tracking_results.json"""
    # frame_files = sorted(os.listdir(frames_dir))
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
    frame_files = [
        f for f in sorted(os.listdir(frames_dir))
        if os.path.splitext(f.lower())[1] in valid_exts
    ]

    if len(frame_files) == 0:
        return


    # init state for this video
    inference_state = video_predictor.init_state(video_path=frames_dir)

    tracking_results = {f_idx: {"detections": [], "tracking": []} for f_idx in range(len(frame_files))}
    obj_id_counter = 0

    for frame_idx, frame_name in enumerate(frame_files):
        frame_path = os.path.join(frames_dir, frame_name)
        image_source, image = load_image(frame_path)

        # --- Run GroundingDINO detection ---
        boxes, confidences, labels = predict(
            model=grounding_model,
            image=image,
            caption=TEXT_PROMPT,
            box_threshold=BOX_THRESHOLD,
            text_threshold=TEXT_THRESHOLD,
        )
        if boxes is None:
            continue
        # post process
        h, w, _ = image_source.shape
        boxes = boxes * torch.Tensor([w, h, w, h])
        input_boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
        confidences = confidences.numpy().tolist()

        video_predictor.reset_state(inference_state)

        # save detection bboxes
        det_bboxes = []
        for idx in range(input_boxes.shape[0]):
            x1, y1, x2, y2 = input_boxes[idx]
            det_bboxes.append({
                "bbox": [x1, y1, x2, y2],
                "score": float(confidences[idx]),
                "obj_id": obj_id_counter
            })
            
            _, out_obj_ids, out_mask_logits = video_predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=frame_idx,
                obj_id=obj_id_counter,
                box=input_boxes[idx],
            )
            obj_id_counter += 1

        tracking_results[frame_idx]["detections"] = det_bboxes

        # --- Tracking Forward ---
        try:
            for out_f_idx, out_obj_ids, out_mask_logits in video_predictor.propagate_in_video(inference_state):
                for obj_id, mask in zip(out_obj_ids, out_mask_logits):
                    bbox = sv.mask_to_xyxy((mask > 0).cpu())
                    tracking_results[out_f_idx]["tracking"].append({
                        "obj_id": int(obj_id),
                        "bbox": to_pylist(bbox)
                    })

            # --- Tracking Backward ---
            for out_f_idx, out_obj_ids, out_mask_logits in video_predictor.propagate_in_video(inference_state, reverse=True):
                for obj_id, mask in zip(out_obj_ids, out_mask_logits):
                    bbox = sv.mask_to_xyxy((mask > 0).cpu())
                    tracking_results[out_f_idx]["tracking"].append({
                        "obj_id": int(obj_id),
                        "bbox": to_pylist(bbox)
                    })
        except:
            continue
    # Save json
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(make_json_serializable(tracking_results), f, indent=2)

    print(f"Saved {output_json}")


# =========================
# Main Loop over all videos/clips
# =========================
if __name__ == "__main__":
    video_folders = sorted(os.listdir(DATASET_DIR))
    for vid in video_folders:
        vid_path = os.path.join(DATASET_DIR, vid)
        if not os.path.isdir(vid_path):
            continue

        clip_folders = sorted(os.listdir(vid_path))
        for clip in clip_folders:
            clip_path = os.path.join(vid_path, clip)
            if not os.path.isdir(clip_path):
                continue

            output_json = os.path.join(OUTPUT_DIR, vid, clip, "tracking_results.json")
            process_clip(clip_path, output_json)
