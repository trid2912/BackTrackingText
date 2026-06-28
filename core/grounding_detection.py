#!/usr/bin/env python3
import os
import cv2
import json
import torch
import numpy as np
import tqdm
import argparse
from pathlib import Path
from torchvision.ops import box_convert
from grounding_dino.groundingdino.util.inference import load_model, load_image, predict
from pycocotools import mask as maskUtils

# =====================
# Config
# =====================
GROUNDING_DINO_CONFIG = "grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CHECKPOINT = "gdino_checkpoints/groundingdino_swint_ogc.pth"
BOX_THRESHOLD = 0.35
TEXT_THRESHOLD = 0.25
TEXT_PROMPT = "word."   # <-- change this prompt
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def box_to_mask(box, height, width):
    """Convert xyxy box to binary mask"""
    x1, y1, x2, y2 = map(int, box)
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.rectangle(mask, (x1, y1), (x2, y2), 1, thickness=-1)
    return mask


def process_clip(clip_path, model, output_json_path):
    # only keep image files
    valid_ext = {".jpg", ".jpeg", ".png", ".bmp"}
    frames = sorted([
        f for f in os.listdir(clip_path)
        if os.path.splitext(f)[1].lower() in valid_ext
    ])

    if not frames:
        print(f"[Skip] No valid images in {clip_path}")
        return

    results = []

    for frame_name in tqdm.tqdm(frames, desc=f"Processing {clip_path}", leave=False):
        frame_path = os.path.join(clip_path, frame_name)
        image_source, image = load_image(frame_path)
        h, w, _ = image_source.shape

        boxes, confidences, labels = predict(
            model=model,
            image=image,
            caption=TEXT_PROMPT,
            box_threshold=BOX_THRESHOLD,
            text_threshold=TEXT_THRESHOLD,
        )

        frame_entry = {"image": frame_name, "masks": []}

        if boxes is not None and len(boxes) > 0:
            # scale from normalized cxcywh → absolute xyxy
            boxes = boxes * torch.tensor([w, h, w, h], device=boxes.device)
            input_boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").cpu().numpy()

            for box in input_boxes:
                mask = box_to_mask(box, h, w)

                # Save as RLE
                rle = maskUtils.encode(np.asfortranarray(mask))
                rle["counts"] = rle["counts"].decode("utf-8")
                frame_entry["masks"].append(rle)

        results.append(frame_entry)

    with open(output_json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[Saved] {output_json_path}")


# --------------------
# Main
# --------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="input root dir (videos/clips/frames)")
    parser.add_argument("--output", required=True, help="output root dir for JSONs")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # init model once
    model = load_model(
        model_config_path=GROUNDING_DINO_CONFIG,
        model_checkpoint_path=GROUNDING_DINO_CHECKPOINT,
        device=DEVICE
    )

    # walk over video → clip
    for video in sorted(os.listdir(args.input)):
        video_in = os.path.join(args.input, video)
        if not os.path.isdir(video_in):
            continue

        video_out = os.path.join(args.output, video)
        os.makedirs(video_out, exist_ok=True)

        for clip in sorted(os.listdir(video_in)):
            clip_in = os.path.join(video_in, clip)
            if not os.path.isdir(clip_in):
                continue

            clip_out_json = os.path.join(video_out, f"{clip}.json")
            process_clip(clip_in, model, clip_out_json)
