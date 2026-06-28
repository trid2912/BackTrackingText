#!/usr/bin/env python3
import json
import argparse
from collections import defaultdict
from pycocotools import mask as maskUtils
import numpy as np


# ---------------------------------------------------
# IoU
# ---------------------------------------------------
def mask_iou(a, b):
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return inter / union if union > 0 else 0.0


# ---------------------------------------------------
# Load tracking grouped by obj_id
# ---------------------------------------------------
def group_tracking_by_id(track_json):

    with open(track_json) as f:
        track = json.load(f)

    id_frames = defaultdict(list)

    for frame_name, objs in track.items():
        for obj in objs:
            id_frames[obj["obj_id"]].append((frame_name, obj))

    return track, id_frames


# ---------------------------------------------------
# Load detections per frame
# ---------------------------------------------------
def load_detections(det_json):

    with open(det_json) as f:
        detections = json.load(f)

    det_per_frame = {}

    for ann in detections:
        det_per_frame[ann["image"]] = ann

    return det_per_frame


# ---------------------------------------------------
# Find detection score that initialized track
# ---------------------------------------------------
def get_birth_detection_score(
    obj_entries,
    det_per_frame
):
    """
    obj_entries = [(frame_name, obj_dict)]
    """

    # birth = latest frame (backward tracking)
    birth_frame, birth_obj = max(
        obj_entries,
        key=lambda x: x[0]
    )

    if birth_frame not in det_per_frame:
        return None

    track_mask = maskUtils.decode(birth_obj["mask"]).astype(bool)

    ann = det_per_frame[birth_frame]

    best_score = None
    best_iou = 0.0

    for rle, score in zip(ann["masks"], ann["score"]):

        det_mask = maskUtils.decode(rle).astype(bool)
        iou = mask_iou(track_mask, det_mask)

        if iou > best_iou:
            best_iou = iou
            best_score = score

    return best_score


# ---------------------------------------------------
# Main filter
# ---------------------------------------------------
def filter_tracklets(
    track_json,
    det_json,
    bin_low,
    bin_high,
    iou_min=0.3
):

    track, id_frames = group_tracking_by_id(track_json)
    det_per_frame = load_detections(det_json)

    keep_ids = set()

    for obj_id, entries in id_frames.items():

        score = get_birth_detection_score(entries, det_per_frame)

        if score is None:
            continue

        if bin_low <= score < bin_high:
            keep_ids.add(obj_id)

    print("Keep IDs:", len(keep_ids))

    # ---------------------------------------------------
    # Filter tracking output
    # ---------------------------------------------------
    filtered = {}

    for frame, objs in track.items():
        filtered[frame] = [
            o for o in objs if o["obj_id"] in keep_ids
        ]

    return filtered


# ---------------------------------------------------
# CLI
# ---------------------------------------------------
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--track_json", required=True)
    parser.add_argument("--det_json", required=True)
    parser.add_argument("--out_json", required=True)

    parser.add_argument("--bin_low", type=float, required=True)
    parser.add_argument("--bin_high", type=float, required=True)

    args = parser.parse_args()

    filtered = filter_tracklets(
        args.track_json,
        args.det_json,
        args.bin_low,
        args.bin_high,
    )

    with open(args.out_json, "w") as f:
        json.dump(filtered, f, indent=2)

    print("Saved →", args.out_json)
