import json
import os
from pathlib import Path
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import argparse

# Root that absolute ground-truth paths are anchored to (override via DATA_ROOT env var).
DATA_ROOT = os.environ.get("DATA_ROOT", "data")

def points_to_xyxy(points):
    """Convert quadrilateral points [[x1,y1], [x2,y2], [x3,y3], [x4,y4]] to xyxy format [x_min, y_min, x_max, y_max]."""
    x_coords = [p[0] for p in points]
    y_coords = [p[1] for p in points]
    return [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]

def load_ground_truth(anno_file):
    """Load ground truth from annotation.json and format as COCO."""
    with open(anno_file, 'r') as f:
        gt_data = json.load(f)

    coco_gt = {
        "info": {
            "description": "Scene Text Dataset",
            "version": "1.0",
            "year": 2025,
            "contributor": "anonymous",
            "date_created": "2025/01/01"
        },
        "images": [],
        "annotations": [],
        "categories": [{"id": 1, "name": "word"}]
    }

    # Map frame_id to image info
    frame_map = {frame['frame_id']: frame for frame in gt_data['frame']}
    
    # Add images
    for frame_id, frame in frame_map.items():
        coco_gt['images'].append({
            "id": frame_id,
            "file_name": frame['frame_jpg'],
            "width": frame['width'],
            "height": frame['height']
        })

    # Add annotations
    ann_id = 1
    for ann in gt_data['annotations']:
        frame_id = ann['frame_id']
        if frame_id not in frame_map:
            print(f"Warning: Annotation with frame_id {frame_id} has no matching frame.")
            continue
        bbox = points_to_xyxy(ann['point'])
        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])  # Approximate area
        coco_gt['annotations'].append({
            "id": ann_id,
            "image_id": frame_id,
            "category_id": 1,  # Assuming "word" as category
            "bbox": bbox,  # [x_min, y_min, x_max, y_max]
            "area": area,
            "segmentation": ann['segmentation'],  # RLE format
            "iscrowd": 0
        })
        ann_id += 1

    # Save to temporary JSON for COCO
    temp_gt_file = "temp_gt.json"
    with open(temp_gt_file, 'w') as f:
        json.dump(coco_gt, f, indent=4)
    
    return COCO(temp_gt_file), frame_map

def load_predictions(pred_folder, frame_map):
    """Load prediction JSONs and format as COCO detections."""
    coco_dt = []
    pred_files = list(Path(pred_folder).glob("*.json"))

    if not pred_files:
        raise FileNotFoundError(f"No JSON files found in {pred_folder}")

    for pred_file in pred_files:
        with open(pred_file, 'r') as f:
            pred_data = json.load(f)
        
        # Match image path to frame_id
        img_path = pred_data['image_path']
        # Convert prediction path (e.g., "../dataset/scene_text/frame/clip_10/clip_10_0000.jpg")
        # to match ground truth (e.g., "<DATA_ROOT>/dataset/scene_text/frame/clip_10/clip_10_0000.jpg")
        if img_path.startswith("../"):
            img_path = os.path.join(DATA_ROOT, img_path[3:])
        frame_id = None
        for fid, frame in frame_map.items():
            if frame['frame_jpg'] == img_path:
                frame_id = fid
                break
        if frame_id is None:
            print(f"Warning: No matching frame for {img_path}")
            continue

        for ann in pred_data['annotations']:
            bbox = ann['bbox']  # Already in xyxy
            area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            coco_dt.append({
                "image_id": frame_id,
                "category_id": 1,  # Assuming "word"
                "bbox": bbox,
                "score": ann['score'],
                "segmentation": ann['segmentation'],
            })

    return coco_dt

def evaluate_ap_ar(gt_file, pred_folder):
    """Evaluate AP and AR using COCOeval."""
    # Load ground truth
    coco_gt, frame_map = load_ground_truth(gt_file)
    
    # Load predictions
    coco_dt = load_predictions(pred_folder, frame_map)
    
    if not coco_dt:
        print("No valid predictions found. Exiting evaluation.")
        return None
    
    # Save predictions to temporary JSON
    temp_dt_file = "temp_dt.json"
    with open(temp_dt_file, 'w') as f:
        json.dump(coco_dt, f, indent=4)
    
    # Load into COCO
    coco_dt = coco_gt.loadRes(temp_dt_file)
    
    # Evaluate bounding boxes
    print("\nEvaluating Bounding Boxes...")
    coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    
    # Store bbox results
    bbox_stats = coco_eval.stats.tolist()  # [AP@0.5:0.95, AP@0.5, AP@0.75, ...]
    
    # Evaluate segmentations
    print("\nEvaluating Segmentations...")
    coco_eval = COCOeval(coco_gt, coco_dt, iouType='segm')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    
    # Store segm results
    segm_stats = coco_eval.stats.tolist()

    # Clean up
    os.remove("temp_gt.json")
    os.remove("temp_dt.json")

    return {
        "bbox": {
            "AP": bbox_stats[0],  # AP@0.5:0.95
            "AP50": bbox_stats[1],  # AP@0.5
            "AP75": bbox_stats[2],  # AP@0.75
            "AR": bbox_stats[8]    # AR@0.5:0.95
        },
        "segm": {
            "AP": segm_stats[0],
            "AP50": segm_stats[1],
            "AP75": segm_stats[2],
            "AR": segm_stats[8]
        }
    }

def main():
    parser = argparse.ArgumentParser(description="Evaluate AP and AR for text detection")
    parser.add_argument('--gt_file', type=str, required=True, help="Path to ground truth annotation.json")
    parser.add_argument('--pred_folder', type=str, required=True, help="Folder containing prediction JSONs")
    args = parser.parse_args()

    results = evaluate_ap_ar(args.gt_file, args.pred_folder)
    
    if results:
        print("\nBounding Box Evaluation:")
        print(f"AP@0.5:0.95: {results['bbox']['AP']:.4f}")
        print(f"AP@0.5: {results['bbox']['AP50']:.4f}")
        print(f"AP@0.75: {results['bbox']['AP75']:.4f}")
        print(f"AR@0.5:0.95: {results['bbox']['AR']:.4f}")
        
        print("\nSegmentation Evaluation:")
        print(f"AP@0.5:0.95: {results['segm']['AP']:.4f}")
        print(f"AP@0.5: {results['segm']['AP50']:.4f}")
        print(f"AP@0.75: {results['segm']['AP75']:.4f}")
        print(f"AR@0.5:0.95: {results['segm']['AR']:.4f}")

if __name__ == "__main__":
    main()