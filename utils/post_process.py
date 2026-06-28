import json
from pycocotools import mask as maskUtils
from copy import deepcopy

def compute_iou(rle1, rle2):
    """Compute IoU between two RLE masks."""
    return maskUtils.iou([rle1], [rle2], [0])[0][0]

def filter_json(input_json, iou_thresh=0.4):
    data = json.load(open(input_json, "r"))
    out = deepcopy(data)

    for frame_id, anns in data.items():
        kept = []
        kept_rles = []

        for det in anns:
            mask_rle = {
                "size": det["mask"]["size"],
                "counts": det["mask"]["counts"].encode("utf-8"),
            }

            # check IoU against already kept detections
            overlap = False
            for krle in kept_rles:
                if compute_iou(mask_rle, krle) > iou_thresh:
                    overlap = True
                    break

            if not overlap:
                kept.append(det)
                kept_rles.append(mask_rle)

        out[frame_id] = kept

    return out


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_json", help="Path to input JSON")
    parser.add_argument("--output_json", help="Path to save filtered JSON")
    parser.add_argument("--iou", type=float, default=0.4, help="IoU threshold")
    args = parser.parse_args()

    filtered = filter_json(args.input_json, args.iou)
    with open(args.output_json, "w") as f:
        json.dump(filtered, f, indent=2)
