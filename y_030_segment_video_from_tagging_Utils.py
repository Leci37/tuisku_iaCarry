import os
import json
import numpy as np
os.environ["OPENCV_VIDEOIO_DEBUG"] = "0"  # 🔇 suppress FFmpeg console noise
import cv2
if hasattr(cv2, "utils") and hasattr(cv2.utils, "logging"):
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
import torch
import torchvision
# cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)

from utils_ import parse_label_map


def load_video_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames

def mask_to_polygon(mask):
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    segmentation = []
    for contour in contours:
        if contour.size >= 6:
            flattened = contour.flatten().tolist()
            segmentation.append(flattened)
    return segmentation

def save_qc_overlay(image, overlay_dir, index):
    os.makedirs(overlay_dir, exist_ok=True)
    qc_path = os.path.join(overlay_dir, f"{index:05d}_overlay.png")
    cv2.imwrite(qc_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

def save_video(frames, path, fps=30):
    import os
    import cv2
    import numpy as np

    if not frames:
        print(f"[WARN] No frames to save at: {path}")
        return

    height, width, _ = frames[0].shape

    # ✅ Override .mp4 → .avi to avoid H264
    if path.endswith(".mp4"):
        path = path.replace(".mp4", ".avi")
        print(f"[INFO] Changing output from .mp4 to .avi for compatibility")

    # ✅ Use MJPG or XVID (safe, widely supported)
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    out = cv2.VideoWriter(path, fourcc, fps, (width, height))

    for frame in frames:
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(bgr)

    out.release()
    print(f"[✓] Video saved to: {path}")


def init_coco_structure(label_metadata):
    category_list = sorted([(meta["id"], name) for name, meta in label_metadata.items()], key=lambda x: x[0])
    return {
        "images": [],
        "annotations": [],
        "categories": [
            {
                "id": cid,
                "name": name,
                "supercategory": "object"
            } for cid, name in category_list
        ]
    }


import numpy as np
import cv2
import math

def clean_and_filter_mask(
    mask,
    min_area=300,
    max_components=4,
    min_solidity=0.10,
    max_aspect_ratio=4.0,
    min_circularity=0.15
):
    """
    Cleans and filters a binary object mask:
    - Keeps only the largest connected component
    - Rejects masks based on area, solidity, fragmentation, elongation, and circularity

    Returns:
        cleaned_mask (np.ndarray) or None if rejected,
        discard_reason (str) or None
    """
    # if np.sum(mask) == 0:
    #     print("[DEBUG] Discarded mask: Empty mask (no foreground pixels).")
    #     return None, "empty_mask"

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("[DEBUG] Discarded mask: No contours found.")
        return None, "no_contours"

    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    if area < min_area:
        print(f"[DEBUG] Discarded mask: Area too small ({area:.1f} < {min_area})")
        return None, "too_small"

    # Solidity = area / convex hull area
    hull = cv2.convexHull(largest)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area > 0 else 0
    if solidity < min_solidity:
        print(f"[DEBUG] Discarded mask: Low solidity ({solidity:.3f} < {min_solidity})")
        return None, "low_solidity"

    # Aspect Ratio
    x, y, w, h = cv2.boundingRect(largest)
    if min(w, h) == 0:
        print("[DEBUG] Discarded mask: Zero dimension in bounding box.")
        return None, "zero_bounding_dim"
    aspect_ratio = max(w, h) / min(w, h)
    if aspect_ratio > max_aspect_ratio:
        print(f"[DEBUG] Discarded mask: Too elongated (aspect ratio {aspect_ratio:.2f} > {max_aspect_ratio})")
        return None, "too_elongated"

    # Circularity = 4π * area / perimeter²
    perimeter = cv2.arcLength(largest, True)
    if perimeter == 0:
        print("[DEBUG] Discarded mask: Zero perimeter.")
        return None, "zero_perimeter"
    circularity = 4 * math.pi * area / (perimeter ** 2)
    if circularity < min_circularity:
        print(f"[DEBUG] Discarded mask: Low circularity ({circularity:.3f} < {min_circularity})")
        return None, "low_circularity"

    if len(contours) > max_components:
        print(f"[DEBUG] Discarded mask: Too many components ({len(contours)} > {max_components})")
        return None, "too_many_components"

    cleaned = np.zeros_like(mask, dtype=np.uint8)
    cv2.drawContours(cleaned, [largest], -1, 1, -1)
    return cleaned, None


def write_global_yolo_classes(label_map_path, output_dir, filename="classes.txt"):

    label_map = parse_label_map(label_map_path)
    sorted_label_names = [name for _, name in sorted(label_map.items())]  # ✅ Fix here

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)

    with open(output_path, "w", encoding="utf-8") as f:
        for name in sorted_label_names:
            f.write(f"{name}\n")

    print(f"[INFO] Wrote global YOLO classes.txt with {len(sorted_label_names)} classes → {output_path}")
    return output_path
