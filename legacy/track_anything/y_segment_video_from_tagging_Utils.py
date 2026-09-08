import os
import json
import cv2
import numpy as np

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
    import torch
    import torchvision
    torch_frames = torch.from_numpy(np.asarray(frames))
    torchvision.io.write_video(path, torch_frames, fps=fps, video_codec="libx264")

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
