import os
import cv2
import numpy as np
import albumentations as A
from itertools import combinations

def build_aug_combinations(aug_base):
    combos = {"original": []}
    for name in aug_base:
        combos[name] = [aug_base[name]]
    for pair in combinations(aug_base.keys(), 2):
        key = "+".join(pair)
        combos[key] = [aug_base[pair[0]], aug_base[pair[1]]]
    for triple in combinations(aug_base.keys(), 3):
        key = "+".join(triple)
        combos[key] = [aug_base[triple[0]], aug_base[triple[1]], aug_base[triple[2]]]
    return combos

def load_image_and_boxes(image_path, label_dir):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    base = os.path.splitext(os.path.basename(image_path))[0]
    label_path = os.path.join(label_dir, base + ".txt")

    boxes, class_ids = [], []
    if os.path.exists(label_path):
        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                cls, x, y, bw, bh = map(float, parts)
                x_min = (x - bw / 2) * w
                y_min = (y - bh / 2) * h
                x_max = (x + bw / 2) * w
                y_max = (y + bh / 2) * h
                boxes.append([x_min, y_min, x_max, y_max])
                class_ids.append(int(cls))
    return img, boxes, class_ids

def draw_bboxes(image, boxes, class_ids, class_names):
    img = image.copy()
    for box, cls_id in zip(boxes, class_ids):
        cls_id = int(cls_id)
        x1, y1, x2, y2 = map(int, box)
        color = (0, 255, 0)
        label = class_names[cls_id] if cls_id < len(class_names) else str(cls_id)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return img

def filter_occluded_boxes(image, boxes, class_ids, threshold=1.0):
    filtered_boxes = []
    filtered_ids = []
    h, w = image.shape[:2]

    for box, cls_id in zip(boxes, class_ids):
        x1, y1, x2, y2 = map(int, box)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        crop = image[y1:y2, x1:x2]

        if crop.size == 0:
            continue

        black_ratio = np.mean(np.all(crop == 0, axis=-1))
        if black_ratio < threshold:
            filtered_boxes.append(box)
            filtered_ids.append(cls_id)
    return filtered_boxes, filtered_ids
