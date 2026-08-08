import os
import cv2
from collections import defaultdict

def rotate_yolo_bbox_90cw(x, y, w, h):
    """Rotate YOLO bbox 90 degrees clockwise in normalized coords."""
    return 1.0 - y, x, h, w

def draw_yolo_bbox_on_image(img, x, y, w, h, label, color=(0, 255, 0)):
    """Draw a single YOLO-format bbox on the image."""
    H, W = img.shape[:2]
    x1 = int((x - w / 2) * W)
    y1 = int((y - h / 2) * H)
    x2 = int((x + w / 2) * W)
    y2 = int((y + h / 2) * H)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return img

def load_class_names(path):
    """Load class names from file."""
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

def ensure_dirs(paths):
    for p in paths:
        os.makedirs(p, exist_ok=True)

def print_summary(class_names, label_counts, total_labels,
                  image_class_hist, total_images,
                  only_label_occurrence, label_occurrence):

    print("\n📈 Class Histogram (Total Labels):")
    for cls_id, count in sorted(label_counts.items()):
        pct = 100 * count / total_labels if total_labels else 0.0
        print(f"  {cls_id:2d} ({class_names[cls_id]:<20}): {count:4d} ({pct:.1f}%)")

    print("\n🧮 Image-Level Class Presence:")
    for k in sorted(image_class_hist.keys()):
        pct = 100 * image_class_hist[k] / total_images
        print(f"  {k} classes in image: {image_class_hist[k]:3d} images ({pct:.1f}%)")

    print("\n🎯 SOLO-Class Image Breakdown:")
    for cls_id, solo in sorted(only_label_occurrence.items()):
        total = label_occurrence[cls_id]
        pct = 100 * solo / total if total else 0
        print(f"  {cls_id:2d} ({class_names[cls_id]:<20}): {solo} solo / {total} → {pct:.1f}%")

    print(f"\n✅ Saved {total_images} images")
