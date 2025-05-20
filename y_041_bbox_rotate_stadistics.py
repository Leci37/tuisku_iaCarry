import os
import cv2
from collections import defaultdict
from y_bbox_utils import (
    load_class_names,
    rotate_yolo_bbox_90cw,
    draw_yolo_bbox_on_image,
    ensure_dirs,
    print_summary
)

# --- Paths ---
ROOT_DIR = "gui_04_bbox_clean"
OUTPUT_ROOT = "gui_041_bbox_clean"
IMAGES_DIR = os.path.join(ROOT_DIR, "frames")
LABELS_DIR = os.path.join(ROOT_DIR, "yolo_labels")
CLASSES_FILE = os.path.join(ROOT_DIR, "classes.txt")
OUTPUT_IMG_DIR = os.path.join(OUTPUT_ROOT, "frames")
OUTPUT_LABEL_DIR = os.path.join(OUTPUT_ROOT, "yolo_labels")
CHECK_DIR = os.path.join(OUTPUT_ROOT, "check")

# --- Config ---
TARGET_SIZE = (720, 1280)  # height x width
class_names = load_class_names(CLASSES_FILE)
ensure_dirs([OUTPUT_IMG_DIR, OUTPUT_LABEL_DIR, CHECK_DIR])

# --- Stats ---
label_counts = defaultdict(int)
only_label_occurrence = defaultdict(int)
label_occurrence = defaultdict(int)
image_class_hist = defaultdict(int)
total_images = 0
total_labels = 0

print("📊 Processing...")

for fname in sorted(os.listdir(IMAGES_DIR)):
    if not fname.endswith(".png"):
        continue

    base = os.path.splitext(fname)[0]
    img_path = os.path.join(IMAGES_DIR, fname)
    label_path = os.path.join(LABELS_DIR, base + ".txt")

    img = cv2.imread(img_path)
    if img is None:
        print(f"[SKIP] Cannot read {fname}")
        continue

    h_orig, w_orig = img.shape[:2]
    rotated = h_orig > w_orig
    if rotated:
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        h_orig, w_orig = w_orig, h_orig
        print(f"[ROTATE] {fname} → {w_orig}x{h_orig}")

    scale_x = TARGET_SIZE[1] / w_orig
    scale_y = TARGET_SIZE[0] / h_orig
    img = cv2.resize(img, (TARGET_SIZE[1], TARGET_SIZE[0]))

    label_lines = []
    detected_classes = set()

    if os.path.exists(label_path):
        with open(label_path, "r") as f:
            for line in f:
                cls_id, x, y, w, h = map(float, line.strip().split())
                cls_id = int(cls_id)
                if rotated:
                    x, y, w, h = rotate_yolo_bbox_90cw(x, y, w, h)

                # Draw and update stats
                img = draw_yolo_bbox_on_image(img, x, y, w, h, class_names[cls_id])
                label_lines.append(f"{cls_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
                label_counts[cls_id] += 1
                label_occurrence[cls_id] += 1
                total_labels += 1
                detected_classes.add(cls_id)

    if len(detected_classes) == 1:
        only_label_occurrence[list(detected_classes)[0]] += 1
    image_class_hist[len(detected_classes)] += 1
    total_images += 1

    # Save outputs
    cv2.imwrite(os.path.join(CHECK_DIR, fname), img)
    cv2.imwrite(os.path.join(OUTPUT_IMG_DIR, fname), img)
    with open(os.path.join(OUTPUT_LABEL_DIR, base + ".txt"), "w") as f:
        f.write("\n".join(label_lines) + "\n")

    print(f"[OK] {fname} → {len(detected_classes)} classes: {sorted(detected_classes)}")

# --- Final Report ---
print_summary(
    class_names, label_counts, total_labels,
    image_class_hist, total_images,
    only_label_occurrence, label_occurrence
)
