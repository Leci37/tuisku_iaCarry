import os
import cv2
import albumentations as A
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
OUTPUT_ROOT = "gui_042_bbox_clean"
IMAGES_DIR = os.path.join(ROOT_DIR, "frames")
LABELS_DIR = os.path.join(ROOT_DIR, "yolo_labels")
CLASSES_FILE = os.path.join(ROOT_DIR, "classes.txt")
OUTPUT_IMG_DIR = os.path.join(OUTPUT_ROOT, "frames")
OUTPUT_LABEL_DIR = os.path.join(OUTPUT_ROOT, "yolo_labels")
CHECK_DIR = os.path.join(OUTPUT_ROOT, "check")
CHECK_AUG_DIR = os.path.join(OUTPUT_ROOT, "check_aug")

# --- Config ---
TARGET_SIZE = (720, 1280)
CHECK_EVERY_N = 60
class_names = load_class_names(CLASSES_FILE)
ensure_dirs([OUTPUT_IMG_DIR, OUTPUT_LABEL_DIR, CHECK_DIR, CHECK_AUG_DIR])

# Class balance targets
class_needs = {
    0: 128, 1: 68, 2: 99, 3: 91, 4: 0, 5: 147, 6: 12,
    7: 117, 8: 48, 9: 28, 10: 165, 11: 38, 12: 58, 13: 0
}

AUGMENT = A.Compose([
    A.RandomBrightnessContrast(p=1.0),
    A.Affine(translate_percent=0.1, p=1.0),
    A.Affine(scale=(0.85, 1.15), p=1.0)
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

# --- Stats ---
label_counts = defaultdict(int)
only_label_occurrence = defaultdict(int)
label_occurrence = defaultdict(int)
image_class_hist = defaultdict(int)
aug_per_class = defaultdict(int)
total_images = 0
total_labels = 0
augmented_count = 0

print("📊 [START] Class-balancing to ~300 samples per class")

for i, fname in enumerate(sorted(os.listdir(IMAGES_DIR))):
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

    img = cv2.resize(img, (TARGET_SIZE[1], TARGET_SIZE[0]))

    label_lines = []
    bboxes = []
    class_ids = []
    detected_classes = set()

    if os.path.exists(label_path):
        with open(label_path, "r") as f:
            for line in f:
                cls_id, x, y, w, h = map(float, line.strip().split())
                cls_id = int(cls_id)
                if rotated:
                    x, y, w, h = rotate_yolo_bbox_90cw(x, y, w, h)

                x = min(max(x, 0.0), 1.0)
                y = min(max(y, 0.0), 1.0)
                w = min(max(w, 0.0), 1.0)
                h = min(max(h, 0.0), 1.0)

                label_lines.append(f"{cls_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
                label_counts[cls_id] += 1
                label_occurrence[cls_id] += 1
                total_labels += 1
                detected_classes.add(cls_id)

                bboxes.append([x, y, w, h])
                class_ids.append(cls_id)

    if len(detected_classes) == 1:
        only_label_occurrence[list(detected_classes)[0]] += 1
    image_class_hist[len(detected_classes)] += 1
    total_images += 1

    # Save clean image
    cv2.imwrite(os.path.join(OUTPUT_IMG_DIR, fname), img)
    with open(os.path.join(OUTPUT_LABEL_DIR, base + ".txt"), "w") as f:
        f.write("\n".join(label_lines) + "\n")

    # Visual preview for every Nth image
    if i % CHECK_EVERY_N == 0:
        img_preview = img.copy()
        for (x, y, w, h), cls_id in zip(bboxes, class_ids):
            img_preview = draw_yolo_bbox_on_image(img_preview, x, y, w, h, class_names[int(cls_id)])
        cv2.imwrite(os.path.join(CHECK_DIR, fname), img_preview)

    print(f"[OK] {fname} → classes: {sorted(detected_classes)}")

    # --- One augmentation pass if needed ---
    if any(class_needs.get(cid, 0) > 0 for cid in class_ids):
        print(f"[AUGMENT] {fname} → balancing with one augmentation...")
        try:
            transformed = AUGMENT(image=img, bboxes=bboxes, class_labels=class_ids)
            aug_img = transformed["image"]
            aug_boxes = transformed["bboxes"]
            aug_ids = transformed["class_labels"]

            aug_name = f"{base}__aug1"
            aug_img_path = os.path.join(OUTPUT_IMG_DIR, f"{aug_name}.png")
            aug_lbl_path = os.path.join(OUTPUT_LABEL_DIR, f"{aug_name}.txt")
            aug_check_path = os.path.join(CHECK_AUG_DIR, f"{aug_name}.png")

            label_lines_aug = []
            qc_img = aug_img.copy()
            updated = False

            for (x, y, w, h), cls_id in zip(aug_boxes, aug_ids):
                cls_id = int(cls_id)

                x = min(max(x, 0.0), 1.0)
                y = min(max(y, 0.0), 1.0)
                w = min(max(w, 0.0), 1.0)
                h = min(max(h, 0.0), 1.0)

                # ✅ always save all boxes
                label_lines_aug.append(f"{cls_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
                qc_img = draw_yolo_bbox_on_image(qc_img, x, y, w, h, class_names[cls_id])

                # ✅ only count if needed for balance
                if class_needs.get(cls_id, 0) > 0:
                    class_needs[cls_id] -= 1
                    aug_per_class[cls_id] += 1

                updated = True

            if updated:
                cv2.imwrite(aug_img_path, aug_img)
                with open(aug_lbl_path, "w") as f:
                    f.write("\n".join(label_lines_aug) + "\n")
                cv2.imwrite(aug_check_path, qc_img)
                print(f"  └── AUG saved: {aug_check_path}")
                augmented_count += 1

        except Exception as e:
            print(f"[ERROR] Augmentation failed on {fname}: {e}")

# --- Final Report ---
print("\n✅ [DONE] Total base images:", total_images)
print("✅ Augmentations generated:", augmented_count)
print("\n📊 Per-class augmentations added:")
for cid in sorted(aug_per_class):
    print(f"  {cid:2d} ({class_names[cid]:<20}): {aug_per_class[cid]}")

print_summary(
    class_names, label_counts, total_labels,
    image_class_hist, total_images,
    only_label_occurrence, label_occurrence
)
