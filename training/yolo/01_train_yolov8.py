import os
import shutil
import random
from sklearn.model_selection import train_test_split
from ultralytics import YOLO
import torch

def main():
    print("[INFO] CUDA Available:", torch.cuda.is_available())
    print("[INFO] CUDA Device Name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")

    # CONFIG
    SRC_IMAGE_DIR = "gui_042_bbox_clean/frames"
    SRC_LABEL_DIR = "gui_042_bbox_clean/yolo_labels"
    CLASS_FILE = "gui_03_video_segm_pod/classes.txt"
    OUTPUT_DIR = "yolo_dataset"
    MODEL_OUT_DIR = "gui_05_model"
    VAL_RATIO = 0.2
    SEED = 42
    IMG_SIZE = 1280

    # Load images and match labels
    images = sorted([f for f in os.listdir(SRC_IMAGE_DIR) if f.endswith((".png", ".jpg", ".jpeg"))])
    labels = sorted([f for f in os.listdir(SRC_LABEL_DIR) if f.endswith(".txt")])
    image_basenames = [os.path.splitext(f)[0] for f in images]
    label_basenames = [os.path.splitext(f)[0] for f in labels]
    valid_basenames = list(set(image_basenames) & set(label_basenames))

    # Split train/val
    train_ids, val_ids = train_test_split(valid_basenames, test_size=VAL_RATIO, random_state=SEED)

    # Create folder structure
    for split in ["train", "val"]:
        os.makedirs(f"{OUTPUT_DIR}/{split}/images", exist_ok=True)
        os.makedirs(f"{OUTPUT_DIR}/{split}/labels", exist_ok=True)

    # Copy data
    def copy_data(ids, split):
        for base in ids:
            img_src = os.path.join(SRC_IMAGE_DIR, f"{base}.png")
            if not os.path.exists(img_src):
                img_src = os.path.join(SRC_IMAGE_DIR, f"{base}.jpg")
            label_src = os.path.join(SRC_LABEL_DIR, f"{base}.txt")
            if os.path.exists(img_src) and os.path.exists(label_src):
                shutil.copy(img_src, f"{OUTPUT_DIR}/{split}/images/{base}.png")
                shutil.copy(label_src, f"{OUTPUT_DIR}/{split}/labels/{base}.txt")

    copy_data(train_ids, "train")
    copy_data(val_ids, "val")

    # Load class names
    with open(CLASS_FILE, "r", encoding="utf-8") as f:
        class_names = [line.strip() for line in f if line.strip()]

    # Write YAML
    yaml_path = os.path.join(OUTPUT_DIR, "yolo_data.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(f"path: {os.path.abspath(OUTPUT_DIR)}\n")
        f.write("train: train/images\n")
        f.write("val: val/images\n")
        f.write("names:\n")
        for idx, name in enumerate(class_names):
            f.write(f"  {idx}: {name}\n")

    print(f"[✓] Dataset prepared at: {OUTPUT_DIR}")
    print(f"[✓] YAML config written to: {yaml_path}")

    # Train YOLOv8 with augmentations
    os.makedirs(MODEL_OUT_DIR, exist_ok=True)
    model = YOLO("yolov8n.pt")
    results = model.train(
        data=yaml_path,
        device=0,  # <--- Force use of first CUDA GPU
        epochs=50,
        imgsz=IMG_SIZE,
        batch=16,
        project=MODEL_OUT_DIR,
        name="yolov8_gui042_augmented",
        exist_ok=True,
        augment=True,
        degrees=10,
        translate=0.1,
        scale=0.5,
        shear=2.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.0,
        mosaic=1.0,
        mixup=0.2,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        erasing=0.1,
        copy_paste=0.0
    )

    # Export model for reuse
    best_model_path = os.path.join(MODEL_OUT_DIR, "yolov8_gui042_augmented", "weights", "best.pt")
    export_dir = os.path.join(MODEL_OUT_DIR, "export")
    os.makedirs(export_dir, exist_ok=True)
    model.export(format="torchscript", path=os.path.join(export_dir, "best.torchscript"))
    model.export(format="onnx", path=os.path.join(export_dir, "best.onnx"))
    print(f"[✓] Exported TorchScript and ONNX models to {export_dir}/")

    # Print key training metrics
    metrics = results.metrics
    print("\n[📊 METRICS SUMMARY]")
    print(f"mAP@0.5        : {metrics.map50:.3f} (ideal > 0.80)")
    print(f"mAP@0.5:0.95    : {metrics.map:.3f} (ideal > 0.50)")
    print(f"Precision       : {metrics.precision:.3f} (ideal > 0.85)")
    print(f"Recall          : {metrics.recall:.3f} (ideal > 0.80)")
    print(f"F1 Score        : {(2 * metrics.precision * metrics.recall / (metrics.precision + metrics.recall + 1e-6)):.3f}")
    print(f"Classes         : {len(metrics.names)}")
    print("\n[📈 Per-class mAP]")
    for i, name in metrics.names.items():
        per_class_map = metrics.ap_class[i] if hasattr(metrics, 'ap_class') else 'N/A'
        print(f" - {name:18}: {per_class_map:.3f}")

    # Save test predictions for multiple samples
    test_output_dir = os.path.join(MODEL_OUT_DIR, "test_images")
    os.makedirs(test_output_dir, exist_ok=True)
    test_images = [os.path.join(OUTPUT_DIR, "val/images", f"{vid}.png") for vid in val_ids[:5]]

    for img_path in test_images:
        if not os.path.exists(img_path):
            continue
        model.predict(
            source=img_path,
            save=True,
            save_txt=True,
            project=test_output_dir,
            name=os.path.splitext(os.path.basename(img_path))[0],
            exist_ok=True
        )

    print(f"[✓] Saved test predictions for sample images in: {test_output_dir}/")

if __name__ == "__main__":
    main()