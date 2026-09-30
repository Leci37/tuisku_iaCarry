import os
import shutil
import random
from sklearn.model_selection import train_test_split
from ultralytics import YOLO

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

print(f"[\u2713] Dataset prepared at: {OUTPUT_DIR}")
print(f"[\u2713] YAML config written to: {yaml_path}")

# Train YOLOv8 with augmentations
os.makedirs(MODEL_OUT_DIR, exist_ok=True)
model = YOLO("yolov8n.pt")
results = model.train(
    data=yaml_path,
    epochs=50,
    imgsz=IMG_SIZE,
    batch=16,
    project=MODEL_OUT_DIR,
    name="yolov8_gui041_augmented",
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

# Save one test prediction
test_img = f"{OUTPUT_DIR}/val/images/{val_ids[0]}.png"
if not os.path.exists(test_img):
    test_img = test_img.replace(".png", ".jpg")

model.predict(
    source=test_img,
    save=True,
    save_txt=True,
    project=MODEL_OUT_DIR,
    name="yolov8_test_result"
)

print(f"[\u2713] Sample prediction saved in: {MODEL_OUT_DIR}/yolov8_test_result/")