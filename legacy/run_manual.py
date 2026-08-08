import os
from your_model_wrapper import init_model, segment_objects
from your_utils import parse_label_map, overlay_mask
import cv2
import easygui
import numpy as np


def extract_first_frame(video_path):
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise ValueError(f"Could not read frame from {video_path}")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


import cv2


def save_masks(masks, save_dir, base_name, original_image):
    os.makedirs(save_dir, exist_ok=True)

    combined_mask = np.zeros(original_image.shape[:2], dtype=np.uint8)

    for i, (label, mask) in enumerate(masks.items(), start=1):
        # Save mask
        path = os.path.join(save_dir, f"{base_name}_{label}.npy")
        # np.save(path, mask)

        # Assign unique value per object for coloring
        combined_mask[mask > 0] = i

    # Overlay segmentation shadow
    painted = overlay_mask(original_image, combined_mask)
    cv2.imwrite(os.path.join(save_dir, f"{base_name}_segmented.png"), cv2.cvtColor(painted, cv2.COLOR_RGB2BGR))


def run_batch_segmentation(video_dir, label_map_path, save_root):
    model = init_model()
    label_map = parse_label_map(label_map_path)
    all_labels = list(label_map.values())

    selected_labels = easygui.multchoicebox(
        msg="Select labels to segment:",
        title="Label Selection",
        choices=all_labels
    )

    if not selected_labels:
        print("[ABORTED] No labels selected.")
        return

    for fname in os.listdir(video_dir):
        if not fname.lower().endswith('.mp4'):
            continue
        video_path = os.path.join(video_dir, fname)
        try:
            frame = extract_first_frame(video_path)
        except Exception as e:
            print(f"[ERROR] Failed to process {fname}: {e}")
            continue

        masks = segment_objects(model, frame, selected_labels)
        base_name = os.path.splitext(fname)[0]
        output_path = os.path.join(save_root, base_name)
        save_masks(masks, output_path, base_name, frame)
        print(f"[INFO] Processed {fname} -> {output_path}")



if __name__ == "__main__":
    video_dir = r"C:\Users\leci\Documents\GitHub\tuisku_iaCarry\RAW"
    label_map_path = "label_map.pbtxt"
    output_dir = "./batch_output"

    run_batch_segmentation(video_dir, label_map_path, output_dir)