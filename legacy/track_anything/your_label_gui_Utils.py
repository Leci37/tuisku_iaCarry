import os
import json
import cv2
import numpy as np
from your_utils import overlay_mask

def select_label(display_label, label_display_to_internal, label_color_map, state):
    internal_label = label_display_to_internal[display_label]
    state["selected_label"] = internal_label
    hex_color = label_color_map.get(internal_label, "#FFFFFF")
    html_box = f"<div style='width:30px;height:20px;background:{hex_color};border:1px solid #000;'></div>"
    return f"Selected: {display_label}", html_box

def clear_all_selections(state):
    state["selected_label"] = None
    state["masks"] = {}
    state["label_ids"] = {}
    state["points"] = []
    state["image"] = state["frame"].copy()
    return state["image"], "Cleared all labels.", ""


def save_npy_mask_as_image(npy_path, image_output_path, mask_color=(0, 255, 0)):
    mask = np.load(npy_path)
    mask_img = np.zeros((*mask.shape, 3), dtype=np.uint8)
    mask_img[mask > 0] = mask_color
    cv2.imwrite(image_output_path, cv2.cvtColor(mask_img, cv2.COLOR_RGB2BGR))


def draw_bboxes_and_labels(image, masks, label_ids, label_color_map):
    img = image.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.9  # ← Bigger text
    font_thickness = 2

    for uid, label in label_ids.items():
        mask = masks[label]
        if np.sum(mask) == 0:
            continue

        y_indices, x_indices = np.where(mask > 0)
        if len(x_indices) == 0 or len(y_indices) == 0:
            continue

        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)

        hex_color = label_color_map.get(label, "#00FF00")
        color = tuple(int(hex_color.lstrip('#')[i:i + 2], 16) for i in (0, 2, 4))
        color_bgr = color[::-1]

        # Draw bounding box
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color_bgr, 2)

        # Draw label background
        label_text = f"{label} ({uid})"
        (text_w, text_h), _ = cv2.getTextSize(label_text, font, font_scale, font_thickness)
        cv2.rectangle(img, (x_min, y_min - text_h - 6), (x_min + text_w, y_min), color_bgr, -1)

        # Draw label text
        cv2.putText(img, label_text, (x_min, y_min - 4), font, font_scale, (255, 255, 255), font_thickness)

    return img


def save_masks_and_image(state, label_color_map, save_individual_images=False):
    if not state["masks"]:
        return "No masks to save."

    video_subdir = state.get("video_name", "video")
    output_dir = os.path.join("gui_output", video_subdir)
    os.makedirs(output_dir, exist_ok=True)
    base_name = "user_frame"

    label_ids = {i + 1: name for i, name in enumerate(state["masks"].keys())}
    reverse_ids = {v: k for k, v in label_ids.items()}

    metadata = {
        "video_name": video_subdir,
        "frame_index": 0,
        "labels": {},
        "combined_mask": f"{base_name}_segmented.png",
        "bbox_mask_overlay": f"{base_name}_bbox_overlay.png"
    }

    combined_mask = np.zeros_like(next(iter(state["masks"].values())), dtype=np.uint8)

    for name, m in state["masks"].items():
        uid = reverse_ids[name]
        combined_mask[m > 0] = uid

        mask_file = f"{base_name}_{name}.npy"
        mask_path = os.path.join(output_dir, mask_file)
        np.save(mask_path, m)

        if save_individual_images:
            image_path = os.path.join(output_dir, f"{base_name}_{name}.png")
            save_npy_mask_as_image(mask_path, image_path)

        metadata["labels"][name] = {
            "id": uid,
            "mask_file": mask_file,
            "color": label_color_map.get(name, "#00FF00")
        }

    # Save combined segmentation overlay
    overlay = overlay_mask(state["frame"], combined_mask, label_color_map, label_ids)
    segmented_path = os.path.join(output_dir, f"{base_name}_segmented.png")
    cv2.imwrite(segmented_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    # Save annotated image with bbox and label text
    annotated = draw_bboxes_and_labels(overlay, state["masks"], label_ids, label_color_map)
    bbox_path = os.path.join(output_dir, f"{base_name}_bbox_overlay.png")
    cv2.imwrite(bbox_path, cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))

    # Save metadata
    metadata_path = os.path.join(output_dir, "tagging_metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    return f"Saved to {output_dir}/"


def draw_bboxes_and_labels(image, masks, label_ids, label_color_map):
    img = image.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.9
    font_thickness = 2

    for uid, label in label_ids.items():
        mask = masks[label]
        if np.sum(mask) == 0:
            continue

        y_indices, x_indices = np.where(mask > 0)
        if len(x_indices) == 0 or len(y_indices) == 0:
            continue

        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)

        hex_color = label_color_map.get(label, "#00FF00")
        color = tuple(int(hex_color.lstrip('#')[i:i + 2], 16) for i in (0, 2, 4))
        color_bgr = color[::-1]

        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color_bgr, 2)
        label_text = f"{label} ({uid})"
        (text_w, text_h), _ = cv2.getTextSize(label_text, font, font_scale, font_thickness)
        cv2.rectangle(img, (x_min, y_min - text_h - 6), (x_min + text_w, y_min), color_bgr, -1)
        cv2.putText(img, label_text, (x_min, y_min - 4), font, font_scale, (255, 255, 255), font_thickness)

    return img
