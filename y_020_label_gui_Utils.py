import os
import json
import re
import cv2
import numpy as np
from your_utils import overlay_mask
from your_model_wrapper import init_model


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
    state["clicks"] = {}  # ✅ Clear label-specific points
    state["image"] = state["frame"].copy()
    print("[CLEAR] All masks, labels, and clicks have been cleared.")
    return state["image"], "Cleared all labels.", ""



def save_npy_mask_as_image(npy_path, image_output_path, mask_color=(0, 255, 0)):
    mask = np.load(npy_path)
    mask_img = np.zeros((*mask.shape, 3), dtype=np.uint8)
    mask_img[mask > 0] = mask_color
    cv2.imwrite(image_output_path, cv2.cvtColor(mask_img, cv2.COLOR_RGB2BGR))


def draw_bboxes_and_labels(image, masks, label_ids, label_color_map, box_thickness=2):
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

        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color_bgr, box_thickness)
        label_text = f"{label} ({uid})"
        (text_w, text_h), _ = cv2.getTextSize(label_text, font, font_scale, font_thickness)
        cv2.rectangle(img, (x_min, y_min - text_h - 6), (x_min + text_w, y_min), color_bgr, -1)
        cv2.putText(img, label_text, (x_min, y_min - 4), font, font_scale, (255, 255, 255), font_thickness)

    return img


def draw_points_on_image(image, clicks, radius=8):
    image_copy = image.copy()
    for label_clicks in clicks.values():
        for x, y, is_positive in label_clicks:
            color = (0, 255, 0) if is_positive else (255, 0, 0)
            cv2.circle(image_copy, (x, y), radius=radius, color=color, thickness=-1)
    return image_copy


def load_first_frame(video_path):
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return None
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    print(f"[LOAD] First frame extracted from: {video_path}")
    return rgb


def load_all_video_paths(video_dir):
    return [os.path.join(video_dir, f) for f in sorted(os.listdir(video_dir)) if f.endswith(".mp4")]


def load_video_by_index(state, index, video_dir):
    if not state["video_paths"]:
        state["video_paths"] = load_all_video_paths(video_dir)
    if not state["video_paths"]:
        return None, "No videos found.", None
    index = index % len(state["video_paths"])
    state["current_index"] = index

    frame = load_first_frame(state["video_paths"][index])
    state["frame"] = frame
    state["image"] = frame.copy()
    state["masks"] = {}
    state["label_ids"] = {}
    state["clicks"] = {}
    state["video_name"] = os.path.splitext(os.path.basename(state["video_paths"][index]))[0]

    model = init_model()
    model.samcontroler.sam_controler.reset_image()
    model.samcontroler.sam_controler.set_image(frame)

    overlay_image = get_overlay_or_placeholder(state["video_name"])
    return frame, os.path.basename(state["video_paths"][index]), overlay_image, overlay_image


def next_video(state, video_dir):
    return load_video_by_index(state, state["current_index"] + 1, video_dir)


def prev_video(state, video_dir):
    return load_video_by_index(state, state["current_index"] - 1, video_dir)


def get_overlay_or_placeholder(video_name):
    output_path = os.path.join("gui_output", video_name, "user_frame_bbox_overlay.png")
    if os.path.exists(output_path):
        image = cv2.imread(output_path)
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        placeholder = np.ones((280, 500, 3), dtype=np.uint8) * 240
        cv2.putText(placeholder, "No Label", (30, 150), cv2.FONT_HERSHEY_SIMPLEX, 2, (100, 100, 100), 4)
        return placeholder


def save_masks_and_image(state, label_color_map, save_individual_images=False):
    video_subdir = state.get("video_name", "video")
    output_dir = os.path.join("gui_output", video_subdir)

    ok, msg = validate_masks_before_save(state, output_dir)
    if not ok:
        state["confirm_overwrite"] = True
        return msg
    state["confirm_overwrite"] = False

    print(f"[SAVE] Saving masks to: {output_dir}")
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
        print(f"  - Saving mask for '{name}'")
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

    overlay = overlay_mask(state["frame"], combined_mask, label_color_map, label_ids)
    segmented_path = os.path.join(output_dir, f"{base_name}_segmented.png")
    cv2.imwrite(segmented_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    annotated = draw_bboxes_and_labels(overlay, state["masks"], label_ids, label_color_map)
    bbox_path = os.path.join(output_dir, f"{base_name}_bbox_overlay.png")
    cv2.imwrite(bbox_path, cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))

    metadata_path = os.path.join(output_dir, "tagging_metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    print(f"[SAVE DONE] Metadata written to: {metadata_path}")
    return f"✅ Saved to {output_dir}/"


def validate_masks_before_save(state, output_dir):
    if not state["masks"]:
        return False, "⚠️ No masks found. Please add at least one mask before saving."

    for name, mask in state["masks"].items():
        if not isinstance(mask, np.ndarray):
            return False, f"❌ Mask for '{name}' is not a valid numpy array."
        if mask.ndim != 2:
            return False, f"❌ Mask for '{name}' must be 2D. Found {mask.ndim}D."

    invalid = [name for name in state["masks"] if not re.match(r'^[\w\-]+$', name)]
    if invalid:
        return False, f"❌ Invalid label names: {', '.join(invalid)}. Only a-z, 0-9, _ and - are allowed."

    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
        except Exception as e:
            return False, f"❌ Cannot create output folder '{output_dir}': {e}"

    if not os.access(output_dir, os.W_OK):
        return False, f"❌ Cannot write to output folder: {output_dir}"

    base_name = "user_frame"
    existing = []
    for label in state["masks"]:
        path = os.path.join(output_dir, f"{base_name}_{label}.npy")
        if os.path.exists(path):
            existing.append(label)

    if existing and not state.get("confirm_overwrite", False):
        return False, (
            f"⚠️ The following masks already exist: {', '.join(existing)}.\n"
            f"Click 'Save' again to confirm overwrite."
        )

    return True, ""
