import gradio as gr
import os
import cv2
import numpy as np
from your_model_wrapper import init_model
from your_utils import parse_label_map, overlay_mask, build_legend_html, format_label
from y_020_label_gui_Utils import (
    select_label, clear_all_selections, save_masks_and_image,
    draw_bboxes_and_labels, draw_points_on_image,
    load_first_frame, load_all_video_paths,
    load_video_by_index, next_video, prev_video,
    get_overlay_or_placeholder, validate_masks_before_save
)

VIDEO_DIR = r"C:\Users\leci\Documents\GitHub\tuisku_iaCarry\RAW_split2"
model = init_model()

# Load label map
raw_label_map = parse_label_map("label_map.pbtxt")
label_internal_to_display = {entry["name"]: format_label(entry["name"]) for entry in raw_label_map.values()}
label_display_to_internal = {v: k for k, v in label_internal_to_display.items()}
label_display_names = list(label_display_to_internal.keys())
label_color_map = {entry["name"]: entry["hex_color"] for entry in raw_label_map.values()}
legend_html = build_legend_html(raw_label_map)

state = {
    "image": None,
    "frame": None,
    "selected_label": None,
    "masks": {},
    "label_ids": {},
    "clicks": {},
    "video_name": "video",
    "video_paths": [],
    "current_index": 0,
    "confirm_overwrite": False
}

def handle_click(evt: gr.SelectData, point_type):
    if state["selected_label"] is None or state["frame"] is None:
        return state["image"], None, None

    x, y = evt.index[0], evt.index[1]
    is_positive = point_type == "Positive"
    label = state["selected_label"]

    print(f"[CLICK] {label} — {'Positive' if is_positive else 'Negative'} point at ({x}, {y})")

    if label not in state["clicks"]:
        state["clicks"][label] = []
    state["clicks"][label].append((x, y, is_positive))

    model.samcontroler.sam_controler.reset_image()
    model.samcontroler.sam_controler.set_image(state["frame"])

    points, labels = zip(*[((px, py), 1 if pos else 0) for (px, py, pos) in state["clicks"][label]])
    mask, _, _ = model.first_frame_click(
        image=state["frame"],
        points=np.array(points),
        labels=np.array(labels),
        multimask="True"
    )

    state["masks"][label] = mask
    label_ids = {i + 1: name for i, name in enumerate(state["masks"].keys())}
    reverse_ids = {v: k for k, v in label_ids.items()}
    combined_mask = np.zeros_like(mask, dtype=np.uint8)
    for name, m in state["masks"].items():
        combined_mask[m > 0] = reverse_ids[name]

    state["label_ids"] = label_ids
    print(f"[MASK] {label} mask updated. Total masks: {len(state['masks'])}")
    painted = overlay_mask(state["frame"], combined_mask, label_color_map, label_ids)
    painted = draw_points_on_image(painted, state["clicks"])
    state["image"] = painted

    annotated = draw_bboxes_and_labels(painted, state["masks"], label_ids, label_color_map, box_thickness=7)
    return painted, annotated, annotated

def save_all():
    print("📝 [SAVE] Saving masks and image...")
    print(f" - Video name: {state['video_name']}")
    print(f" - Masks: {list(state['masks'].keys())}")
    result = save_masks_and_image(state, label_color_map)
    print("✅ [SAVE COMPLETE]")
    overlay = get_overlay_or_placeholder(state["video_name"])
    return result, overlay, overlay

with gr.Blocks(css="#right-col { display: flex; flex-direction: column; justify-content: flex-end; height: 100%; }") as demo:
    gr.Markdown("### 🎯 Label-based Point Segmentation + 🎞️ Video Navigation")

    with gr.Row():
        with gr.Column(scale=70):
            label_status = gr.Textbox(label="📁 Current Video")
            video_display = gr.Image(type="numpy", interactive=True, height=600)
            label_info = gr.Textbox(label="Status")
            full_overlay_display = gr.Image(type="numpy", visible=True, label="🔍 Full Overlay View", height=700)
            save_button = gr.Button("💾 Save Masks + Image")
            gr.HTML(value=legend_html)

        with gr.Column(scale=30, elem_id="right-col"):
            prev_button = gr.Button("◀ Previous Video")
            next_button = gr.Button("Next Video ▶")
            overlay_display = gr.Image(type="numpy", label="🖼️ Overlay", height=280, elem_id="hover-overlay")
            clear_all_button = gr.Button("🧹 Clear All Selections")
            label_dropdown = gr.Radio(choices=label_display_names, label="🏷️ Select Label", interactive=True)
            point_type = gr.Radio(choices=["Positive", "Negative"], value="Positive", label="🖱️ Click Type")
            color_display = gr.HTML()

    video_display.select(
        fn=handle_click,
        inputs=point_type,
        outputs=[video_display, overlay_display, full_overlay_display]
    )

    save_button.click(
        fn=save_all,
        outputs=[label_info, overlay_display, full_overlay_display]
    )

    label_dropdown.change(
        fn=lambda label: select_label(label, label_display_to_internal, label_color_map, state),
        inputs=label_dropdown,
        outputs=[label_info, color_display]
    )

    clear_all_button.click(
        fn=lambda: clear_all_selections(state),
        outputs=[video_display, label_info, color_display]
    )

    prev_button.click(
        fn=lambda: (
            print(f"[NAV] ← Previous video"),
            *prev_video(state, VIDEO_DIR)
        )[1:],  # skip print return
        outputs=[video_display, label_status, overlay_display, full_overlay_display]
    )

    next_button.click(
        fn=lambda: (
            print(f"[NAV] → Next video"),
            *next_video(state, VIDEO_DIR)
        )[1:],
        outputs=[video_display, label_status, overlay_display, full_overlay_display]
    )

    demo.load(
        fn=lambda: (
            print("[INIT] Loading first video..."),
            *load_video_by_index(state, 0, VIDEO_DIR)
        )[1:],
        outputs=[video_display, label_status, overlay_display, full_overlay_display]
    )

demo.launch()
