import gradio as gr
import cv2
import numpy as np
from your_model_wrapper import init_model
from your_utils import parse_label_map, overlay_mask, build_legend_html, format_label
from your_label_gui_Utils import select_label, clear_all_selections, save_masks_and_image
import os

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
    "masks": {},  # name → mask
    "label_ids": {},  # id → name
    "points": [],
    "video_name": "video"  # default fallback name
}


def load_first_frame(video_path):
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return None
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    state["frame"] = rgb
    state["image"] = rgb.copy()
    state["masks"] = {}
    state["label_ids"] = {}
    state["points"] = []

    # Store video name
    state["video_name"] = os.path.splitext(os.path.basename(video_path))[0]

    model.samcontroler.sam_controler.reset_image()
    model.samcontroler.sam_controler.set_image(rgb)
    return rgb


def draw_points_on_image(image, points, color=(0, 255, 0), radius=8):
    image_copy = image.copy()
    for pt in points:
        cv2.circle(image_copy, pt, radius=radius, color=color, thickness=-1)
    return image_copy


def handle_click(evt: gr.SelectData):
    if state["selected_label"] is None or state["frame"] is None:
        return state["image"]

    x, y = evt.index[0], evt.index[1]
    point = np.array([[x, y]])
    labels = np.array([1])

    mask, _, _ = model.first_frame_click(
        image=state["frame"],
        points=point,
        labels=labels,
        multimask="True"
    )

    label = state["selected_label"]
    state["masks"][label] = mask
    label_ids = {i + 1: name for i, name in enumerate(state["masks"].keys())}
    reverse_ids = {v: k for k, v in label_ids.items()}

    combined_mask = np.zeros_like(mask, dtype=np.uint8)
    for name, m in state["masks"].items():
        combined_mask[m > 0] = reverse_ids[name]

    state["label_ids"] = label_ids
    state["points"].append((x, y))

    painted = overlay_mask(state["frame"], combined_mask, label_color_map, label_ids)
    painted = draw_points_on_image(painted, state["points"])
    state["image"] = painted
    return painted


with gr.Blocks(css="#right-col { display: flex; flex-direction: column; justify-content: flex-end; height: 100%; }") as demo:
    gr.Markdown("### Label-based Point Segmentation")

    with gr.Row():
        with gr.Column(scale=9):
            video_input = gr.Video()
        with gr.Column(scale=1, elem_id="right-col"):
            load_button = gr.Button("Load First Frame")

    with gr.Row():
        with gr.Column(scale=76):
            image_display = gr.Image(type="numpy", interactive=True, height=500)
            label_info = gr.Textbox(label="Status")
            gr.HTML(value=legend_html)

        with gr.Column(scale=24):
            clear_all_button = gr.Button("Clear All Selections")
            label_dropdown = gr.Radio(choices=label_display_names, label="Select Label", interactive=True)
            color_display = gr.HTML()

    save_button = gr.Button("Save Masks + Image")

    load_button.click(fn=load_first_frame, inputs=video_input, outputs=image_display)
    clear_all_button.click(
        fn=lambda: clear_all_selections(state),
        outputs=[image_display, label_info, color_display]
    )
    label_dropdown.change(
        fn=lambda label: select_label(label, label_display_to_internal, label_color_map, state),
        inputs=label_dropdown,
        outputs=[label_info, color_display]
    )
    image_display.select(fn=handle_click, outputs=image_display)
    save_button.click(
        fn=lambda: save_masks_and_image(state, label_color_map),
        outputs=label_info
    )

demo.launch()
