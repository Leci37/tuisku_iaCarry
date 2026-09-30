import os
import cv2
import numpy as np
import gradio as gr

from segment_anything import sam_model_registry
from tracker.model.inference import XMem
from track_anything import TrackingAnything


class TrackAnythingWrapper:
    def __init__(self, model_type="vit_h", sam_checkpoint="models/sam_vit_h.pth", xmem_checkpoint="models/XMem.pth", device="cuda:0"):
        # Load SAM model
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device=device)
        self.tracker = TrackingAnything(sam, xmem_checkpoint, device=device)
        self.video_frames = []
        self.masks = []
        self.bboxes = []

    def load_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        self.video_frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            self.video_frames.append(frame)
        cap.release()
        self.tracker.set_images(self.video_frames)

    def set_initial_click(self, frame_idx, x, y):
        image = self.video_frames[frame_idx]
        self.tracker.set_first_frame(image, frame_idx, point_coords=[[x, y]], point_labels=[1])

    def run_tracking(self):
        self.masks = []
        self.bboxes = []
        for idx in range(len(self.video_frames)):
            mask, log = self.tracker.generator(idx)
            self.masks.append(mask)
            bbox = self.mask_to_bbox(mask)
            self.bboxes.append(bbox)
        return self.masks, self.bboxes

    @staticmethod
    def mask_to_bbox(mask):
        if mask is None:
            return None
        x, y, w, h = cv2.boundingRect(mask.astype(np.uint8))
        return [x, y, x + w, y + h]

    def export_coco_annotations(self, class_name="product", start_frame=0):
        annotations = []
        for i, bbox in enumerate(self.bboxes):
            if bbox:
                annotations.append({
                    "frame": start_frame + i,
                    "bbox": bbox,
                    "class": class_name
                })
        return annotations

# Gradio GUI
tracker = TrackAnythingWrapper()


with gr.Blocks() as demo:
    video_input = gr.Video(label="Upload Video")
    click_frame = gr.Slider(0, 100, label="Initial Frame Index", step=1)
    click_x = gr.Number(label="Click X")
    click_y = gr.Number(label="Click Y")
    track_button = gr.Button("Run Tracking")
    output_json = gr.JSON(label="COCO Bounding Boxes")

    def process_video(video_path, frame_idx, x, y):
        tracker.load_video(video_path)
        tracker.set_initial_click(int(frame_idx), int(x), int(y))
        _, bboxes = tracker.run_tracking()
        return tracker.export_coco_annotations()

    track_button.click(fn=process_video, inputs=[video_input, click_frame, click_x, click_y], outputs=output_json)

if __name__ == "__main__":
    demo.launch()
