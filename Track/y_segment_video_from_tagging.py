import os
import json
import numpy as np
import cv2
from your_label_gui_Utils import draw_bboxes_and_labels
from your_model_wrapper import init_model
from y_segment_video_from_tagging_Utils import (
    load_video_frames,
    mask_to_polygon,
    save_qc_overlay,
    save_video,
    init_coco_structure
)

def segment_video_from_tagging(gui_output_dir, video_path, output_dir):
    model = init_model()

    metadata_path = os.path.join(gui_output_dir, "tagging_metadata.json")
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    labels = metadata["labels"]
    first_mask_files = {name: os.path.join(gui_output_dir, meta["mask_file"]) for name, meta in labels.items()}

    frames = load_video_frames(video_path)
    if not frames:
        raise ValueError("No frames extracted from video")

    first_frame = frames[0]
    h, w = first_frame.shape[:2]

    model.samcontroler.sam_controler.reset_image()
    model.samcontroler.sam_controler.set_image(first_frame)

    template_mask = np.zeros((h, w), dtype=np.uint8)
    for i, (label, path) in enumerate(first_mask_files.items(), start=1):
        m = np.load(path)
        template_mask[m > 0] = i

    model.xmem.clear_memory()
    masks, logits, painted_frames = model.generator(images=frames, template_mask=template_mask)
    model.xmem.clear_memory()

    os.makedirs(output_dir, exist_ok=True)
    mask_dir = os.path.join(output_dir, "masks")
    qc_dir = os.path.join(output_dir, "qc_pngs")
    os.makedirs(mask_dir, exist_ok=True)
    os.makedirs(qc_dir, exist_ok=True)

    frame_annotations = []
    bbox_frames = []

    label_ids = {meta["id"]: name for name, meta in labels.items()}
    label_colors = {name: meta["color"] for name, meta in labels.items()}
    coco = init_coco_structure(labels)

    ann_id = 1
    for i, (frame, mask) in enumerate(zip(painted_frames, masks)):
        frame_anno = {
            "frame_index": i,
            "objects": []
        }
        img_id = i + 1

        frame_mask_path = os.path.join(mask_dir, f"{i:05d}.npy")
        np.save(frame_mask_path, mask)

        annotated = frame.copy()
        coco["images"].append({
            "id": img_id,
            "file_name": f"{i:05d}.npy",
            "width": w,
            "height": h
        })

        for uid, label in label_ids.items():
            binary_mask = (mask == uid).astype(np.uint8)
            if np.sum(binary_mask) == 0:
                continue

            y_indices, x_indices = np.where(binary_mask > 0)
            x_min, x_max = int(np.min(x_indices)), int(np.max(x_indices))
            y_min, y_max = int(np.min(y_indices)), int(np.max(y_indices))
            bbox = [x_min, y_min, x_max - x_min, y_max - y_min]
            area = int(np.sum(binary_mask))
            segmentation = mask_to_polygon(binary_mask)

            frame_anno["objects"].append({
                "label": label,
                "id": uid,
                "bbox": [x_min, y_min, x_max, y_max],
                "area": area,
                "segmentation": segmentation,
                "mask_file": os.path.relpath(frame_mask_path, output_dir).replace("\\", "/")
            })

            coco["annotations"].append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": uid,
                "bbox": bbox,
                "area": area,
                "iscrowd": 0,
                "segmentation": segmentation
            })
            ann_id += 1

            color = tuple(int(label_colors[label].lstrip('#')[j:j + 2], 16) for j in (0, 2, 4))[::-1]
            cv2.rectangle(annotated, (x_min, y_min), (x_max, y_max), color, 2)
            text = f"{label} ({uid})"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
            cv2.rectangle(annotated, (x_min, y_min - th - 6), (x_min + tw, y_min), color, -1)
            cv2.putText(annotated, text, (x_min, y_min - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        bbox_frames.append(annotated)
        frame_annotations.append(frame_anno)

        if i == 0 or i % 60 == 0:
            save_qc_overlay(annotated, qc_dir, i)

    save_video(painted_frames, os.path.join(output_dir, "segmented_video.mp4"))
    save_video(bbox_frames, os.path.join(output_dir, "segmented_video_with_bbox.mp4"))

    with open(os.path.join(output_dir, "frames_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(frame_annotations, f, indent=4)

    with open(os.path.join(output_dir, "coco_annotations.json"), "w", encoding="utf-8") as f:
        json.dump(coco, f, indent=4)

    print(f"[✓] Outputs saved in: {output_dir}")


if __name__ == "__main__":
    video_path = "TEST_troley_3s.mp4"
    gui_output_dir = "gui_output/TEST_troley_3s"
    output_dir = "gui_video_segs/TEST_troley_3s/segmentation_result"

    segment_video_from_tagging(gui_output_dir, video_path, output_dir)
