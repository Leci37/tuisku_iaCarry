import os
import json
import numpy as np
import cv2
import shutil
import sys
from y_020_label_gui_Utils import draw_bboxes_and_labels
from your_model_wrapper import init_model, track_with_mask_refinement
from y_030_segment_video_from_tagging_Utils import (
    load_video_frames,
    mask_to_polygon,
    save_qc_overlay,
    save_video,
    init_coco_structure, clean_and_filter_mask, write_global_yolo_classes
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
        if m.shape != (h, w):
            print(f"[WARN] Resizing mask '{label}' from {m.shape} to {(h, w)}")
            m = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
        template_mask[m > 0] = i

    model.xmem.clear_memory()
    # masks, logits, painted_frames = model.generator(images=frames, template_mask=template_mask)
    masks, painted_frames = track_with_mask_refinement(model, frames, template_mask)

    model.xmem.clear_memory()

    os.makedirs(output_dir, exist_ok=True)
    mask_dir = os.path.join(output_dir, "masks")
    qc_dir = os.path.join(output_dir, "qc_pngs")
    yolo_bbox_dir = os.path.join(output_dir, "yolo_labels")
    yolo_seg_dir = os.path.join(output_dir, "yolo_segments")
    os.makedirs(mask_dir, exist_ok=True)
    os.makedirs(qc_dir, exist_ok=True)
    os.makedirs(yolo_bbox_dir, exist_ok=True)
    os.makedirs(yolo_seg_dir, exist_ok=True)

    # Read and copy global class index
    global_classes_path = os.path.join(os.path.dirname(output_dir), "classes.txt")
    with open(global_classes_path, "r", encoding="utf-8") as f:
        class_names = [line.strip() for line in f if line.strip()]
    label_to_class_id = {name: i for i, name in enumerate(class_names)}
    shutil.copy(global_classes_path, os.path.join(output_dir, "classes.txt"))

    label_colors = {name: meta["color"] for name, meta in labels.items()}
    coco = {
        "images": [],
        "annotations": [],
        "categories": [
            {"id": i, "name": name, "supercategory": "object"}
            for i, name in enumerate(class_names)
        ]
    }

    frame_annotations = []
    bbox_frames = []
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
        skip_frame = False
        discarded_reasons = []

        bbox_lines = []
        segment_lines = []
        coco_annos = []

        for label in labels:
            uid = labels[label]["id"]
            if label not in label_to_class_id:
                print(f"[WARN] Label '{label}' not found in global class index. Skipping.")
                continue
            class_index = label_to_class_id[label]
            binary_mask = (mask == uid).astype(np.uint8)
            cleaned_mask, discard_reason = clean_and_filter_mask(binary_mask)

            if cleaned_mask is None:
                if discard_reason == "no_contours":
                    print(f"[INFO] Skipping label '{label}' on frame {i:05d} → {discard_reason}")
                    continue  # Skip just this label
                else:
                    print(f"[INFO] Discarding frame {i:05d} due to label '{label}' → {discard_reason}")
                    discarded_reasons.append(f"{label}_{discard_reason}")
                    skip_frame = True
                    break  # Discard full frame

            y_indices, x_indices = np.where(cleaned_mask > 0)
            x_min, x_max = int(np.min(x_indices)), int(np.max(x_indices))
            y_min, y_max = int(np.min(y_indices)), int(np.max(y_indices))
            bbox = [x_min, y_min, x_max - x_min, y_max - y_min]
            area = int(np.sum(cleaned_mask))
            segmentation = mask_to_polygon(cleaned_mask)

            frame_anno["objects"].append({
                "label": label,
                "id": uid,
                "bbox": [x_min, y_min, x_max, y_max],
                "area": area,
                "segmentation": segmentation,
                "mask_file": os.path.relpath(frame_mask_path, output_dir).replace("\\", "/")
            })

            coco_annos.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": class_index,
                "bbox": bbox,
                "area": area,
                "iscrowd": 0,
                "segmentation": segmentation
            })

            class_index = label_to_class_id[label]
            x_c = (x_min + x_max) / 2 / w
            y_c = (y_min + y_max) / 2 / h
            bw = (x_max - x_min) / w
            bh = (y_max - y_min) / h
            bbox_lines.append(f"{class_index} {x_c:.6f} {y_c:.6f} {bw:.6f} {bh:.6f}")

            for seg in segmentation:
                if len(seg) < 6:
                    continue
                norm_coords = [f"{(seg[j] / w if j % 2 == 0 else seg[j] / h):.6f}" for j in range(len(seg))]
                segment_lines.append(f"{class_index} " + " ".join(norm_coords))

            color = tuple(int(label_colors[label].lstrip('#')[j:j + 2], 16) for j in (0, 2, 4))[::-1]
            cv2.rectangle(annotated, (x_min, y_min), (x_max, y_max), color, 2)
            text = f"{label} ({uid})"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
            cv2.rectangle(annotated, (x_min, y_min - th - 6), (x_min + tw, y_min), color, -1)
            cv2.putText(annotated, text, (x_min, y_min - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        if skip_frame:
            if i % 40 == 0:
                reason_str = "_".join(discarded_reasons)
                qc_discard_path = os.path.join(qc_dir, f"{i:05d}_FRAME_DISC_{reason_str}.png")
                cv2.imwrite(qc_discard_path, cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
            else:
                print(
                    f"[SKIP_QC] Frame {i:05d} discarded (reason: {', '.join(discarded_reasons)}), not saved to conserve memory.")
            continue  # Always skip frame from annotation

        # Save YOLO txts only if valid
        with open(os.path.join(yolo_bbox_dir, f"{i:05d}.txt"), "w") as f:
            f.write("\n".join(bbox_lines) + "\n" if bbox_lines else "")
        with open(os.path.join(yolo_seg_dir, f"{i:05d}.txt"), "w") as f:
            f.write("\n".join(segment_lines) + "\n" if segment_lines else "")

        # Save annotations
        coco["images"].append({
            "id": img_id,
            "file_name": f"{i:05d}.npy",
            "width": w,
            "height": h
        })
        coco["annotations"].extend(coco_annos)
        ann_id += len(coco_annos)

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


def sort_key(filename):
    name = filename.lower()
    if "a_mini" in name:
        return (0, name)
    elif "vid_" in name:
        return (1, name)
    elif "_z" in name:
        return (3, name)
    else:
        return (2, name)

if __name__ == "__main__":
    sys.path.append("/workspace")
    # base_video_dir = r"C:\Users\leci\Documents\GitHub\tuisku_iaCarry\RAW_split2"
    if not os.path.exists("RAW_split2"):
        base_video_dir = r"C:\Users\leci\Documents\GitHub\tuisku_iaCarry\RAW_split2"
    else:
        base_video_dir = "RAW_split2"
    base_gui_output = "gui_output"
    label_map_path = "label_map.pbtxt"
    base_output = "gui_video_segmen"
    write_global_yolo_classes(label_map_path, base_output)

    # Collect and sort .mp4 files
    video_files = [f for f in os.listdir(base_video_dir) if f.endswith(".mp4")]
    # sorted_videos = sorted(video_files, key=sort_key)  LUIS
    sorted_videos = [f for f in os.listdir(base_video_dir) if f.endswith(".mp4") and "n2 (" in f][::-1]

    print("[INFO] Processing order VIDEOS:")
    for f in sorted_videos:
        print(f"  - {f}")

    for fname in sorted_videos[:]:#LUIS
        video_path = os.path.join(base_video_dir, fname)
        video_id = os.path.splitext(fname)[0]
        gui_output_dir = os.path.join(base_gui_output, video_id)
        output_dir = os.path.join(base_output, video_id)
        output_video_path = os.path.join(output_dir, "segmented_video.mp4")

        if not os.path.exists(gui_output_dir):
            print(f"[SKIP] No gui_output found for {video_id}")
            continue

        if os.path.exists(output_video_path):
            print(f"[SKIP] Segmentation already exists for {video_id}")
            continue

        print(f"[INFO] Processing video: {video_id}")
        segment_video_from_tagging(gui_output_dir, video_path, output_dir)


