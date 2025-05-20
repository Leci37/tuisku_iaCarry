import os
import cv2
import numpy as np
import json
from collections import defaultdict

from your_utils import get_video_props

HEX_COLOR_MAP = {
    "aguila_33cl": "#0078FF",
    "axedrak_150ml": "#A500FF",
    "chipsahoy_300g": "#FF4C00",
    "cocacola_33cl": "#D10000",
    "colacao_383g": "#00C900",
    "colgate_75ml": "#FFD700",
    "estrellag_33cl": "#FF0033",
    "fanta_33cl": "#004CFF",
    "hysori_300ml": "#66FF00",
    "mahou00_33cl": "#9900FF",
    "mahou5_33cl": "#00FFA2",
    "smacks_330g": "#FF9500",
    "tostarica_570g": "#00E0FF",
    "asturiana_1l": "#B0FF00"
}

# ------------------------------------------
# Product label codes used in frame ID naming
# ------------------------------------------
PRODUCTS_CODES_NAMES = {
    "Ag": "aguila_33cl",
    "Ax": "axedrak_150ml",
    "Ch": "chipsahoy_300g",
    "Cc": "cocacola_33cl",
    "Co": "colacao_383g",
    "Cg": "colgate_75ml",
    "Eg": "estrellag_33cl",
    "Fa": "fanta_33cl",
    "Hy": "hysori_300ml",
    "M0": "mahou00_33cl",
    "M5": "mahou5_33cl",
    "Sm": "smacks_330g",
    "Tr": "tostarica_570g",
    "As": "asturiana_1l"
}

def load_yolo_bbox(txt_path, img_shape):
    h, w = img_shape[:2]
    boxes = []
    if not os.path.exists(txt_path):
        return boxes
    with open(txt_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            class_id, x_c, y_c, bw, bh = map(float, parts)
            x_c, y_c, bw, bh = x_c * w, y_c * h, bw * w, bh * h
            x1 = int(x_c - bw / 2)
            y1 = int(y_c - bh / 2)
            x2 = int(x_c + bw / 2)
            y2 = int(y_c + bh / 2)
            boxes.append((int(class_id), x1, y1, x2, y2))
    return boxes

def load_class_names(class_file):
    with open(class_file, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]

def hex_to_bgr(hex_color):
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i:i+2], 16) for i in (4, 2, 0))

def draw_boxes(frame, boxes, class_names):
    """
    Draw enhanced bounding boxes with clearer lines and readable labels.

    Args:
        frame (np.ndarray): The image to draw on.
        boxes (list): List of (class_id, x1, y1, x2, y2) tuples.
        class_names (list): List of class names by index.

    Returns:
        np.ndarray: Annotated image.
    """
    img = frame.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.9
    font_thickness = 2
    box_thickness = 3

    for class_id, x1, y1, x2, y2 in boxes:
        label = class_names[class_id] if class_id < len(class_names) else str(class_id)
        hex_color = HEX_COLOR_MAP.get(label.lower(), "#00FF00")
        color = hex_to_bgr(hex_color)

        # Draw rectangle
        cv2.rectangle(img, (x1, y1), (x2, y2), color, box_thickness)

        # Text label with background
        label_text = f"{label}"
        (text_w, text_h), _ = cv2.getTextSize(label_text, font, font_scale, font_thickness)

        # Background rectangle
        cv2.rectangle(img, (x1, y1 - text_h - 10), (x1 + text_w + 6, y1), color, -1)

        # Foreground white text
        cv2.putText(img, label_text, (x1 + 3, y1 - 5), font, font_scale, (255, 255, 255), font_thickness)

    return img

def render_yolo_segmentation_overlay(frame, segment_path, classes_path, class_names=None, alpha=0.4):
    """
    Draws polygon segments from a YOLO-format .txt on a given image, using label color.

    Args:
        frame (np.ndarray): RGB or BGR image.
        segment_path (str): Path to YOLO segment .txt (polygon format).
        classes_path (str): Path to classes.txt file.
        class_names (list): Optional list of class names to avoid re-reading.
        alpha (float): Overlay transparency.

    Returns:
        np.ndarray: Image with polygons drawn in matching colors.
    """
    h, w = frame.shape[:2]
    overlay = frame.copy()

    if class_names is None:
        with open(classes_path, "r", encoding="utf-8") as f:
            class_names = [line.strip() for line in f if line.strip()]

    if not os.path.exists(segment_path):
        print(f"[WARN] Segment file not found: {segment_path}")
        return frame

    with open(segment_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 8:
                continue  # skip too-short polygons
            class_id = int(parts[0])
            coords = list(map(float, parts[1:]))

            pts = np.array([
                [int(coords[i] * w), int(coords[i + 1] * h)]
                for i in range(0, len(coords), 2)
            ], np.int32).reshape((-1, 1, 2))

            label = class_names[class_id]
            color = hex_to_bgr(HEX_COLOR_MAP.get(label.lower(), "#00FF00"))

            cv2.fillPoly(overlay, [pts], color)

    # Blend overlay with original
    return cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)


def annotate_corner_title(frame, video_id, idx):
    text = f"{video_id} — Frame {idx:05d}"
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, lineType=cv2.LINE_AA)
    return frame

def correct_box(frame, box):
    temp = frame.copy()
    x1, y1, x2, y2 = box[1:]
    dragging = False

    def mouse_callback(event, x, y, flags, param):
        nonlocal x1, y1, x2, y2, dragging
        if event == cv2.EVENT_LBUTTONDOWN:
            dragging = True
            x1, y1 = x, y
        elif event == cv2.EVENT_MOUSEMOVE and dragging:
            temp_display = frame.copy()
            cv2.rectangle(temp_display, (x1, y1), (x, y), (0, 0, 255), 2)
            cv2.imshow("Correct BBox", temp_display)
        elif event == cv2.EVENT_LBUTTONUP:
            dragging = False
            x2, y2 = x, y
            cv2.destroyWindow("Correct BBox")

    cv2.namedWindow("Correct BBox")
    cv2.setMouseCallback("Correct BBox", mouse_callback)
    cv2.imshow("Correct BBox", temp)
    cv2.waitKey(0)
    return box[0], min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)


def save_cleaned_data(frame_name, frame, boxes, IMG_DIR, LABEL_DIR):
    COCO_JSON_PATH = os.path.join(LABEL_DIR, "..", "coco_annotations.json")
    img_out = os.path.join(IMG_DIR, f"{frame_name}.png")
    txt_out = os.path.join(LABEL_DIR, f"{frame_name}.txt")

    # cv2.imwrite(img_out, frame)
    cv2.imwrite(img_out, frame)
    h, w = frame.shape[:2]

    # Write YOLO label file
    with open(txt_out, 'w') as f:
        for cls, x1, y1, x2, y2 in boxes:
            x_c = (x1 + x2) / 2 / w
            y_c = (y1 + y2) / 2 / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h
            f.write(f"{cls} {x_c:.6f} {y_c:.6f} {bw:.6f} {bh:.6f}\n")

    # Generate COCO annotation objects
    annotations = []
    for ann_id, (cls, x1, y1, x2, y2) in enumerate(boxes, start=1):
        bbox = [x1, y1, x2 - x1, y2 - y1]
        annotations.append({
            "id": hash(f"{frame_name}_{ann_id}") % (10**8),
            "image_id": frame_name,
            "category_id": cls,
            "bbox": bbox,
            "area": bbox[2] * bbox[3],
            "iscrowd": 0,
            "segmentation": []
        })

    # If file exists, load it, else create structure
    if os.path.exists(COCO_JSON_PATH):
        with open(COCO_JSON_PATH, 'r') as f:
            coco = json.load(f)
    else:
        coco = {
            "images": [],
            "annotations": [],
            "categories": [
                {"id": i, "name": name, "supercategory": "product"}
                for i, name in enumerate(PRODUCTS_CODES_NAMES.values())
            ]
        }

    # Append new image and annotations
    coco["images"].append({
        "id": frame_name,
        "file_name": f"{frame_name}.png",
        "width": w,
        "height": h
    })
    coco["annotations"].extend(annotations)

    with open(COCO_JSON_PATH, 'w') as f:
        json.dump(coco, f, indent=2)



def resize_for_display(image, max_height=720):
    h, w = image.shape[:2]
    if h > max_height:
        scale = max_height / h
        return cv2.resize(image, (int(w * scale), max_height))
    return image



def draw_key_legend(img):
    legend = [
        "s → Save frame and bbox",
        "c → Enter correction mode",
        "d → Discard and skip",
        "q → Quit the tool"
    ]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    color = (255, 255, 255)
    bg_color = (50, 50, 50)
    x, y0 = 10, img.shape[0] - 80

    for i, line in enumerate(legend):
        y = y0 + i * 20
        (tw, th), _ = cv2.getTextSize(line, font, font_scale, thickness)
        cv2.rectangle(img, (x - 5, y - th - 5), (x + tw + 5, y + 5), bg_color, -1)
        cv2.putText(img, line, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)
    return img


def resize_for_display(image, max_height=720):
    h, w = image.shape[:2]
    if h > max_height:
        scale = max_height / h
        return cv2.resize(image, (int(w * scale), max_height))
    return image

# ------------------------------------------
# Save / naming helpers
# ------------------------------------------
def encode_element_code(labels):
    used = set()
    code = []
    for label in sorted(set(labels)):
        for short, full in PRODUCTS_CODES_NAMES.items():
            if full == label and short not in used:
                code.append(short)
                used.add(short)
                break
    if len(used) == len(PRODUCTS_CODES_NAMES):
        return "All"
    return ''.join(code)

def generate_frame_id(state):
    video_code = ''.join(filter(str.isalnum, state["video_id"][:8]))
    frame_index = state["current_frame_idx"]
    element_count = len(state["bboxes"])
    label_names = [state["class_names"][b[0]] for b in state["bboxes"]]
    element_code = encode_element_code(label_names)
    return f"{video_code}_{frame_index:05d}_n{element_count}_{element_code}"


def prepare_frame_state(frame_meta, frame, boxes, SEGM_DIR, state):

    # Load class names
    class_path = os.path.join(SEGM_DIR, frame_meta["video"], "classes.txt")
    class_names = load_class_names(class_path) if os.path.exists(class_path) else []

    # Populate shared state
    state["video_id"] = frame_meta["video"]
    state["current_frame_idx"] = int(frame_meta["frame_id"])
    state["bboxes"] = boxes
    state["class_names"] = class_names

    return generate_frame_id(state), class_names


# ------------------------------------------
# Frame navigation and video loading
# ------------------------------------------
def scan_videos(state, VIDEO_DIR, SEGM_DIR):
    video_files = sorted([f for f in os.listdir(VIDEO_DIR) if f.endswith(".mp4")])
    state["video_list"] = video_files
    return load_next_video(state, VIDEO_DIR, SEGM_DIR)

def load_next_video(state, VIDEO_DIR, SEGM_DIR):
    if state["current_video_idx"] >= len(state["video_list"]):
        return np.zeros((100, 100, 3), dtype=np.uint8), "No more videos to process."

    video_name = state["video_list"][state["current_video_idx"]]
    state["video_id"] = os.path.splitext(video_name)[0]
    video_path = os.path.join(VIDEO_DIR, video_name)
    label_path = os.path.join(SEGM_DIR, state["video_id"], "yolo_labels")
    class_file = os.path.join(SEGM_DIR, state["video_id"], "classes.txt")

    if not os.path.exists(label_path) or not os.path.exists(class_file):
        state["current_video_idx"] += 1
        return load_next_video(state, VIDEO_DIR, SEGM_DIR)

    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    if not frames:
        state["current_video_idx"] += 1
        return load_next_video(state, VIDEO_DIR, SEGM_DIR)

    state["frames"] = frames
    state["class_names"] = load_class_names(class_file)
    state["current_frame_idx"] = 0
    return update_frame(state, SEGM_DIR)

def update_frame(state, SEGM_DIR):
    idx = state["current_frame_idx"]
    frame = state["frames"][idx].copy()
    txt_file = os.path.join(SEGM_DIR, state["video_id"], "yolo_labels", f"{idx:05d}.txt")
    boxes = load_yolo_bbox(txt_file, frame.shape)
    state["bboxes"] = boxes
    display = draw_boxes(frame.copy(), boxes, state["class_names"])
    display = annotate_corner_title(display, state["video_id"], idx)
    return display.astype(np.uint8), f"{state['video_id']} — Frame {idx+1} / {len(state['frames'])}"

def next_frame(state, VIDEO_DIR, SEGM_DIR):
    if state["current_frame_idx"] >= len(state["frames"]):
        state["current_video_idx"] += 1
        return load_next_video(state, VIDEO_DIR, SEGM_DIR)
    return update_frame(state, SEGM_DIR)


# def load_next_valid_frame(state, VIDEO_DIR, SEGM_DIR):
#     while state["current_index"] < len(state["frames"]):
#         frame_meta = state["frames"][state["current_index"]]
#         full_video_path = os.path.join(VIDEO_DIR, frame_meta["video_path"])
#         print(f"\n▶️ Loading frame: {frame_meta['video']} — Frame {frame_meta['frame_id']}")
#
#         if not os.path.exists(full_video_path):
#             print(f"❌ Missing video: {full_video_path}")
#             state["current_index"] += 1
#             continue
#
#         cap = cv2.VideoCapture(full_video_path)
#         frame_number = int(frame_meta["frame_id"])
#         cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
#         ret, frame = cap.read()
#         cap.release()
#
#         if not ret or frame is None or frame.sum() == 0:
#             print(f"❌ Skipped unreadable or black frame at {frame_number}")
#             state["current_index"] += 1
#             continue
#
#         boxes = []
#         if frame_meta["has_yolo"]:
#             boxes = load_yolo_bbox(frame_meta["yolo_path"], frame.shape)
#
#         class_path = os.path.join(SEGM_DIR, frame_meta["video"], "classes.txt")
#         class_names = load_class_names(class_path) if os.path.exists(class_path) else []
#
#         annotated = draw_boxes(frame.copy(), boxes, class_names)
#         return frame_meta, frame, boxes, annotated
#
#     print("✅ No more valid frames to load.")
#     return None, None, None, None


def load_next_valid_frame(state, VIDEO_DIR, SEGM_DIR):
    while state["current_index"] < len(state["frames"]):
        frame_meta = state["frames"][state["current_index"]]
        full_video_path = os.path.join(VIDEO_DIR, frame_meta["video_path"])

        if not os.path.exists(full_video_path):
            print(f"❌ Missing video: {full_video_path}")
            state["current_index"] += 1
            continue

        # 🧠 Load frame from video
        cap = cv2.VideoCapture(full_video_path)
        frame_number = int(frame_meta["frame_id"])
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        cap.release()

        if not ret or frame is None or frame.sum() == 0:
            print(f"❌ Skipped unreadable or black frame at {frame_number}")
            state["current_index"] += 1
            continue

        cv2.imwrite('TEST_rotation_image1.png', frame)

        # 🧭 Get rotation metadata
        rotation = get_video_props(full_video_path)
        print(f"\n▶️ Loading frame: {frame_meta['video']} — Frame {frame_meta['frame_id']}  Rotation {rotation}")

        # 🔄 Handle rotation
        if rotation in [90, 180, 270]:
            print(f"↩️ Auto-rotating due to metadata rotation: {rotation}°")
            if rotation == 90:
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            elif rotation == 180:
                if not "_n2 (" in full_video_path:  # for Charlie_n2 (3.mp4
                    frame = cv2.rotate(frame, cv2.ROTATE_180)
            elif rotation == 270:
                if not "_n2 (" in full_video_path: # for Alpha_n2 (1).mp4
                    frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)


        # ✅ Convert to RGB after any rotation
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 🎯 Load bounding boxes
        boxes = []
        if frame_meta["has_yolo"]:
            boxes = load_yolo_bbox(frame_meta["yolo_path"], frame.shape)

        # 🏷 Load class names
        class_path = os.path.join(SEGM_DIR, frame_meta["video"], "classes.txt")
        class_names = load_class_names(class_path) if os.path.exists(class_path) else []

        # 🖍 Draw boxes and render segments
        annotated = draw_boxes(frame_rgb.copy(), boxes, class_names)

        segment_path = frame_meta.get("yolo_seg_path")
        if segment_path and os.path.exists(segment_path):
            annotated = render_yolo_segmentation_overlay(annotated, segment_path, class_path, class_names)

        # cv2.imwrite('TEST_rotation_image2.png', frame)
        # cv2.imwrite('TEST_rotation_image3.png', annotated)

        return frame_meta, frame, boxes, annotated

    print("✅ No more valid frames to load.")
    return None, None, None, None



import os

def match_segment_folders_to_video_files(segment_dir, video_dir):
    """
    Match each folder under gui_03_video_segm_pod to its corresponding video in RAW_split2.
    The match is based on exact folder name == video file name (minus .mp4).
    """

    segment_folders = sorted(
        f for f in os.listdir(segment_dir) if os.path.isdir(os.path.join(segment_dir, f))
    )
    video_files = {
        os.path.splitext(v)[0]: v
        for v in os.listdir(video_dir)
        if v.endswith(".mp4")
    }

    matched = {}
    unmatched_folders = []
    unmatched_videos = set(video_files.keys())

    for folder in segment_folders:
        if folder in video_files:
            matched[folder] = video_files[folder]
            unmatched_videos.discard(folder)
        else:
            unmatched_folders.append(folder)

    missing_folders = sorted(unmatched_videos)

    print("\n✅ Matched Segment Folders to Videos:")
    for folder, video in matched.items():
        print(f"[{folder}] ←→ {video}")

    print("\n❌ Unmatched Segment Folders (no corresponding .mp4):")
    for f in missing_folders:
        print(f"- {f}")

    print("\n🎥 Unmatched Videos (no corresponding segment folder):")
    for v in sorted(unmatched_videos):
        print(f"- {v}")

    return matched, unmatched_folders, sorted(unmatched_videos)




def match_segment_folders_to_video_files(segment_dir, video_dir):
    video_files = {
        os.path.splitext(v)[0]: v
        for v in os.listdir(video_dir)
        if v.endswith(".mp4")
    }

    matched = {}
    for folder in os.listdir(segment_dir):
        folder_path = os.path.join(segment_dir, folder)
        if not os.path.isdir(folder_path):
            continue
        if folder in video_files:
            matched[folder] = video_files[folder]
    return matched

def collect_frames_with_yolo_and_stats(gui_folder="gui_03_video_segm_pod", video_dir="tuisku_iaCarry/RAW_split2"):
    matched_folders = match_segment_folders_to_video_files(gui_folder, video_dir)

    all_frames = []
    stats = {
        "total_videos": 0,
        "total_frames": 0,
        "frames_with_yolo": 0,
        "frames_missing_yolo": 0,
        "frames_lost": 0,
        "per_video": defaultdict(lambda: {
            "total_detected": 0,
            "with_yolo": 0,
            "missing_yolo": 0,
            "lost": 0,
            "loss_percent": 0.0,
            "quarters": [
                {"total": 0, "with_yolo": 0, "missing_yolo": 0, "lost": 0} for _ in range(4)
            ]
        })
    }

    for video_dirname in sorted(matched_folders.keys()):
        base_path = os.path.join(gui_folder, video_dirname)
        yolo_path = os.path.join(base_path, "yolo_labels")
        seg_path = os.path.join(base_path, "yolo_segments")
        mask_path = os.path.join(base_path, "masks")
        video_file_path = matched_folders[video_dirname]

        if not os.path.isdir(base_path):
            print("DOES NOT EXIST:", base_path)
            continue

        mask_ids = set()
        yolo_ids = set()

        if os.path.isdir(mask_path):
            mask_ids = {f.split(".")[0] for f in os.listdir(mask_path) if f.endswith(".npy")}
        if os.path.isdir(yolo_path):
            yolo_ids = {f.split(".")[0] for f in os.listdir(yolo_path) if f.endswith(".txt")}

        frame_ids = sorted(mask_ids.union(yolo_ids))
        if not frame_ids:
            continue

        stats["total_videos"] += 1
        stats["total_frames"] += len(frame_ids)
        stats["per_video"][video_dirname]["total_detected"] = len(frame_ids)

        for i, frame_id in enumerate(frame_ids):
            has_yolo = frame_id in yolo_ids
            has_mask = frame_id in mask_ids

            quarter_index = (i * 4) // len(frame_ids)
            quarter_index = min(quarter_index, 3)

            frame_entry = {
                "video": video_dirname,
                "frame_id": frame_id,
                "has_yolo": has_yolo,
                "has_mask": has_mask,
                "quarter": quarter_index,
                "video_path": video_file_path,
                "yolo_path": os.path.join(yolo_path, f"{frame_id}.txt") if has_yolo else None,
                "yolo_seg_path": os.path.join(seg_path, f"{frame_id}.txt") if os.path.exists(os.path.join(seg_path, f"{frame_id}.txt")) else None
            }

            all_frames.append(frame_entry)

            q = stats["per_video"][video_dirname]["quarters"][quarter_index]
            q["total"] += 1

            if has_yolo:
                stats["frames_with_yolo"] += 1
                stats["per_video"][video_dirname]["with_yolo"] += 1
                q["with_yolo"] += 1
            else:
                stats["frames_missing_yolo"] += 1
                stats["per_video"][video_dirname]["missing_yolo"] += 1
                q["missing_yolo"] += 1

            if has_mask and not has_yolo:
                stats["frames_lost"] += 1
                stats["per_video"][video_dirname]["lost"] += 1
                q["lost"] += 1

        pv = stats["per_video"][video_dirname]
        if pv["total_detected"]:
            pv["loss_percent"] = round(100 * pv["lost"] / pv["total_detected"], 2)

    # 🖨️ REPORT SUMMARY
    print("=== FRAME REVIEW SUMMARY ===")
    print(f"Total videos: {stats['total_videos']}")
    print(f"Total frames: {stats['total_frames']}")
    print(f"YOLO labels available: {stats['frames_with_yolo']}")
    print(f"YOLO labels missing: {stats['frames_missing_yolo']}")
    print(f"Total lost frames (mask exists, YOLO missing): {stats['frames_lost']}")

    print("\n--- PER VIDEO ---")
    for vid, s in stats["per_video"].items():
        print(f"[{vid}] total={s['total_detected']}  ✅ YOLO={s['with_yolo']}  ❌ missing={s['missing_yolo']}  🔥 lost={s['lost']}  📉 loss={s['loss_percent']}%")
        for i, q in enumerate(s["quarters"]):
            qloss = round(100 * q["lost"] / q["total"], 2) if q["total"] else 0
            print(f"  Q{i+1}: total={q['total']} ✅={q['with_yolo']} ❌={q['missing_yolo']} 🔥={q['lost']} → loss {qloss}%")

    return all_frames, stats





def filter_yolo_center_frames(frames, window_size=5):
    """
    Filters a list of frame metadata to return only the center frame of every
    non-overlapping N-frame sequence where all frames have 'has_yolo' == True.

    Args:
        frames (list): A list of dicts, each representing a frame with keys like 'video', 'frame_id', 'has_yolo'.
        window_size (int): The number of consecutive frames to check (default: 5).

    Returns:
        list: A filtered list containing only the center frame of each valid N-frame YOLO-positive sequence.
    """
    from collections import defaultdict

    # Group frames by video
    grouped = defaultdict(list)
    for f in frames:
        grouped[f["video"]].append(f)

    result = []

    half = window_size // 2  # Index of the center frame in the window

    for video, f_list in grouped.items():
        # Sort frames by numeric value of frame_id
        sorted_f = sorted(f_list, key=lambda x: int(x["frame_id"]))
        i = 0

        while i <= len(sorted_f) - window_size:
            window = sorted_f[i:i + window_size]

            # If all frames in the window have YOLO boxes
            if all(f["has_yolo"] for f in window):
                result.append(window[half])  # Add only the center frame
                i += window_size  # Skip past the window to avoid overlaps
            else:
                i += 1  # Slide forward by one frame if not all are valid

    return result


# =======================
# Frame Cache I/O
# =======================
CACHE_PATH = "gui_04_bbox_clean/.frame_cache.json"

def load_frame_cache():
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, "r") as f:
            return json.load(f)
    return {}

def update_frame_cache(video, frame_id, action):
    cache = load_frame_cache()
    cache[f"{video}_{frame_id}"] = action
    with open(CACHE_PATH, "w") as f:
        json.dump(cache, f, indent=2)

def is_frame_cached(video, frame_id):
    cache = load_frame_cache()
    return f"{video}_{frame_id}" in cache


def count_cache_stats():
    cache = load_frame_cache()
    total = len(cache)
    accepted = sum(1 for v in cache.values() if v == "accept")
    discarded = sum(1 for v in cache.values() if v == "discard")
    return accepted, discarded, total
