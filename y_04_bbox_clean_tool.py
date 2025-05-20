import gradio as gr
import os
import cv2
import json
import random
random.seed(42)  # Set a fixed seed for consistent shuffling
import numpy as np
from y_040_bbox_clean_tool_Utils import (
    generate_frame_id,
    save_cleaned_data,
    collect_frames_with_yolo_and_stats,
    load_yolo_bbox,
    prepare_frame_state,
    load_next_valid_frame, filter_yolo_center_frames, load_frame_cache, update_frame_cache, is_frame_cached,
    count_cache_stats
)

VIDEO_DIR = "C:/Users/leci/Documents/GitHub/tuisku_iaCarry/RAW_split2"
SEGM_DIR = "gui_03_video_segm_pod"
SAVE_DIR = "gui_04_bbox_clean"

IMG_DIR = os.path.join(SAVE_DIR, "frames")
LABEL_DIR = os.path.join(SAVE_DIR, "yolo_labels")
os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(LABEL_DIR, exist_ok=True)


# Load frame list and stats
all_frames, stats = collect_frames_with_yolo_and_stats(gui_folder=SEGM_DIR, video_dir=VIDEO_DIR)
filtered_frames = filter_yolo_center_frames(all_frames)


from collections import defaultdict
import itertools

# def interleave_unique_videos(frames):
#     # Group frames by video
#     video_groups = defaultdict(list)
#     for f in frames:
#         video_groups[f["video"]].append(f)
#
#     # Sort each group by frame index
#     for video in video_groups:
#         video_groups[video] = sorted(video_groups[video], key=lambda f: int(f["frame_id"]))
#
#     # Interleave one frame from each video at a time
#     interleaved = list(itertools.chain.from_iterable(itertools.zip_longest(*video_groups.values())))
#     return [f for f in interleaved if f is not None]

def interleave_unique_videos(frames):
    from collections import defaultdict
    import itertools

    # Group frames by video name
    video_groups = defaultdict(list)
    for f in frames:
        video_groups[f["video"]].append(f)

    # Sort video names alphabetically
    sorted_video_names = sorted(video_groups.keys())

    # Sort frames within each video numerically by frame_id
    for video in sorted_video_names:
        video_groups[video] = sorted(video_groups[video], key=lambda f: int(f["frame_id"]))

    # Interleave: one frame from each video per round
    ordered_groups = [video_groups[v] for v in sorted_video_names]
    interleaved = list(itertools.chain.from_iterable(itertools.zip_longest(*ordered_groups)))
    return [f for f in interleaved if f is not None]

# # Apply interleaving instead of shuffle
# def get_frame_key(f):  # unique ID per frame
#     return f"{f['video']}_{f['frame_id']}"

filtered_frames = filter_yolo_center_frames(all_frames)
interleaved_frames = [
    f for f in interleave_unique_videos(filtered_frames)
    if not is_frame_cached(f["video"], f["frame_id"])
]
#LUIS
interleaved_frames = sorted(
    [
        f for f in interleave_unique_videos(filtered_frames)
        if not is_frame_cached(f["video"], f["frame_id"])
    ],
    key=lambda f: (f["video"].lower(), int(f["frame_id"]))
)
# interleaved_frames = interleave_unique_videos(filtered_frames)

state = {
    "current_index": 0,
    "frames": interleaved_frames,
    "history": []  # stack of (frame_meta, frame_id)
}

def undo_last_action():
    print("--- Undoing last action ---")
    if not state["history"]:
        return np.zeros((100, 100, 3), dtype=np.uint8), "", "⚠️ No previous action to undo"

    frame_meta, frame_id = state["history"].pop()
    video = frame_meta["video"]
    frame_num = frame_meta["frame_id"]

    # Delete saved files
    img_path = os.path.join(IMG_DIR, f"{frame_id}.png")
    label_path = os.path.join(LABEL_DIR, f"{frame_id}.txt")
    for p in [img_path, label_path]:
        if os.path.exists(p):
            os.remove(p)
            print(f"🗑️ Deleted: {p}")

    # Remove from cache
    cache = load_frame_cache()
    key = f"{video}_{frame_num}"
    if key in cache:
        del cache[key]
        with open("gui_04_bbox_clean/.frame_cache.json", "w") as f:
            json.dump(cache, f, indent=2)
        print(f"🧹 Removed cache entry: {key}")

    # Re-insert frame at previous position
    state["frames"].insert(state["current_index"], frame_meta)

    # Load frame again
    return discard_frame()  # reloads current_index

def accept_frame():
    print("--- Accepting frame ---")
    frame_meta, frame, boxes, annotated = load_next_valid_frame(state, VIDEO_DIR, SEGM_DIR)
    if frame is None:
        return np.zeros((100, 100, 3), dtype=np.uint8), "✅ All frames processed", ""

    frame_id, info = prepare_frame_state(frame_meta, frame, boxes, SEGM_DIR, state)
    print(f"✅ Saved frame: {frame_id}")
    save_cleaned_data(frame_id, frame, boxes, IMG_DIR, LABEL_DIR)
    update_frame_cache(frame_meta["video"], frame_meta["frame_id"], "accept")
    state["current_index"] += 1

    top_info = f"🎞 Video: {frame_meta['video']}   🖼 Frame: {frame_meta['frame_id']}"
    bottom_info = f"[SAVED] {frame_id}\n{info}"

    update_frame_cache(frame_meta["video"], frame_meta["frame_id"], "accept")
    accepted, discarded, total = count_cache_stats()
    summary = f"✅ Accepted: {accepted}  ⛔ Discarded: {discarded}  🎯 Remaining: {len(state['frames']) - state['current_index']}"
    state["history"].append((frame_meta, frame_id))
    return annotated, top_info, f"{bottom_info}\n\n{summary}"


def discard_frame():
    print("--- Discarding frame ---")
    state["current_index"] += 1
    frame_meta, frame, boxes, annotated = load_next_valid_frame(state, VIDEO_DIR, SEGM_DIR)
    if frame_meta is None:
        return np.zeros((100, 100, 3), dtype=np.uint8), "", "✅ All frames discarded or processed"
    cv2.imwrite('TEST_rotation_image1.png', annotated)

    frame_id, info = prepare_frame_state(frame_meta, frame, boxes, SEGM_DIR, state)
    print(f"⏭️ Discarded — moved to next: {frame_id}")
    top_info = f"🎞 Video: {frame_meta['video']}   🖼 Frame: {frame_meta['frame_id']}"
    bottom_info = f"[DISCARDED → NEXT] {frame_id}\n{info}"

    update_frame_cache(frame_meta["video"], frame_meta["frame_id"], "discard")
    accepted, discarded, total = count_cache_stats()
    summary = f"✅ Accepted: {accepted}  ⛔ Discarded: {discarded}  🎯 Remaining: {len(state['frames']) - state['current_index']}"
    state["history"].append((frame_meta, frame_id))
    return annotated, top_info, f"{bottom_info}\n\n{summary}"

with gr.Blocks() as demo:
    gr.Markdown("### 🧹 Auto Frame Validator — Bounding Box QC")

    # Frame info above image
    frame_info_top = gr.Textbox(label="🎬 Video / Frame", lines=1)

    # Annotated image (full width)
    frame_display = gr.Image(type="numpy", label="Annotated Frame", tool=None).style(height=680)
    with gr.Row():
        undo_btn = gr.Button("↩️ Undo")
        discard_btn = gr.Button("⛔ Discard")
        accept_btn = gr.Button("✅ Accept")
    # Frame info below image
    frame_info_bottom = gr.Textbox(label="📝 Frame Info", lines=4)

    undo_btn.click(fn=undo_last_action, outputs=[frame_display, frame_info_top, frame_info_bottom])
    accept_btn.click(fn=accept_frame, outputs=[frame_display, frame_info_top, frame_info_bottom])
    discard_btn.click(fn=discard_frame, outputs=[frame_display, frame_info_top, frame_info_bottom])
    demo.load(fn=accept_frame, outputs=[frame_display, frame_info_top, frame_info_bottom])

    gr.HTML("""
    <script>
    document.addEventListener("keydown", function(e) {
        const buttons = Array.from(document.querySelectorAll("button"));
        if (e.key === "ArrowRight") {
            const acceptBtn = buttons.find(b => b.innerText.includes("Accept"));
            if (acceptBtn) acceptBtn.click();
        }
        if (e.key === "ArrowLeft") {
            const discardBtn = buttons.find(b => b.innerText.includes("Discard"));
            if (discardBtn) discardBtn.click();
        }
    });
    </script>
    """)

demo.launch(debug=True)
