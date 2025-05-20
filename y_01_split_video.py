import os
import cv2
import math

# Configuration
input_dir = r"C:\Users\leci\Documents\GitHub\tuisku_iaCarry\RAW"
output_dir = r"C:\Users\leci\Documents\GitHub\tuisku_iaCarry\RAW_split2"
os.makedirs(output_dir, exist_ok=True)

# Options
ENABLE_SPLIT = False           # Set to False to skip splitting (just copy videos)
RENAME_OUTPUT = True          # Set to False to keep original video names
MAX_DURATION_SEC = 30         # Maximum allowed duration per segment

# Radiotelephony alphabet
alphabet = [
    "Alpha", "Bravo", "Charlie", "Delta", "Echo", "Fox", "Golf", "Hotel",
    "India", "Juliett", "Kilo", "Lima", "Mike", "November", "Oscar", "Papa",
    "Quebec", "Romeo", "Sierra", "Tango", "Uniform", "Victor", "Whiskey",
    "Xray", "Yankee", "Zulu"
]

def find_optimal_split_duration(video_lengths_sec, min_sec=14, max_sec=100):
    max_sec = min(max_sec, MAX_DURATION_SEC)
    best_d = min_sec
    lowest_remainder = float('inf')
    print("[INFO] Evaluating optimal split duration:")
    for d in range(min_sec, max_sec + 1):
        remainders = [v % d for v in video_lengths_sec]
        avg_rem = sum(remainders) / len(remainders)
        print(f"  - {d}s → average leftover: {avg_rem:.2f}s")
        if avg_rem < lowest_remainder:
            lowest_remainder = avg_rem
            best_d = d
    return best_d

def split_video(path, alphabet_prefix, output_folder, segment_duration, rename=True):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / fps
    frames_per_segment = int(min(segment_duration, MAX_DURATION_SEC) * fps)

    print(f"\n[PROCESSING] {os.path.basename(path)}")
    print(f"  Duration    : {duration:.2f} sec")
    print(f"  FPS         : {fps}")
    print(f"  Resolution  : {width}x{height}")
    print(f"  Total Frames: {total_frames}")
    print(f"  Split Length: {segment_duration}s → {frames_per_segment} frames")

    frame_index = 0
    part_index = 1
    out = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_index % frames_per_segment == 0:
            if out:
                out.release()

            part_str = f"{part_index:02d}p"
            if rename:
                filename = f"{alphabet_prefix}_{part_str}_{segment_duration}s_{frames_per_segment}f.mp4"
            else:
                base_name = os.path.splitext(os.path.basename(path))[0]
                filename = f"{base_name}_{part_str}.mp4"

            segment_path = os.path.join(output_folder, filename)
            out = cv2.VideoWriter(segment_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
            print(f"    [NEW SEGMENT] {filename}")
            part_index += 1

        out.write(frame)
        frame_index += 1

    if out:
        out.release()
    cap.release()
    print(f"  → Done. {part_index - 1} segments created.")

def copy_video(path, output_folder, rename=True, prefix=None):
    base_name = os.path.splitext(os.path.basename(path))[0]
    new_name = f"{prefix}_{base_name}.mp4" if rename and prefix else f"{base_name}.mp4"
    dst_path = os.path.join(output_folder, new_name)
    if not os.path.exists(dst_path):
        os.system(f'copy "{path}" "{dst_path}"')  # Windows copy
        print(f"[COPY] {base_name}.mp4 → {new_name}")
    else:
        print(f"[SKIP] Already exists: {new_name}")

# Step 1: Gather video info
video_lengths = []
video_paths = []

print("[INFO] Scanning input directory for .mp4 videos...")
list_files = sorted(os.listdir(input_dir)) # sorted(os.listdir(input_dir))
for fname in list_files:
    if fname.lower().endswith(".mp4"):
        fpath = os.path.join(input_dir, fname)
        cap = cv2.VideoCapture(fpath)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        duration = frames / fps
        video_lengths.append(duration)
        video_paths.append(fpath)
        print(f"  - Found: {fname} → {duration:.2f}s, {int(frames)} frames, {fps:.2f} FPS")
        cap.release()

# Step 2: Determine best split length
if ENABLE_SPLIT:
    optimal_duration = find_optimal_split_duration(video_lengths)
    print(f"\n[INFO] Selected optimal split duration: {optimal_duration} seconds")

# Step 3: Process each video using its own alphabet prefix
for i, path in enumerate(video_paths):
    alphabet_prefix = alphabet[i % len(alphabet)]

    if ENABLE_SPLIT:
        split_video(path, alphabet_prefix, output_dir, optimal_duration, rename=RENAME_OUTPUT)
    else:
        copy_video(path, output_dir, rename=RENAME_OUTPUT, prefix=alphabet_prefix if RENAME_OUTPUT else None)

print("\n[✓] All videos processed successfully.")
