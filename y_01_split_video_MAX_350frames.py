import os
import cv2

# Configuration
input_dir = r"C:\Users\leci\Documents\GitHub\tuisku_iaCarry\RAW"
output_dir = r"C:\Users\leci\Documents\GitHub\tuisku_iaCarry\RAW_split2"
os.makedirs(output_dir, exist_ok=True)

# Options
RENAME_OUTPUT = True
MAX_AA_DURATION_SEC = 10

# Radiotelephony alphabet
alphabet = [
    "Alpha", "Bravo", "Charlie", "Delta", "Echo", "Fox", "Golf", "Hotel",
    "India", "Juliett", "Kilo", "Lima", "Mike", "November", "Oscar", "Papa",
    "Quebec", "Romeo", "Sierra", "Tango", "Uniform", "Victor", "Whiskey",
    "Xray", "Yankee", "Zulu"
]

def trim_video(path, output_path, max_duration_sec):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames_to_write = int(min(max_duration_sec, total_frames / fps) * fps)

    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    written = 0
    while written < frames_to_write:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
        written += 1

    cap.release()
    out.release()
    print(f"[TRIMMED] {os.path.basename(output_path)} → {frames_to_write} frames (~{frames_to_write/fps:.2f}s)")

def copy_video(path, output_path):
    if not os.path.exists(output_path):
        os.system(f'copy "{path}" "{output_path}"')  # Windows copy
        print(f"[COPY] {os.path.basename(path)} → {os.path.basename(output_path)}")
    else:
        print(f"[SKIP] Already exists: {os.path.basename(output_path)}")

# Process videos
print("[INFO] Scanning input directory for .mp4 videos...")
video_files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(".mp4")])

for idx, fname in enumerate(video_files):
    input_path = os.path.join(input_dir, fname)
    base_name = os.path.splitext(fname)[0]
    alphabet_prefix = alphabet[idx % len(alphabet)]
    output_name = f"{alphabet_prefix}_{base_name}.mp4" if RENAME_OUTPUT else f"{base_name}.mp4"
    output_path = os.path.join(output_dir, output_name)

    if "_aa" not in output_name.lower():
        print(f"[SKIP] Output name does not contain '_aa': {output_name}")
        continue

    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = total_frames / fps
    cap.release()

    if duration > MAX_AA_DURATION_SEC:
        print(f"[TRIM] {output_name} is {duration:.2f}s. Trimming to {MAX_AA_DURATION_SEC}s.")
        trim_video(input_path, output_path, MAX_AA_DURATION_SEC)
    else:
        print(f"[COPY] {output_name} is already <= {MAX_AA_DURATION_SEC}s")
        copy_video(input_path, output_path)

print("\n[✓] All matching videos processed.")
