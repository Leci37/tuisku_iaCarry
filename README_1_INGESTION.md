# 1 — Data ingestion

From a raw video of a product to a dataset a trainer can read. This is the
longest phase in the project and the one with the most hand work in it.

There are **two ingestion tracks in this repository**, built years apart, and they
share no files:

| | Track A — `ingestion/synthetic/` | Track B — `ingestion/video/` |
|---|---|---|
| Idea | Photograph products alone, **compose** fake carts from cut-outs | Film **real** carts, segment the products out of the video |
| Labels come from | The compositor — it knows where it pasted each item | A human clicking once per product, then mask tracking |
| Annotation format | COCO → TFRecord | YOLO `.txt` (+ COCO alongside) |
| Feeds | `README_2_TRAINING.md` track A (TF2 Object Detection API) | `README_2_TRAINING.md` track B (YOLOv8) |
| Cloud dependency | **Azure Custom Vision** (upload, tag, download) | None |
| Status | Produced the model the server runs today | Newer, better labels, **output not used by anything yet** |

Read the track you are actually working in. Track A is what is in production;
track B is where the recent effort went.

---

## Track A — synthetic composition (`GT_*`)

The premise: photographing a product on a turntable is cheap, photographing a
thousand different *carts* is not. So film each product alone, cut it out, and
paste products onto backgrounds in random arrangements. The compositor knows the
paste coordinates, so **the bounding boxes are free and pixel-exact** — no human
ever draws a box in this track.

### A1 · Frames out of the product videos

| File | In | Out |
|---|---|---|
| `ingestion/synthetic/01_video_to_frames.py` | one video per product | one folder of frames per product |
| `ingestion/synthetic/01b_rename_frames.py` | those frames | renamed to the product key |

Keeps **1 frame in every 10** (`COUNT_EACH_FRAME_TAKE_IMAG = 10`). Consecutive
video frames are nearly identical, so keeping all of them inflates the dataset
without adding information.

### A2 · Cut the product out of its background

| File | Does |
|---|---|
| `ingestion/synthetic/02_remove_bg_frames.py` | `rembg` → RGBA with the background transparent |
| `ingestion/synthetic/02b_remove_bg_white.py` | Same, tuned for the white-backdrop shots |

Both then call `reduce_size_for_transparency_size_png()` (in `common/image_utils.py`) to
**crop to the visible pixels** and record the before/after size into a DataFrame.
That size table matters later — the compositor uses it to scale products
plausibly relative to each other.

Output convention: `…\Blanco_rb\` (background removed) and `…\Blanco_cut\`
(cropped tight).

### A3 · Normalise

`ingestion/synthetic/03_resize_rotate.py` — resize to the `640×640` working frame
(`IMAGE_WITHD_BG` / `IMAGE_HIGNT_BG`) and apply random rotations, so the model
does not learn that every tin is perfectly upright. Writes a CSV of the resulting
geometry per image.

### A4 · Compose synthetic carts — **this is where labels are born**

| File | Variant |
|---|---|
| `ingestion/synthetic/04_compose_from_frames.py` | Products onto random backgrounds |
| `ingestion/synthetic/04b_compose_from_white.py` | The main one — white-backdrop cut-outs onto real scenes |
| `ingestion/synthetic/04c_compose_from_front.py` | Front-facing arrangement |
| `legacy/compose_front_1dim.py` | Marked for deletion by its own filename, still committed |

Helpers:
- `common/image_utils.py` — overlay, crop, scale, `avoid_overflow_photo_x_y_w_h()` (keeps a
  pasted product inside the frame), `get_scalated_00_x_y_w_h()` /
  `get_scalated_00_xmin_ymin_xmax_ymax()` (pixel ↔ normalised box conversion).
- `common/perspective.py` — perspective and scale warps, so a
  pasted product looks photographed rather than stuck on.

Seeded (`random.seed(123)`, `np.random.seed(123)`), so a run is reproducible.

### A5 · Azure Custom Vision round trip

| File | Direction |
|---|---|
| `ingestion/synthetic/05_azure_upload.py` | ↑ Real tagged photos |
| `legacy/carve_test_split.py` | ↑ Variant |
| `ingestion/synthetic/05b_azure_upload_augmented.py` | ↑ Synthetic/augmented scenes |
| `ingestion/synthetic/05c_azure_upload_white.py` | ↑ Mixed-background scenes |
| `ingestion/synthetic/06_azure_download_coco.py` | ↓ Images + labels back, **as COCO** |

Azure is used here as a **dataset store and review UI**, not as the trainer —
the boxes were already known before upload. The round trip exists so a human can
inspect and correct the set in the Custom Vision web interface.

> 🔴 The Azure **training key is hardcoded in plaintext** in
> `ingestion/synthetic/05b_azure_upload_augmented.py:14` and `ingestion/synthetic/06_azure_download_coco.py:22`,
> and the repository is public. Rotate that key and move it to an environment
> variable before anything else on this list.

### A6 · Split and convert

| File | Does |
|---|---|
| `ingestion/synthetic/07_coco_split.py` | COCO → train / val / test |
| `ingestion/synthetic/08_coco_to_tfrecord.bat` | COCO → TFRecord |
| `ingestion/synthetic/08b_coco_to_tfrecord_from_folder.bat` | Same, from an image folder |
| `ingestion/synthetic/08c_tfrecord_preview.bat` | Render a TFRecord back to images to eyeball it |
| `ingestion/synthetic/README_tfrecord.md` | The notes for the above |

**Track A ends with TFRecord shards + `label_map.pbtxt`.** That is the input to
training track A.

---

## Track B — real video + segmentation (`y_*`)

The premise: synthetic carts never quite look like real ones — occlusion,
lighting and stacking are wrong. So film real carts and get the labels out of the
footage. The cost is human time, and the whole track is designed to minimise it:
**a person clicks each product once, on one frame, and the mask is tracked
through the rest of the video.**

### B1 · Split the raw footage

`ingestion/video/01_split_videos.py` (and `ingestion/video/01b_trim_videos.py`)

```
RAW/  ──►  RAW_split2/
```

- `find_optimal_split_duration()` picks a segment length between 14 s and
  `MAX_DURATION_SEC = 30` that **minimises the leftover tail** across all videos,
  so no clip ends in a two-second stub.
- `RENAME_OUTPUT = True` renames clips onto the NATO alphabet (Alpha, Bravo,
  Charlie…), which is what makes them referable out loud and greppable later.
- `ENABLE_SPLIT = False` by default — as committed it **copies and renames
  without splitting**. Set it True if you want the segmentation.

### B2 · Tag by hand — one click per product · **the only manual step**

`ingestion/video/02_label_gui.py` + `ingestion/video/label_gui_utils.py` — a **Gradio** app.

Flow per video:
1. `load_first_frame()` pulls frame 0 of the clip.
2. The operator picks a label from the legend (`build_legend_html()`, colours and
   names read from `label_map.pbtxt`) and **clicks on the product in the image**.
3. `segment_objects()` (`ingestion/video/sam_wrapper.py`, SAM) turns those clicks into a
   mask. `apply_mask_postprocessing()` drops blobs under `min_area = 500` px.
4. `draw_bboxes_and_labels()` / `draw_points_on_image()` show the result live.
5. `validate_masks_before_save()` refuses to save if a mask is empty or the
   output folder is not writable — **this is the first validation gate**.
6. `save_masks_and_image()` writes, per clip, into `gui_output/<video>/`:

```
gui_output/<video>/
  <base>_<label>.npy          one boolean mask per product
  <base>_<label>.png          optional visual of each mask
  <base>_segmented.png        all masks overlaid
  <base>_bbox_overlay.png     boxes + labels, for review
  tagging_metadata.json       {labels: {name: {mask_file, …}}}  ← the handoff
```

`next_video()` / `prev_video()` walk the folder so an operator can sit and work
through a batch.

### B3 · Propagate the masks through the video

`ingestion/video/03_track_masks.py` + `ingestion/video/track_masks_utils.py`

Reads `tagging_metadata.json`, loads the clip, and calls
`track_with_mask_refinement()` (`ingestion/video/sam_wrapper.py`) to carry each frame-0
mask forward across every frame. One click becomes hundreds of labelled frames.

Per frame it then:
- `clean_and_filter_mask()` — **the second validation gate.** Drops masks that
  are too small, that drifted, or that broke apart; a frame failing the check is
  discarded entirely rather than half-labelled.
- `mask_to_polygon()` — mask → polygon, for the segmentation labels.
- Writes YOLO **bbox** and YOLO **segmentation** `.txt` per frame (`00000.txt`, …).
- Appends to a COCO structure (`init_coco_structure()`).
- `save_qc_overlay()` every 60th frame, plus a `_FRAME_DISC_<reason>.png` every
  40th discarded frame — so rejects are auditable, not silent.

Outputs per clip:

```
<output_dir>/
  yolo_bbox/00000.txt …        YOLO boxes
  yolo_seg/00000.txt …         YOLO polygons
  coco_annotations.json
  frames_metadata.json
  classes.txt                  copied from the global one
  segmented_video.mp4          masks painted on, for review
  segmented_video_with_bbox.mp4
  qc/                          spot-check overlays + discard reasons
```

`write_global_yolo_classes()` derives `classes.txt` from `label_map.pbtxt` once,
at the top level, so every clip shares one class ordering. **Getting this wrong
silently relabels the whole dataset**, which is why it is written centrally.

> ⚠️ **Folder-name break.** `03_track_masks.py` writes to
> `base_output = "gui_video_segmen"`, but `04_bbox_review.py` reads
> `SEGM_DIR = "gui_03_video_segm_pod"`. Nothing renames it —
> you have to do it by hand between the two steps, and nothing says so.

### B4 · Review the boxes by hand — **the validation step**

`ingestion/video/04_bbox_review.py` + `ingestion/video/bbox_review_utils.py` — a second Gradio app.

- `collect_frames_with_yolo_and_stats()` gathers every frame that has labels.
- `filter_yolo_center_frames(window_size=5)` keeps the **middle** frame of each
  run of five — adjacent tracked frames are near-duplicates, so this thins the
  set before a human ever looks at it.
- The operator accepts, fixes (`correct_box()`) or rejects each frame.
  `draw_key_legend()` shows the keyboard map; `PRODUCTS_CODES_NAMES` and
  `HEX_COLOR_MAP` give each class a stable name and colour.
- `.frame_cache.json` (`load_frame_cache` / `update_frame_cache` / `is_frame_cached`)
  records what has already been judged, so the review can be stopped and resumed
  — which matters when the set runs to thousands of frames.
- `save_cleaned_data()` writes the accepted frame, its YOLO label, **and** appends
  to `coco_annotations.json`.

```
gui_04_bbox_clean/
  frames/          accepted images
  yolo_labels/     accepted labels
  coco_annotations.json
  .frame_cache.json
```

### B5 · Orient, count, balance

| File | Out | Does |
|---|---|---|
| `ingestion/video/05_rotate_and_stats.py` | `gui_041_bbox_clean/` | Rotate 90° CW to `720×1280` (`rotate_yolo_bbox_90cw()` rotates the boxes with the pixels), write per-class counts |
| `ingestion/video/06_rotate_stats_balance.py` | `gui_042_bbox_clean/` | The same **plus `albumentations` augmentation aimed at class balance** |

`720×1280` is portrait — it matches the overhead camera's mounting, not the
model's input. Both write a `check/` folder of drawn-on samples; step 06 also
writes `check_aug/` every `CHECK_EVERY_N = 60` images. `print_summary()`
(`ingestion/video/bbox_tools.py`) reports total labels, per-class counts, how often a class
appears **alone** (`only_label_occurrence`), and a histogram of classes-per-image.
That last one is the number to watch: a set where most images hold one product
teaches the model nothing about occlusion.

Helpers, shared across track B: `ingestion/video/bbox_tools.py` (box rotation,
drawing, class names, `print_summary()`), `ingestion/video/label_tools.py`
(`parse_label_map()`, `overlay_mask()`, `build_legend_html()` — the legend the
tagging GUI shows), `ingestion/video/sam_wrapper.py` (SAM init, masking,
tracking) and `ingestion/video/augment_tools.py` (`build_aug_combinations()`,
`filter_occluded_boxes()`).

**Track B ends at `gui_042_bbox_clean/{frames,yolo_labels}`.** That is the input
to training track B.

### Dead and duplicated in track B

All of it now lives in `legacy/` — see `legacy/README.md`. The one worth knowing
about: `legacy/track_anything/` was a fork of these tools where the run script was
byte-identical to the live copy but the model wrapper and utils had **diverged**,
with nothing marking which was current. That ambiguity is why it is archived
rather than merged.

---

## The full ingestion chain at a glance

```
TRACK A   video ─► frames ─► rembg cut-out ─► resize/rotate ─► COMPOSE (boxes born)
          ─► Azure up ─► Azure down (COCO) ─► split ─► TFRecord ──► training A

TRACK B   RAW ─► split/rename ─► CLICK ONCE per product (SAM)
          ─► track masks through video (auto-labels + QC)
          ─► [rename folder by hand]
          ─► human box review (accept/fix/reject, resumable)
          ─► rotate 720x1280 + class stats ─► balance aug ──────► training B
```

---

## What is missing from ingestion

1. **No dataset versioning.** Nothing records which frames went into which model.
   There is no manifest, no hash, no run id. Six months on there is no way to
   answer "what was `save_model_sig_54` trained on?" This is the single biggest
   gap in the phase.
2. **`label_map.pbtxt` is not in the repository**, yet `02_label_gui.py`,
   `03_track_masks.py` and training A all read it. Its home in the new layout is
   `common/label_map.pbtxt`. It is the file that defines the class ordering for
   everything; losing it invalidates every label set.
3. **`utils_bbox` does not exist** — imported by `06_azure_download_coco.py` and
   `07_coco_split.py`. Those two scripts cannot run from a clean clone.
   `tools/check_imports.py` reports it as a known gap rather than as breakage.
4. **`from Utils import COCO_json_format_validator` fails** — the old `Utils/`
   folder held only the server instance counter (now `serving/check_instances.py`)
   and had no `__init__.py`. The COCO validator `07_coco_split.py` calls was never
   committed.
5. **Every path is a hardcoded absolute Windows path** (`E:\iaCarry_img_eroski\…`,
   `C:\Users\leci\Documents\GitHub\…`). No config file, no CLI arguments. The
   chain runs on exactly one machine.
6. **The `gui_video_segmen` → `gui_03_video_segm_pod` rename is undocumented**
   (B3 above), and `training/yolo/01_train_yolov8.py` reads `classes.txt` from
   `gui_03_video_segm_pod/` while steps 05 and 06 read it from
   `gui_04_bbox_clean/` — two sources for the file that must not disagree.
7. **No test covers any of this.** Not frame extraction, not background removal,
   not box rotation, not the COCO/YOLO writers. A rotation bug that silently
   moved every box would be caught only by eye, in `check/`.
8. **No held-out test set in track B.** `training/yolo/01_train_yolov8.py` splits
   train/val 80/20 and stops. There is no third split kept aside, so there is no
   honest final number. `legacy/carve_test_split.py` is the only code that ever
   carved one, for track A, and its upload path is commented out.
