# 2 — Training, creation, evaluation, export

From a labelled dataset to a model file the server can load.

As in ingestion, there are **two tracks**. Track A produced the model in production;
track B is newer and its output is not wired to anything.

| | Track A | Track B |
|---|---|---|
| Framework | TensorFlow 2 + Object Detection API | Ultralytics YOLOv8 (PyTorch) |
| Base model | SSD MobileNet V2 FPNLite 640, EfficientDet D1/D2 | `yolov8n.pt` |
| Input | TFRecord + `label_map.pbtxt` | `gui_042_bbox_clean/` + `classes.txt` |
| Output | SavedModel with a `detect` signature | `gui_05_model/…/best.pt` |
| Consumed by | **`README_3_SERVER.md`** | nothing |

---

## Track A — TF2 Object Detection API

### A0 · What must exist first

- The **TF Object Detection API** installed (`object_detection.*` is imported
  everywhere). It is not vendored here and not in `requirements.txt`.
- A **base checkpoint** downloaded from the TF2 Detection Zoo, unpacked as
  `./ssd_mobilenet_v2_fpnlite_640x640/` containing `pipeline.config` and
  `checkpoint/`.
- `label_map_eroski.pbtxt` beside it.

### A1 · Warm-up: the 3-class model

`training/tensorflow/01_train_medium.py` — the smallest thing that exercises the whole path.
`num_classes = 3`, ground-truth boxes typed **literally into the source**
(`gt_boxes`, `gt_labels`), base config
`./ssd_mobilenet_v2_fpnlite_640x640/pipeline.config`.

Its point is not the model — it is proving the API, the checkpoint restore and
the fine-tuning loop work before spending hours on the real set. `training/tensorflow/02_eval_medium.py`
scores it and `training/tensorflow/medium_utils.py` holds the shared pieces.

### A2 · The real training run

`training/tensorflow/03_train_eroski.py`

```
LABEL_MAP_PATH  ./ssd_mobilenet_v2_fpnlite_640x640/label_map_eroski.pbtxt
NUM_CLASSES     derived from the label map — never hardcoded
PATH_BBOX       E:\iaCarry_img_eroski\df_size_Rota_img.csv   ← boxes come from a CSV
NUM_ELE_PER_CLASS = 50
pipeline_config ./ssd_mobilenet_v2_fpnlite_640x640/pipeline.config
model_dir       ./ssd_mobilenet_v2_fpnlite_640x640/checkpoint
```

Sequence: read the label map → build `category_index` → load the box CSV written
back in ingestion A3/A4 → assemble `Train_image_filenames`, `Gt_boxes`, `Gt_labels`
→ `model_builder.build(is_training=True)` → restore the base checkpoint via
`tf.compat.v2.train.Checkpoint` → fine-tune.

`NUM_ELE_PER_CLASS = 50` caps how many examples each class contributes, which is
the crude class-balancing lever on this track — track B does the same job
properly with augmentation in ingestion step 06.

`training/tensorflow/04_train_eroski_multi.py` is the multi-object variant. It
was `Transfer_L_Train_Eroski_[n,4].py`: the brackets made it un-importable and
awkward to type in a shell, which is one of the reasons the tree was renamed.

`training/tensorflow/train_colab.py` is the same training ported to Colab, for
when the local GPU was not enough.

### A3 · Saving — checkpoint *and* SavedModel

`common/detection_model.py :: save_detecion_pd_checkpoint()` (this file was
`Ultils_model_creation.py` — the typo was carried at every import site, and is
gone now that both `training/tensorflow/` and `training/tflite/` import it from
`common/`):

```
config_util.save_pipeline_config(...)            pipeline.config beside the weights
CheckpointManager(... max_to_keep=1)             <MODEL>/checkpoint/
tf.saved_model.save(...)                         <MODEL>/saved_model/
```

Other utilities in that file: `load_image_into_numpy_array()`,
`plot_detections()`, `get_category_index()`, `detect()`,
`predict_and_save_imagen()`, `get_valitation_images()`,
`generate_file_log_PB_info()` (dumps what a `.pb` actually contains).

### A4 · Evaluation

`training/tensorflow/05_eval_eroski.py`

```
MODEL_TO_LOAD    "model_efi_d2_aug"
PIPELINE_CONFIG  model_efi_d2_aug/pipeline.config
NUM_CHECK_POINT  1203          ← edited by hand per run
MIN_SCORE        0.4
LABEL_ID_OFFSET  1
```

`restore_model_from_ckp()` rebuilds the graph from `pipeline.config` and restores
`ckpt-<N>`. It carries a loud warning worth repeating: **do not include the
`.index` extension in the checkpoint path.** If you do, there is no error — the
weights simply never load, and you spend the run watching a loss that will not
come down.

It writes annotated test images to
`<MODEL>/test_imgA_<checkpoint>/img_NN.jpg`. That is the evaluation artifact:
**pictures, judged by eye.** There is no mAP, no PR curve, no confusion matrix
anywhere in this repository.

### A5 · The `detect` signature — the step that connects training to the server

The server does **not** load a checkpoint. It loads a SavedModel and asks for a
named signature:

```python
# serving/detector.py
SIGNATURE_REF = "detect"
PATH_TO_SAVED_MODEL_INTERFACE_GRAPH = "model_efi_d1C/save_model_sig_54"
PATH_PICKLE_CAT_INDEX = "model_efi_d1C/P_Category_index.pickle"
```

A plain `tf.saved_model.save()` does **not** produce a `detect` signature — the
model has to be exported/frozen with `exporter_main_v2.py` from the Object
Detection API first. `serving/detector.py:48` says exactly this in a debug
line, and `exporter_main_v2.py` is not in this repository.

Inspection tools for this:

| File | Does |
|---|---|
| `training/tensorflow/06_inspect_signature_default.py` | Loads a SavedModel, prints `serving_default` structured outputs / dtypes / shapes |
| `training/tensorflow/07_inspect_signature_detect.py` | Same for the `detect` signature |
| `common/detection_signature.py` | `img_to_tensor()`, `img_proccess()`, box drawing for signature testing |

> ⚠️ Both `06_inspect_signature_*.py` still run against the **cat / dog / zombie**
> tutorial images (`list_paths` is a hardcoded list of `cat.2000.jpg` …). They
> were never repointed at iaCarry data. They are signature probes, not tests.

The class names do not come from `label_map.pbtxt` at serve time either — they
come from `P_Category_index.pickle`, a pickled `category_index` saved during
training. **The pickle and the SavedModel must be exported together**; a mismatch
renames every class silently and the front end then logs the leftovers as
`skippedUnknownTags`.

### A6 · TFLite export (edge, not used by the server)

| File | Does |
|---|---|
| `training/tflite/01_convert.py` | `TFLiteConverter.from_saved_model()` → `.tflite`, with and without a pinned `signature_keys` |
| `training/tflite/tflite_metadata.py` | Attaches metadata + label file (`mdata_write_all_in_tflite_SIMPLE` / `_FULL`, `input_norm_mean=[127.5]`) |
| `training/tflite/tflite_info.py` | Dumps input/output tensor details |
| `training/tflite/02_detect.py` | Runs a `.tflite` interpreter over a folder (`input_mean/std = 127.5`, `HUMBRAL_PREDTIC = 0.5`) |
| `training/tflite/tflite_detect_utils.py` | Interpreter helper |

`training/tflite/01_convert.py:36` prints the prerequisite it cannot perform itself:

```
export_tflite_graph_tf2.py --pipeline_config_path=model_101_C/pipeline.config
                           --trained_checkpoint_dir model_101_C/checkpoint
                           --output_directory model_101_C/frozen
```

Note there are **two different freezers** in play and they are easy to confuse:
`export_tflite_graph_tf2.py` for the TFLite path, `exporter_main_v2.py` for the
server's `detect` signature. `training/tflite/02_detect.py` also still points at the cat/dog
model.

---

## Track B — YOLOv8

`training/yolo/01_train_yolov8.py` (and `legacy/generate_aug_samples.py`,
which duplicates its dataset-prep half and stops there).

```
SRC_IMAGE_DIR  gui_042_bbox_clean/frames
SRC_LABEL_DIR  gui_042_bbox_clean/yolo_labels
CLASS_FILE     gui_03_video_segm_pod/classes.txt
OUTPUT_DIR     yolo_dataset
MODEL_OUT_DIR  gui_05_model
VAL_RATIO 0.2   SEED 42   IMG_SIZE 1280   epochs 50   base yolov8n.pt
```

Steps: intersect image and label basenames (so a frame missing either is dropped)
→ `train_test_split(test_size=0.2, random_state=42)` → copy into
`yolo_dataset/{train,val}/{images,labels}` → write `yolo_data.yaml` with an
absolute `path:` → `YOLO("yolov8n.pt").train(...)` → predict on the first five
validation images as a smoke test. `torch.cuda.is_available()` is printed at
startup, because the difference between GPU and CPU here is hours.

Two naming traps:
- The old filename said **`gui041`** while the code reads **`gui_042_bbox_clean`**.
  The augmented set is the one actually used; the misleading name is gone with
  the rename, the mismatch in the code is not.
- `CLASS_FILE` points at `gui_03_video_segm_pod/classes.txt` while ingestion
  steps 05 and 06 read `gui_04_bbox_clean/classes.txt`. If those two files ever diverge, the
  class ids in the labels stop meaning what the trainer thinks they mean.

---

## What is missing from training

1. **Track B's output goes nowhere.** It produces `gui_05_model/…/best.pt`. The
   server loads a TF SavedModel with a `detect` signature. There is no converter,
   and no YOLO inference path in `serving/`. **All the track B work is currently
   disconnected from production** — the single most consequential gap in the
   project. Either export YOLOv8 → SavedModel/ONNX and add a loader, or accept
   track B as research and say so.
2. **No model artifacts are in the repository.** `model_efi_d1C/save_model_sig_54`,
   `P_Category_index.pickle`, `model_efi_d2_aug/`, `ssd_mobilenet_v2_fpnlite_640x640/`,
   `label_map*.pbtxt` — all referenced, none present, no download script, no note
   on where they live. A clean clone cannot train or serve.
3. **No quantitative evaluation, on either track.** No mAP, no per-class PR, no
   confusion matrix committed. Track A evaluates by looking at JPEGs. Meanwhile
   `README.md` advertises **98% accuracy** and the checkout screen shows
   `98% accuracy` / `1.2s inference` — `serving/FLOW.md` §8 confirms both are
   **hardcoded constants, not measurements**. Either measure them or stop
   displaying them.
4. **`utils_transfer_learning` does not exist** — imported by
   `training/tensorflow/03_train_eroski.py:22`. That import fails from a clean clone.
5. **The export step is missing.** `exporter_main_v2.py` is the bridge from a
   trained checkpoint to the `detect` signature the server needs, and it is
   neither vendored nor scripted. It is the one step nobody can reproduce from
   this repository.
6. **Run configuration lives in edited source.** `NUM_CHECK_POINT = 1203`,
   `MODEL_TO_LOAD`, epochs, thresholds — all changed by hand between runs, with
   `#TODO cambiar esto para RUN` next to them. Nothing records what any past run
   used.
7. **No experiment tracking.** No TensorBoard export, no run log, no metrics file.
   Combined with gap 1 in `README_1_INGESTION.md` (no dataset versioning), no result
   in this project is reproducible.
8. **Toy leftovers still committed and still runnable**: `training/tensorflow/06_inspect_signature_default.py`,
   `training/tensorflow/07_inspect_signature_detect.py` and `training/tflite/02_detect.py` all reference cat/dog/zombie
   images and models.
9. **`requirements.txt` covers track A only** — no `ultralytics`, `torch`,
   `scikit-learn`, `albumentations`, `gradio`.
