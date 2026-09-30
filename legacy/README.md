# legacy — kept for reference, not for running

Nothing here is part of a live pipeline. It is kept because deleting it would
lose context that the git history alone does not make obvious, and because two
of these files are the only place a particular idea was ever written down.

**Their imports are deliberately not updated.** They still refer to `GT_Utils`,
`your_utils` and the other pre-reorganisation names, which is why
`tools/check_imports.py` skips this folder. Reviving anything here means fixing
its imports first — treat that as the signal that you are reviving it.

| File | Was | Why it is here |
|---|---|---|
| `track_anything/` | `Track/` | An older fork of the track B tooling. `run_manual.py` was byte-identical to the root copy; `wrapper.py` and its utils had **diverged**, with nothing marking which was current. That ambiguity is the reason it is archived rather than merged. |
| `track_anything/wrapper.py` | `TrackAnythingWrapper.py` | The Track-Anything integration, superseded by `ingestion/video/sam_wrapper.py`. |
| `label_gui_old_video.py` | `y_02_label_gui_OLD_video.py` | Superseded by `ingestion/video/02_label_gui.py`. Named OLD by its author. |
| `run_manual.py` | `your_RUN.py` | A manual driver using `easygui` file pickers, from before the Gradio GUIs existed. |
| `utils_old.py` | `utils_.py` | Orphan helpers; nothing imported it. |
| `compose_front_1dim.py` | `GT_03_add_bg_FRONT_1dim_DELETE.py` | Named `_DELETE` by its author and still committed. A one-dimension variant of the front-facing compositor. |
| `generate_aug_samples.py` | `y_05_generate_augmented_bbox_samples.py` | The dataset-prep half of the YOLO trainer, duplicated and stopping before training. `training/yolo/01_train_yolov8.py` does the same split and then trains. |
| `carve_test_split.py` | `GT_04.1_Upload_azure.py` | **Read this one before deleting it.** It began as a copy of the Azure uploader, but the entire upload path is commented out. What it does now is `os.rename` the slice `df_n[80:-50]` from `frames_rota_fon` to `frames_rota_fon_TEST` — it carves a hold-out test set. It is the only code in the repository that does that, and no phase currently has a held-out split. Worth reviving properly rather than losing. |
| `iacarry_azure.html` | `iaCarry_azure_JS_1.html` | The Azure-era front end, superseded by `serving/templates/iacarry_checkout.html`. |
| `iacarry_checkout.old.html` | `server/iaCarry_Local_JS_1_Clouding.old.html` | The previous checkout screen. |

## Not archived, but worth knowing

Three live scripts still run against the **cat / dog / zombie** tutorial data
they were written with and were never repointed at iaCarry:
`training/tensorflow/06_inspect_signature_default.py`,
`training/tensorflow/07_inspect_signature_detect.py` and
`training/tflite/02_detect.py`. They are kept out of `legacy/` because they are
the only tooling for the `detect` signature the server requires — see
`README_2_TRAINING.md`.
