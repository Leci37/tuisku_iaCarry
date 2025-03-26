# 🧠 iaCarry AI Object Detection Pipeline

This project is an end-to-end pipeline for training and deploying object detection models. It combines traditional computer vision, data augmentation, Azure Custom Vision, TensorFlow 2, and TFLite to create robust models for detecting products in real-world environments — such as retail shelves, store counters, or mobile UIs.

---

## 📌 Core Features

- 🎞 Extract product frames from videos
- ✂️ Remove image backgrounds (with transparency support)
- 🧪 Augment images via rotation, scaling, and perspective
- 🧩 Compose realistic product scenes using synthetic data
- ☁️ Upload images and bounding boxes to Azure Custom Vision
- ⬇️ Download annotated data and convert to COCO format
- 🧠 Train and evaluate models using TensorFlow Object Detection API
- 🔁 Export models to TensorFlow Lite (.tflite)
- 📲 Run detection on-device using TFLite
- 🖥️ Web HTML+JS frontend for visualization and UI integration

---

## 📁 Project Structure (Overview)

<details>
<summary><strong>Click to expand full file breakdown</strong></summary>

### 📦 Data Creation & Augmentation

| Script | Function |
|--------|----------|
| `GT_01_split_video_in_frames.py` | Extracts frames from video (used as base dataset) |
| `GT_02_remove_bg.py` / `GT_02.1_remove_bg_Blanc.py` | Removes background (using `rembg` and alpha mask logic) |
| `GT_03.1_resize_rotate.py` | Rotates, resizes and normalizes transparent images |
| `GT_03_add_bg_ramdom_*.py` | Synthesizes new training samples by overlaying products on real backgrounds |
| `GT_Utils.py` / `GT_Utils_ImageAugmentation_presp.py` | Image manipulation, overlay, random transforms, and helpers |

### ☁️ Azure Upload / Download

| Script | Function |
|--------|----------|
| `GT_04_Upload_azure.py` | Uploads labeled images with bounding boxes |
| `GT_04.2_Upload_azureBlanco.py` / `GT_04.2_Upload_azureAug.py` | Upload specific batches (e.g. with augmented products) |
| `GT_05_Azure_API_GetImg_LABELs_coco.py` | Downloads tagged images + boxes from Azure and converts to COCO |
| `GT_06_Azure_split_coco_train_test_val.py` | Splits COCO dataset for training/evaluation |

### 🔁 Dataset Conversion (TFRecord)

| Script | Function |
|--------|----------|
| `GT_07.1_COCO_to_TFRecord.bat` | Converts COCO → TFRecord (Windows batch) |
| `GT_07_COCO_to_TFRecord_check.md` | Docs for COCO format and tools used |

### 🧠 Model Training & Evaluation

| Script | Function |
|--------|----------|
| `Transfer_L_Train_Eroski.py` / `[n,4].py` | Transfer learning using TF2 pretrained models |
| `Transfer_L_Mediun_Train.py` | Training using smaller synthetic classes |
| `Transfer_L_Eval_ckt_Eroski.py` / `Transfer_L_Mediun_Eval.py` | Evaluates models using test images |
| `model_Creation_Rubber_tf2_colab.py` | Colab version for quick training |

### 🧰 Utilities

| Script | Function |
|--------|----------|
| `Ultils_model_creation.py` | Utilities for visualization, model loading, configs |
| `Utils_Detect_Signature.py` | Visualize TF predictions with serving signatures |
| `Utils_detect_TFlite.py` | Runs inference using TFLite interpreter |
| `Utils_TFlite_see_info.py` | Logs details of TFLite model structure and metadata |

### 📲 TensorFlow Lite Conversion + Inference

| Script | Function |
|--------|----------|
| `TFlite_convert.py` | Converts SavedModel → TFLite (float + quantized) |
| `TFlite_convert_mdata.py` | Adds metadata to TFLite models |
| `TFlite_detect.py` | Runs object detection on images using `.tflite` models |

### 🖥️ Web UI

| File | Function |
|------|----------|
| `iaCarry_azure_JS_1.html` | Web interface for showing detected products with name, price, etc. (JS/CSS/HTML) |

</details>

---

## 🚀 Example Workflow (Full Pipeline)

### 1️⃣ Extract Images from Video
```bash
python GT_01_split_video_in_frames.py
