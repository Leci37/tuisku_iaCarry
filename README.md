# 🧠 iaCarry AI Object Detection Pipeline

This project is an **end-to-end pipeline** for training and deploying object detection models. It combines **traditional computer vision**, **data augmentation**, **Azure Custom Vision**, **TensorFlow 2**, and **TFLite** to create robust models for detecting products in real-world environments — such as **retail shelves**, **store counters**, or **mobile UIs**.

---

## 📌 Core Features

- 🎮 Extract product frames from videos
- ✂️ Remove image backgrounds (with transparency support)
- 🧪 Augment images via rotation, scaling, and perspective
- 🧹 Compose realistic product scenes using synthetic data
- ☁️ Upload images and bounding boxes to Azure Custom Vision
- ⬇️ Download annotated data and convert to COCO format
- 🧠 Train and evaluate models using TensorFlow Object Detection API
- 🔀 Export models to TensorFlow Lite (.tflite)
- 📲 Run detection on-device using TFLite
- 🖥️ Web HTML+JS frontend for visualization and UI integration

---

## 🎯 Project Objective

The goal of this project is to build a complete, scalable object detection system for identifying and classifying retail products in images or video frames. It supports both local (TensorFlow) and cloud (Azure Custom Vision) training, along with deployment-ready inference using TensorFlow Lite.

This pipeline enables:
- Automated dataset creation (real and synthetic)
- Easy integration with Azure AI tools
- Lightweight inference suitable for mobile/edge devices
- Visual inspection and interactive demos via a web interface

---

## 📊 Completion and Results

- ✅ Models trained on synthetic and real-world product data (Eroski dataset)
- ✅ Exported and validated in `.tflite` format for mobile usage
- ✅ Detection accuracy visualized through evaluation scripts
- ✅ Labels and bounding boxes confirmed with COCO viewers and Azure dashboards
- ✅ Ready-to-use HTML/JS frontend created for visual product selection

---

## 📁 Project Structure (Overview)

### 1. Data Preprocessing
| File | Description |
|------|-------------|
| `GT_01_split_video_in_frames.py` | Extracts frames from product videos. |
| `GT_02.1_remove_bg_Blanc.py` | Removes white backgrounds using `rembg`. |
| `GT_03.1_resize_rotate.py` | Resizes and rotates PNG images with transparency. |
| `GT_03_add_bg_ramdom_Blanco.py` | Composes synthetic scenes with multiple products. |

### 2. Data Augmentation Utilities
| File | Description |
|------|-------------|
| `GT_Utils.py` | Core image manipulation utilities (e.g. cropping, overlay). |
| `GT_Utils_ImageAugmentation_presp.py` | Applies perspective and scale transformations. |

### 3. Model Training and Evaluation
| File | Description |
|------|-------------|
| `Transfer_L_Mediun_Train.py` | Transfer learning with simple 3-class dataset. |
| `Transfer_L_Train_Eroski.py` | Training using Eroski product dataset. |
| `Transfer_L_Train_Eroski_[n,4].py` | Advanced multi-object training. |
| `Transfer_L_Mediun_Eval.py` | Evaluate the 3-class model visually. |
| `Transfer_L_Eval_ckt_Eroski.py` | Evaluation of Eroski model with checkpoint restore. |
| `Ultils_model_creation.py` | Shared model utilities. |

### 4. TFLite Conversion & Inference
| File | Description |
|------|-------------|
| `TFlite_convert.py` | Converts a model to `.tflite`. |
| `TFlite_convert_mdata.py` | Adds metadata to `.tflite` files. |
| `TFlite_detect.py` | Runs object detection using a TFLite model. |
| `Utils_TFlite_see_info.py` | Shows model input/output details. |
| `Utils_detect_TFlite.py` | Helper for detection via TFLite interpreter. |

### 5. Azure Integration
| File | Description |
|------|-------------|
| `GT_04_Upload_azure.py` | Uploads tagged product images to Azure. |
| `GT_04.2_Upload_azureAug.py` | Uploads augmented (synthetic) data to Azure. |
| `GT_04.2_Upload_azureBlanco.py` | Uploads mixed-background product scenes. |
| `GT_05_Azure_API_GetImg_LABELs_coco.py` | Downloads Azure images and converts to COCO. |
| `GT_06_Azure_split_coco_train_test_val.py` | Splits COCO dataset into train/val/test. |

### 6. TFRecord / COCO Conversion
| File | Description |
|------|-------------|
| `GT_07_COCO_to_TFRecord_check.md` | Instructions and notes for conversion. |
| `GT_07.2_COCO_to_TFRecord_imfolder.bat` | Batch file for TFRecord generation. |

### 7. Web Visualization
| File | Description |
|------|-------------|
| `iaCarry_azure_JS_1.html` | HTML+JS frontend for product visualization. |

---

## 🚀 Example Workflow (Full Pipeline)

### 1️⃣ Extract Images from Video
Extract frames from videos of products. These will be used as raw training data.
```bash
python GT_01_split_video_in_frames.py
```

### 2️⃣ Remove Background from Frames
Use rembg to strip white backgrounds and save alpha-transparent PNGs.
```bash
python GT_02.1_remove_bg_Blanc.py
```

### 3️⃣ Resize and Rotate for Uniformity
Standardize image dimensions (e.g., 640x640), apply random rotations for diversity.
```bash
python GT_03.1_resize_rotate.py
```

### 4️⃣ Generate Synthetic Product Scenes
Randomly compose multiple product images on real backgrounds, simulate store shelves, add BBoxes.
```bash
python GT_03_add_bg_ramdom_Blanco.py
```

### 5️⃣ Upload Images to Azure (optional)
Send labeled data to Azure Custom Vision for cloud training or dataset management.
```bash
python GT_04_Upload_azure.py
python GT_04.2_Upload_azureAug.py  # For synthetic scenes
```

### 6️⃣ Download Azure Annotations (optional)
Pull tagged images and labels from Azure and convert to COCO format for local training.
```bash
python GT_05_Azure_API_GetImg_LABELs_coco.py
```

### 7️⃣ Train Model with TensorFlow API
Fine-tune a pre-trained model like SSD MobileNet or EfficientDet using your custom dataset.
```bash
python Transfer_L_Train_Eroski.py
```

### 8️⃣ Evaluate Model Performance
Visualize model predictions on test data to verify accuracy and detection quality.
```bash
python Transfer_L_Eval_ckt_Eroski.py
```

### 9️⃣ Convert to TensorFlow Lite
Optimize model for edge devices by converting it to `.tflite` format.
```bash
python TFlite_convert.py
```

### 🔍 Run Local TFLite Inference
Use the optimized `.tflite` model for efficient on-device inference (e.g. Raspberry Pi, Android).
```bash
python TFlite_detect.py
```

---

## 📖 References
- TensorFlow Object Detection API: https://github.com/tensorflow/models
- Azure Custom Vision Docs: https://learn.microsoft.com/en-us/azure/cognitive-services/custom-vision-service/
- Rembg: https://github.com/danielgatis/rembg

---

## 🙌 Author
**@Leci37** — Built for advanced object detection workflows using a blend of local + cloud data pipelines.
