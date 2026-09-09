# 🧠 iaCarry AI Object Detection Pipeline

This project is an **end-to-end pipeline** for training and deploying object detection models. It combines **traditional computer vision**, **data augmentation**, **Azure Custom Vision**, **TensorFlow 2**, and **TFLite** to create robust models for detecting products in real-world environments — such as **retail shelves**, **store counters**, or **mobile UIs**.

---

## 🗺️ Where to start

The project has grown four distinct phases, each with its own flow, its own
files and its own failure modes. This file is the overview; the detail is split
so you can read only the phase you are working in.

| Phase | File | Covers |
|---|---|---|
| 0 · Transition | [`TRANSITION.md`](TRANSITION.md) | **What replaces phases 1 and 2, and in what order.** The confirmed defects, the intake steps and the training steps side by side with what they should be, the 2026 tool landscape, and the three decisions the plan cannot make. Read this first if you are about to record video or train a model |
| 1 · Ingestion | [`README_1_INGESTION.md`](README_1_INGESTION.md) | Raw video → labelled dataset. Frame extraction, background removal, synthetic composition, the click-once tagging GUI, mask tracking, box review, class balancing |
| 2 · Training | [`README_2_TRAINING.md`](README_2_TRAINING.md) | Dataset → model file. Transfer learning, checkpoints, the `detect` signature, evaluation, TFLite export |
| 3 · Server | [`README_3_SERVER.md`](README_3_SERVER.md) | The Flask backend. Routes, the model singleton, the inference lock, the response contract, the three thresholds |
| 4 · Front end | [`README_4_FRONTEND.md`](README_4_FRONTEND.md) | The checkout screen. State machine, themes and languages, offline rendering, the simulated seams, the verification suite |

Two things to know before reading any of them:

- **There are two parallel pipelines in this repository**, built years apart and
  sharing no files: track **A** (`synthetic/` — composed carts, Azure, TF2) and
  track **B** (`video/` — real carts, SAM, YOLOv8). Each appears as a subfolder of
  both `ingestion/` and `training/`, and phases 1 and 2 describe both. Track A is
  what the server runs today; track B has the better labels and its output is not
  yet wired to anything.
- Each of the five files ends with a **"what is missing"** section. Those are the
  open gaps, not a description of what works.

`serving/FLOW.md` complements phases 3 and 4 with the customer-level narrative and
the exact log line each step emits.

---

## 📂 Layout

```
ingestion/synthetic/   track A — frames, cut-outs, composed carts, Azure, TFRecord
ingestion/video/       track B — split, tag by hand, track masks, review boxes, balance
training/tensorflow/   track A — transfer learning, evaluation, the `detect` signature
training/yolo/         track B — YOLOv8
training/tflite/       edge export (not used by the server)
serving/               the Flask app; templates/ holds the checkout screen
common/                helpers shared by more than one phase
tools/                 check_imports.py
models/                weights and checkpoints          [gitignored]
outputs/               everything the pipelines write   [gitignored]
legacy/                superseded, kept for reference — see legacy/README.md

pyproject.toml         declares `common` as a package, for `pip install -e .`
.gitignore             outputs/, models/, and the folders the pipelines write
```

**Two naming rules, and the reason for them.** A Python module name has to be a
valid identifier, so it cannot start with a digit:

- **Numbered files are entry points** — `01_video_to_frames.py`. You run them;
  nothing imports them, so the digit is harmless and the order is visible.
- **Unnumbered files are helpers** — `label_gui_utils.py`, `common/image_utils.py`.
  Something imports them, so they carry plain identifier names.

The same rule is why the phase folders are `ingestion/` and not `1_ingestion/`:
`common/` has to be importable from both `ingestion/synthetic/` and
`training/tensorflow/`, and a package path cannot contain a digit-leading segment.
The phase numbers live in the README filenames instead.

## ▶️ Running it

```bash
pip install -e .                      # REQUIRED before running any step script
python3 tools/check_imports.py        # every local import resolves? (needs no deps)
python3 serving/verify/verify.py      # 78 browser checks against a stub /upload
```

`pip install -e .` is not optional. Python puts only the *script's own* folder on
the path, so `python ingestion/synthetic/03_resize_rotate.py` cannot see
`common/` on its own and fails with `ModuleNotFoundError: No module named
'common'`. The install is what makes the shared helpers reachable from every
folder.

`tools/check_imports.py` is static — it parses the tree rather than importing it,
so it works without TensorFlow or torch installed. It exists because the step
scripts import each other by bare module name, and nothing else in this project
notices when a rename breaks one.

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

> Track A only, and a selection rather than a full listing — this table predates
> both `ingestion/video/` (track B) and `serving/`. For the complete tree see
> **Layout** above; for the detail, the four phase documents.

### 1. Data Preprocessing
| File | Description |
|------|-------------|
| `ingestion/synthetic/01_video_to_frames.py` | Extracts frames from product videos. |
| `ingestion/synthetic/02b_remove_bg_white.py` | Removes white backgrounds using `rembg`. |
| `ingestion/synthetic/03_resize_rotate.py` | Resizes and rotates PNG images with transparency. |
| `ingestion/synthetic/04b_compose_from_white.py` | Composes synthetic scenes with multiple products. |

### 2. Data Augmentation Utilities
| File | Description |
|------|-------------|
| `common/image_utils.py` | Core image manipulation utilities (e.g. cropping, overlay). |
| `common/perspective.py` | Applies perspective and scale transformations. |

### 3. Model Training and Evaluation
| File | Description |
|------|-------------|
| `training/tensorflow/01_train_medium.py` | Transfer learning with simple 3-class dataset. |
| `training/tensorflow/03_train_eroski.py` | Training using Eroski product dataset. |
| `training/tensorflow/04_train_eroski_multi.py` | Advanced multi-object training. |
| `training/tensorflow/02_eval_medium.py` | Evaluate the 3-class model visually. |
| `training/tensorflow/05_eval_eroski.py` | Evaluation of Eroski model with checkpoint restore. |
| `common/detection_model.py` | Shared model utilities. |

### 4. TFLite Conversion & Inference
| File | Description |
|------|-------------|
| `training/tflite/01_convert.py` | Converts a model to `.tflite`. |
| `training/tflite/tflite_metadata.py` | Adds metadata to `.tflite` files. |
| `training/tflite/02_detect.py` | Runs object detection using a TFLite model. |
| `training/tflite/tflite_info.py` | Shows model input/output details. |
| `training/tflite/tflite_detect_utils.py` | Helper for detection via TFLite interpreter. |

### 5. Azure Integration
| File | Description |
|------|-------------|
| `ingestion/synthetic/05_azure_upload.py` | Uploads tagged product images to Azure. |
| `ingestion/synthetic/05b_azure_upload_augmented.py` | Uploads augmented (synthetic) data to Azure. |
| `ingestion/synthetic/05c_azure_upload_white.py` | Uploads mixed-background product scenes. |
| `ingestion/synthetic/06_azure_download_coco.py` | Downloads Azure images and converts to COCO. |
| `ingestion/synthetic/07_coco_split.py` | Splits COCO dataset into train/val/test. |

### 6. TFRecord / COCO Conversion
| File | Description |
|------|-------------|
| `ingestion/synthetic/README_tfrecord.md` | Instructions and notes for conversion. |
| `ingestion/synthetic/08b_coco_to_tfrecord_from_folder.bat` | Batch file for TFRecord generation. |

### 7. Web Visualization
| File | Description |
|------|-------------|
| `serving/templates/iacarry_checkout.html` | The checkout screen served at `GET /`. See `README_4_FRONTEND.md`. |
| `legacy/iacarry_azure.html` | The older Azure-era viewer, superseded — kept for reference only. |

---

## 🚀 Example Workflow (track A)

> This is the synthetic-composition track end to end. It does not cover track B
> (`ingestion/video/`, `training/yolo/`) or running the station itself
> (`serving/`) — see `README_1_INGESTION.md` and `README_3_SERVER.md`.

**Run this first.** Every step below imports `common/`, and Python puts only the
*script's own* folder on the path, so without the install they fail with
`ModuleNotFoundError: No module named 'common'`:

```bash
pip install -r requirements.txt
pip install -e .
```

### 1️⃣ Extract Images from Video
Extract frames from videos of products. These will be used as raw training data.
```bash
python ingestion/synthetic/01_video_to_frames.py
```

### 2️⃣ Remove Background from Frames
Use rembg to strip white backgrounds and save alpha-transparent PNGs.
```bash
python ingestion/synthetic/02b_remove_bg_white.py
```

### 3️⃣ Resize and Rotate for Uniformity
Standardize image dimensions (e.g., 640x640), apply random rotations for diversity.
```bash
python ingestion/synthetic/03_resize_rotate.py
```

### 4️⃣ Generate Synthetic Product Scenes
Randomly compose multiple product images on real backgrounds, simulate store shelves, add BBoxes.
```bash
python ingestion/synthetic/04b_compose_from_white.py
```

### 5️⃣ Upload Images to Azure (optional)
Send labeled data to Azure Custom Vision for cloud training or dataset management.
```bash
python ingestion/synthetic/05_azure_upload.py
python ingestion/synthetic/05b_azure_upload_augmented.py  # For synthetic scenes
```

### 6️⃣ Download Azure Annotations (optional)
Pull tagged images and labels from Azure and convert to COCO format for local training.
```bash
python ingestion/synthetic/06_azure_download_coco.py
```

### 7️⃣ Train Model with TensorFlow API
Fine-tune a pre-trained model like SSD MobileNet or EfficientDet using your custom dataset.
```bash
python training/tensorflow/03_train_eroski.py
```

### 8️⃣ Evaluate Model Performance
Visualize model predictions on test data to verify accuracy and detection quality.
```bash
python training/tensorflow/05_eval_eroski.py
```

### 9️⃣ Convert to TensorFlow Lite
Optimize model for edge devices by converting it to `.tflite` format.
```bash
python training/tflite/01_convert.py
```

### 🔍 Run Local TFLite Inference
Use the optimized `.tflite` model for efficient on-device inference (e.g. Raspberry Pi, Android).
```bash
python training/tflite/02_detect.py
```

---

## 📖 References
- TensorFlow Object Detection API: https://github.com/tensorflow/models
- Azure Custom Vision Docs: https://learn.microsoft.com/en-us/azure/cognitive-services/custom-vision-service/
- Rembg: https://github.com/danielgatis/rembg

---

## 🙌 Author
**@Leci37** — Built for advanced object detection workflows using a blend of local + cloud data pipelines.


---

# 🛒 iaCarry: Revolutionizing the Checkout Experience

**Website:** http://iacarry.tuisku.eu  
**Demo Video:** http://www.youtube.com/watch?v=Sj-IvGnjODE  
**GitHub:** https://github.com/Leci37/tuisku_iaCarry

iaCarry is an innovative AI-based checkout system designed to drastically reduce transaction times and eliminate the bottlenecks caused by traditional scanning. Developed by tuisku.eu, it leverages cutting-edge technologies like Google TensorFlow and OpenCV to visually recognize products in a shopping cart — with no need for barcodes to be scanned manually.

---

## ✨ Key Features of iaCarry

### ⚡ Rapid Checkout Process
Customers can complete their purchases in as little as 6 seconds, greatly reducing queues and wait times.  
📍 *Source: ia_carry_home2*

### 🧠 Advanced AI & Computer Vision
Trained using models developed in collaboration with Google TensorFlow and Facebook AI, iaCarry can identify products by simply analyzing an overhead image of the cart.  
📍 *Source: ia_carry_home2, YouTube demos*

### 💸 Economic and Resource Efficiency
Eliminates the need for a dedicated cashier scanning items, allowing personnel to focus on value-added tasks while significantly cutting operational costs.  
📍 *Source: ia_carry_home2*

### 🔧 Simplified Maintenance
Unlike RFID or cart-mounted camera systems, iaCarry centralizes the technology in a single station, simplifying maintenance and lowering long-term costs.  
📍 *Source: Product*

### 🏪 Adaptability to Store Size
iaCarry is modular and fits into spaces as small as 4.5 square meters, making it ideal for both large supermarkets and compact retail stores.  
📍 *Source: Product*

---

## 🧠 How iaCarry Works (Full Explanation)

1. **Cart Positioning:** Customer places their cart on the iaCarry-Station equipped with a scale.
2. **Product Recognition:** Overhead camera takes a photo of the contents.
3. **AI Analysis:** The image is processed to detect products and retrieve their prices/barcodes/weights.
4. **Customer Interaction:** Real-time display with visual and sound feedback.
5. **Payment:** Customer pays via contactless methods at the station.
6. **Completion:** Gate opens, customer leaves — no lines, no friction.

---

## 🍌 Handling Bulk / Fresh Products

iaCarry supports:
- **Pre-Weighing:** Items like fruits and meat can be weighed in-store and labeled, then recognized by the system.
- **Integrated Weighing:** Additional scales can be installed beside the station for on-the-spot weighing.

---

## 🛡️ Anti-Fraud Measures

- Detects missing or hidden products.
- Verifies total weight of items to catch unscanned products.
- Prompts users with visual/auditory warnings when anomalies are detected.

---

## 🧩 Product Offerings

### 🏢 Large Supermarkets
- Full-featured iaCarry Station
- Requires 4.5 m² and integration with POS systems
- Includes staff training and customer education

### 🛍️ Small Supermarkets
- Compact version
- Customized software
- Cost-effective hardware + remote support

### 👕 Clothing & Furniture Stores
- Uses QR codes, RFID or visual features (color, size)
- Specialized AI models for non-barcode products
- Recognition of furniture models, apparel styles, etc.

---

## 📈 Investment Opportunities

iaCarry is open to:
- Retail partnerships
- Franchise or white-label solutions
- Technology investment or acquisition

**Benefits for investors:**
- Breakthrough in AI retail automation
- High efficiency = increased throughput = higher revenue
- Attracts modern shoppers who expect self-service and speed
- Scalable across markets (grocery, fashion, electronics, etc.)

---

## 🎥 Demonstrations and Visuals

- ✅ Main Demo – How iaCarry Works
- ⚡ Express Checkout Video 1
- ⚡ Express Checkout Video 2

These videos showcase real-world usage, transaction speed, and user experience at iaCarry stations.

---

## 📬 Contact & Info

**Developer:** tuisku.eu  
**Location:** Calle la Fanderia, 2, 48901 Barakaldo, Biscay, Spain  
**Website:** http://iacarry.tuisku.eu

---

## 🧾 Summary

iaCarry is not just a self-checkout system — it’s a complete AI-powered retail transformation tool. From lightning-fast payment to fraud detection and plug-and-play scalability, iaCarry offers the future of shopping today.

---

## 💼 Pitch Deck Highlights

**What is iaCarry?**  
iaCarry is an AI-powered checkout station designed to complete a full shopping cart transaction in just 6 seconds using an overhead image and advanced computer vision.

- Developed by tuisku.eu
- Built with Google TensorFlow and Facebook’s Computer Vision tools
- Embedded in a physical station: scale + overhead camera + screen + payment terminal

---

## 🌍 Market Opportunity

- 23,572 supermarkets in Spain and over 234,000 stores across Europe still depend on conventional barcode scanning.
- iaCarry targets:
   - Large supermarkets
   - Small convenience stores
   - Clothing and furniture retailers (customized with QR or smart labels)

---

## 🎯 Vision & Value Proposition

- "Payment in 6 seconds!" – drastically reduces customer time spent at checkout.
- AI recognition replaces barcode scanning
- Seamless and scalable integration with existing retail infrastructure
- Optimized for store staff productivity – staff can focus on stocking instead of scanning
- Error reduction – less manual intervention = fewer mistakes

---

## ⚙️ How Does It Work?

1. Cart Placement: Customer pushes their cart onto the scale.
2. Overhead Image Capture: The system takes an aerial photo.
3. AI Product Detection: TensorFlow-powered model detects all products in view.
4. Price/Barcode Match: Items are matched to SKUs (price, barcode, weight).
5. Real-Time Feedback: Customers see the scanned items on a screen.
6. Payment: They pay via electronic payment (card, NFC).
7. Exit: Once paid, the system unlocks the gate automatically.

---

## 🔐 Anti-Fraud Features

- Picaresque control: Validates cart weight to prevent hidden/missing products.
- Bulk product detection: Camera AI can handle unpackaged items without extra scales.
- Alerts customer in real-time for inconsistencies.

---

## 🏪 Retail Formats Supported

| Retail Type         | Customization Features                                    |
|---------------------|-----------------------------------------------------------|
| Large supermarkets  | Standard iaCarry-Station (4.5 m²), fully integrated       |
| Small stores        | Space-optimized versions, remote support                  |
| Clothing/Furniture  | Smart label support (QR, color/style detection), specialized cameras |

---

## 🏆 Competitive Advantages

- Uses conventional trolleys (no hardware required on carts)
- Minimal infrastructure: scale, camera, screen, dataphone
- Easy to install, easy to scale
- Cheaper than competitors (no cart computers, charging stations, Wi-Fi mesh)
- Real fraud detection, unlike "take and go" competitors

---

## 📊 Revenue Model

| Revenue Source     | Details                                                       |
|--------------------|---------------------------------------------------------------|
| Station installation | €12,000 per station                                          |
| Trolley cost         | €38 per trolley                                              |
| Software license     | €420/month per station or recognition-based (e.g. €0.01/item) |
| Maintenance          | €165/month per station                                       |
| Optional training    | Included in B2B sales packages                               |

---

## 📈 Scalability Strategy

- First 10 stations operational → global rollout
- Scalable for:
   - Supermarkets
   - Retail apparel/furniture
   - Tech/consumer goods stores
- The underlying iaCarry Kernel is also reusable for other applications like iaRecycle (AI-based recycling automation).

---

## 💼 Investment Needs

To scale the project, iaCarry is looking for:

1. Implementation partner (to deploy prototype units)
2. Cloud compute rental (for model resolution)
3. Optional team expansions:
   - 2 more developers
   - 1 QA engineer (to eliminate critical bugs)
   - 7 commercial/sales staff (national + international)

---

## 👨‍💻 Meet the Founder

**Luis Castillo**  
Founder of tuisku.eu  
- AI/ML developer, specialized in neural networks  
- GitHub: https://github.com/Leci37  
- LinkedIn: https://linkedin.com/in/luislcastillo/


---

## 🖼️ Checkout screen assets

The self-checkout screen serves every image it draws from `serving/static/assets/`
(product thumbnails, demo frames, client logos), so it renders completely with
outbound internet blocked — a requirement on shop floors with restricted egress.
See `serving/static/assets/README.md` for contents, including the three client
logos under `logos/`.

**Trademarks:** the Eroski, AhorraMas and Condis marks are registered trademarks
of their respective owners, included for demo use with each retailer's permission
only. They are not covered by this repository's licence.
