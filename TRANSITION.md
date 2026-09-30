# 0 — Transition plan

**The point this repository is turning at.** Phases 1 and 2 describe the pipeline
as it was built. This file describes what replaces it, why, and in what order.

Read this before `README_1_INGESTION.md` and `README_2_TRAINING.md` if you are
about to record more video or train another model. Read those two first if you
need to understand what exists today.

Two parts, matching the two phases being replaced:

- **[Part A — Intake](#part-a--intake)** · raw video → a dataset you can trust
- **[Part B — Training](#part-b--training)** · that dataset → a model you can measure

Everything here is either (a) read in the code on this branch, with `file:line`,
or (b) checked against a vendor or primary source on 2026-09-09, listed in
[Sources](#sources). Nothing was measured on iaCarry data — no dataset, weights,
`label_map.pbtxt` or checkpoint exists on the machine this was written on.

---

## Why now

The pipeline's design is sound. The click-once-and-track idea was early and
correct: model-assisted annotation with human quality control is what the
industry settled on. What has gone wrong is narrower and worse.

> 🔴 **Both quality gates are broken, and both fail silently.** The project has
> exactly two mechanisms meant to guarantee data quality — a human who reviews
> each frame, and a validation split that scores the model. The reviewer approves
> a *different frame* from the one on screen, and the validation set contains
> augmented copies of the training images. The human gate blesses unseen data;
> the numeric gate marks its own homework. This is the whole reason the project's
> state cannot currently be assessed.

And one thing changed in the wider world that removes the deepest design flaw for
free:

> ✅ **SAM 3 returns every instance of a concept at once, each with its own ID.**
> SAM 1 and SAM 2 predicted one object per prompt — which is exactly why
> `02_label_gui.py:64` stores one mask per product *name*, and why two identical
> products collapse into one. Prompt `cocacola_33cl` once and all of them come
> back separately. The counting bug is fixed by the model, not by code anyone has
> to write. SAM 3 shipped 2025-11-19; SAM 3.1 followed 2026-03-27 and tracks up to
> **16 objects in a single forward pass**. A cart holds roughly 8–15 items.

---

## Confirmed defects

Ranked by damage to final model quality. Every row was read in the source on
`_co_dev1`. These are the reason for the transition, not a wish list.

| Sev | Defect | Where | Effect |
|---|---|---|---|
| 🔴 | Masks keyed by product label — no instance identity | `ingestion/video/02_label_gui.py:64`, `label_gui_utils.py:146` | A second click on the same label **overwrites** the first. One `.npy` per label per frame; downstream keeps only the largest connected component. Quantity is unrepresentable — while the checkout screen displays `✓ × N` badges |
| 🔴 | Review GUI approves the wrong frame | `ingestion/video/04_bbox_review.py:128`, `:187` | `accept_frame()` loads the *next* frame, saves it, marks it accepted, *then* displays it. `demo.load(fn=accept_frame)` auto-accepts frame 1 sight-unseen. The operator judges frame N; the keystroke commits N+1 |
| 🔴 | Augmented copies split across train and val | `ingestion/video/06_rotate_stats_balance.py:131`, `training/yolo/01_train_yolov8.py:30` | `X__aug1.png` is written beside `X.png`; the trainer then splits basenames at random. The same photograph, colour-jittered, lands on both sides |
| 🔴 | Near-duplicate frames split across train and val | `training/yolo/01_train_yolov8.py:30` | Frames pooled from all clips and split individually. Frames 4 and 9 of one clip — same cart, 5/30 s apart — routinely land on opposite sides |
| ⚠️ | Mask post-processing selects the background | `ingestion/video/sam_wrapper.py:58`, `:69` | `labels` is shifted `+1`, then component `largest_idx + 1` is selected. For a single foreground blob that expression resolves to the background. The mask is inverted |
| ⚠️ | Inverted masks pass every quality gate | `ingestion/video/track_masks_utils.py:89` | There is a `min_area` and **no maximum**. A near-full-frame mask has aspect ratio 1.78, solidity ≈ 1.0, circularity ≈ 0.79, one contour — it clears all five filters and becomes a whole-image box labelled as that product |
| ⚠️ | Tracker retry re-prompts with a hardcoded centre box | `ingestion/video/sam_wrapper.py:148` | When a tracked mask drops below `min_area` — i.e. exactly when the product becomes small or occluded — SAM is re-prompted with the middle 50 % of the frame, not the object. Live, via `03_track_masks.py:47` |
| ⚠️ | Geometry filters reject the target catalogue | `ingestion/video/track_masks_utils.py:89` | `max_aspect_ratio=4.0`, `min_circularity=0.15`. Toothbrushes, toothpaste tubes edge-on, spaghetti boxes, thin visible slivers of occluded items — all systematically discarded |
| ⚠️ | No held-out test set | `training/yolo/01_train_yolov8.py:122` | Train/val only, then `val_ids[:5]` reused as "test predictions". No honest final number even before the leakage |
| ⚠️ | No correction path in review | `ingestion/video/bbox_review_utils.py:163` | `correct_box()` exists and is wired to no control. A nearly-right box must be discarded |
| ⚠️ | Metrics block never runs | `training/yolo/01_train_yolov8.py:106-117` | Reads `results.metrics`, `.map50`, `.precision`, `.ap_class` — not where Ultralytics puts them. Raises `AttributeError`, so no metrics summary has ever printed. Verify against the pinned version |
| ⚠️ | Unknown classes dropped from the basket | `serving/FLOW.md:330` | A product the model names but the front-end catalogue lacks is discarded and never charged — classified as "allowed". A correct prediction the application throws away is still a product failure |
| 🔴 | Azure training key in plaintext, public repo | `ingestion/synthetic/05b_azure_upload_augmented.py:14`, `06_azure_download_coco.py:22` | Rotate it. The only item here with a deadline set by someone else |
| ℹ️ | "98 % accuracy" is a literal | `serving/templates/iacarry_checkout.html:194` | Hardcoded, as is "1.2s inference". Shown to customers and quoted in `README.md`. Measure it or remove it |
| ✅ | The labelling GUI itself is **not** at fault | `ingestion/video/02_label_gui.py:57` | Clicks *are* passed to SAM correctly via `first_frame_click`. The broken centre-box `segment_objects()` is dead code, reachable only from `legacy/` |

> ⚠️ **The READMEs are reliable on structure and unreliable on behaviour.** They
> state that quantitative YOLO evaluation and export are absent; the script
> attempts both. The real problem is subtler — the metrics call throws.

---

# Part A — Intake

Raw video → a dataset you can trust. The workflow shape does not change. Almost
every step below is a repair or a tool swap; exactly one is a rewrite.

## The target loop

```
YOU      prompt each SKU ──────────────► correct once ──────────► review the export
         text or a few clicks           at the FIRST bad frame    tens, not thousands
            │                                  ▲   │                      ▲
            ▼                                  │   ▼                      │
TOOL     all instances found ─► propagate ─────┘   re-propagate ─► export diverse ─┘
         separate IDs · SAM 3   through clip        that span only    frames → boxes
```

Two rules make this work, and both are absent today:

1. **Correct at the first bad frame, not the last.** Fixing only where the drift
   was noticed leaves the whole corrupted span in the dataset. Today
   `03_track_masks.py` *discards* a failing frame and moves on — there is no path
   back to repair the track, which is why bad spans survive as silently-missing
   labels.
2. **Propagate densely, export selectively.** Dense tracking maintains identity;
   dense export just multiplies near-identical images and the review burden with
   them.

## A0 · Rules and catalogue — before any tooling

Nothing here is code. It is the cheapest step and prevents the most rework, and
every tool below will ask these questions anyway.

| Artefact | Contents |
|---|---|
| SKU catalogue | Stable id, name, variant, size, reference photo. One CSV. Two units of one product share a SKU and take different instance ids |
| Annotation rules | Minimum visibility to count as present · how repeated units are handled · how occlusion is handled · box-tightness policy |
| Capture plan | Camera geometry, what each session must cover |

**Box policy, decided once and applied everywhere:** boxes enclose the **visible
extent** of each instance, derived from its visible mask. Preserve disconnected
visible parts of one object where occlusion splits the mask. Never invent a box
for a fully hidden object — it may stay in a temporal inventory record, but it
gets no image annotation.

**Effort:** half a day *(estimate)*. Also rotate the Azure key here.

## A1 · Capture — variety, not translation

| Now | Should be |
|---|---|
| Ad-hoc clips. `01_split_videos.py` splits to 14–30 s and renames onto the NATO alphabet. `ENABLE_SPLIT = False` as committed, so it copies without splitting | Planned sessions against the capture plan: rotations, front/back, leaning packs, touching objects, **repeated units of one SKU**, overlap, basket edges, reflections, varied lighting, empty carts, non-product distractors |

Use **move → withdraw hand → pause** sequences so settled views exist. Keep some
clips with hands and motion if the deployed system must work during interaction.

> ⚠️ **Ten videos are not ten thousand independent examples.** Ten one-minute
> clips at 30 fps hold 18,000 frames; every 60th gives ~300 images, one or two per
> second gives 600–1,200, and at ~8 visible items each that is roughly
> 4,800–9,600 **box annotations**. Those are annotation counts, not independent
> scenes. Frames 120 and 180 of one clip show the same products in nearly the same
> arrangement under identical lighting from one camera position. For
> generalisation, ten videos of one cart are closer to *ten* samples than ten
> thousand. The tracking trick buys an enormous reduction in labelling **effort**
> — keep it — but it cannot manufacture information the camera never saw. Variety
> comes from sessions.

**What ten videos can honestly deliver:** proof the workflow is efficient, and a
first pilot dataset. Not certification of anything.

## A2 · Identify — the one rewrite

| Now | Should be |
|---|---|
| Operator picks a label and clicks; SAM returns a mask. Correct as far as it goes — but `state["masks"][label] = mask` keys by label, so there is **one mask per product name per frame** | One identity per **physical item**. Every annotation carries three separable facts: **category**, **SKU**, **instance**. Two cartons of the same milk share a SKU and hold different instance ids |

A useful annotation record distinguishes:

| Field | Example |
|---|---|
| Broad category | Milk |
| Exact SKU | Brand X whole milk, 1 L |
| Physical instance | Carton 17, session B |
| Visibility | Partially occluded |
| Annotation origin | SAM proposal, corrected by reviewer |
| Review state | Approved |
| Source | Video, timestamp, frame index |

A mask supplies geometry. It does not prove the SKU — that is a human judgement,
and it is the one judgement worth your time.

> ✅ **Commit to this schema now even while keeping a single-stage detector.**
> Deferring the two-stage *architecture* (locate, then identify) until there is
> evidence of a scaling problem is correct. Deferring the *schema* is not: a
> schema that cannot express instances forces a re-annotation project later, and
> the instance fields are needed anyway to fix counting.

## A3 · Propagate — swap the engine

| Now | Should be |
|---|---|
| SAM 1 ViT-H + XMem through a Track-Anything fork. On low mask area it re-prompts with a hardcoded centre box (`sam_wrapper.py:148`) and then inverts the mask (`:58`, `:69`) | A promptable video segmenter with native memory-based propagation, driven from an annotation tool rather than a bespoke script. Handle entry, exit and reappearance explicitly. Correct at the first bad frame and re-propagate that interval |

`ingestion/video/sam_wrapper.py` and `03_track_masks.py` are the files this step
retires. Two generations behind, and both defects above are live.

## A4 · Quality filtering — repair, do not delete

| Now | Should be |
|---|---|
| Generic geometry gates: `min_area=300`, `max_aspect_ratio=4.0`, `min_circularity=0.15`, `min_solidity=0.10`, `max_components=4`. No maximum area. A frame failing the check is dropped whole | Filters appropriate to product geometry — a toothbrush is legitimately long and thin. **Add a maximum-area gate** so an inverted mask cannot pass. Route suspicious masks to human review rather than deleting them silently. Flag **missing** instances, the error class the current gates cannot see at all |

Automated warnings worth having, in priority order — these prioritise review, they
do not certify labels:

| Warning | Likely problem |
|---|---|
| Mask area changes abruptly | Background included, or part of the object lost |
| An instance disappears unexpectedly | Tracking loss |
| Two instances occupy nearly the same region | Merge, or identity swap |
| Image, label or source reference missing | Incomplete export |

Also sample the cases that raise *no* warning — a stable error goes unnoticed.

## A5 · Frame selection — positional → informative

| Now | Should be |
|---|---|
| Track every frame, keep the middle one of each run of five (`filter_yolo_center_frames(window_size=5)`). Purely positional | Propagate densely, export selectively: keep frames where pose, overlap, visibility or lighting actually changed; drop near-duplicates on embedding distance |

This is where FiftyOne earns its place — `compute_near_duplicates()` answers
"which of my thousands of tracked frames are actually different?"

## A6 · Human validation — repair first, then swap

| Now | Should be |
|---|---|
| A Gradio app that saves and marks accepted the frame it is *about to show*. Loading the page accepts frame 1. No correction control. Accept or discard only | The decision must apply to the frame **on screen** — fix this before anything else in review. Then: accept, correct, reject, **and uncertain**; an explicit "is every visible in-scope product labelled?" check; recorded reviewer identity and timestamp |

Five questions that decide whether a label is good. Apply all five, every time:

| Question | Example failure |
|---|---|
| Is it the right product? | `colgate_75ml` labelled as another variant |
| Is each unit separate? | Two cans inside one box |
| Does the box follow the agreed boundary policy? | Clips the product, or includes a neighbour |
| Is **every visible** product labelled? | The tracker lost a toothbrush and the frame was saved without it |
| Does the label belong to **this exact image**? | Coordinates from before a rotation |

Review happens at two levels: **during tracking**, play the clip with masks
painted to spot jumps, losses and merges — watch crossings and occlusions
especially; and **before training**, review every exported frame of the first
batch, looking for objects the model never proposed.

**Consistency check:** have someone re-review ~50 varied frames, including
repeated products and occlusions, without seeing the original decisions. Count
wrong-SKU, omitted objects, merged units and badly-fitted boxes. That is an
initial audit, not a statistical certification. A single failure usually means a
whole video span is affected — go back to the source.

## A7 · Splitting — decided in intake, not in the trainer

| Now | Should be |
|---|---|
| Not part of intake at all. The trainer splits image basenames at random, after offline augmentation | Group by **recording session**, split whole groups, **before** augmentation, and record the assignment in the manifest |

```
AS COMMITTED                         AS IT SHOULD BE
one session, one cart                separate sessions
  f_060 ─► TRAIN                       sessions A–D ─► TRAIN   (all frames + aug)
  f_120 ─► TRAIN                       session  E   ─► VAL     (all frames)
  f_180 ─► VAL    ◄── same photo!      session  F   ─► TEST    (untouched)
  f_180__aug1 ─► TRAIN                 
  f_240 ─► VAL                       no frame shares a cart with the other side
```

Hold out unseen **sessions**, unseen **lighting** and unseen **arrangements** —
not unseen frames. Grouping key = recording session. With only six-ish groups the
test set is a pilot diagnostic, not a production gate.

## A8 · Output and lineage

| Now | Should be |
|---|---|
| YOLO and COCO files across hand-configured folders, with an undocumented rename between steps 03 and 04. `classes.txt` read from two different directories that must not disagree | One versioned dataset release with a manifest carrying, per image: source video, frame index, instance and SKU ids, annotation origin (proposed vs. approved), reviewer decision, content hash |

Store training frames **clean** — no painted masks, no drawn boxes. Keep overlays
separately. Validate that every exported label matches its image's actual
orientation and dimensions.

> 🔴 **`label_map.pbtxt` defines what class id 5 means for every label ever
> produced, and it exists on exactly one machine and nowhere else.** It is not in
> this repository. Getting it under version control with a hash is a
> prerequisite for every other claim in Part B.

## A9 · Synthetic composition — demote and schedule its retirement

| Now | Should be |
|---|---|
| Cut-outs composited onto backgrounds, Azure Custom Vision round trip. **This is what the production model was trained on** | An optional supplement whose benefit is **measured** on real held-out carts. Keep composited siblings grouped; never derive synthetic assets from held-out recordings |

Compare real-only training against real-plus-synthetic on the same real
validation set. If it cannot show a gain there, delete it and reclaim the
maintenance.

> ⚠️ **Azure Custom Vision retires 2028-09-25**, and Microsoft's guidance was to
> have a transition plan in place by **2026-09-25** — about two weeks from this
> writing. The dependency was going to be retired anyway; this makes it scheduled.

---

# Part B — Training

That dataset → a model you can measure. Ordered so that each step is meaningful
only once the one above it is done.

## B0 · Framework — the question answered

> **"Is TensorFlow really the best choice?"** The framework was never the problem.
> Training already happens in PyTorch — Ultralytics is PyTorch. The legacy TF2
> Object Detection API path serves the production model and depends on an
> `exporter_main_v2.py` step that is not in this repository and that nobody can
> currently reproduce. That is a **reproducibility** problem, not a framework
> problem. Migrating frameworks would fix nothing on the defect list.

| Now | Should be |
|---|---|
| Two stacks: legacy TF2 OD API in production, Ultralytics/PyTorch for new work | One primary reproducible route — the PyTorch one already in use. Keep the TF SavedModel serving until a replacement is measured. Preserve the legacy model for comparison |

## B1 · Splits — fix this before anything else

| Now | Should be |
|---|---|
| `train_test_split` over pooled frame basenames, after offline augmentation (`01_train_yolov8.py:30`) | Grouped by session (see [A7](#a7--splitting--decided-in-intake-not-in-the-trainer)). Augmentation at **training time only** — never as files on disk. A third split of whole sessions, untouched during model selection and threshold tuning |

Then retrain `yolov8n` **unchanged** and **expect the number to fall**. That fall
is the most valuable result in this plan: it is the first trustworthy measurement,
and every comparison afterwards depends on it existing.

## B2 · Metrics — make results exist

| Now | Should be |
|---|---|
| A print block that raises `AttributeError`, with hand-written `ideal >` thresholds beside each line. Track A evaluates by eye on JPEGs. No mAP, PR curve or confusion matrix stored anywhere in the repo | Saved, versioned metrics. Ultralytics already provides mAP, PR curves and confusion matrices — no new architecture is needed to start measuring |

The printed `ideal >` values are **not** release criteria.

## B3 · The scorecard

Detection metrics are necessary and not sufficient. Every target below is a
**proposal to be agreed — not a measurement and not an industry standard**. Report
all of them **with sample counts**, sliced by SKU, crowding, occlusion level,
small-object size, lighting, repeated SKU and session.

### Annotation health — is the data worth training on?

| Measure | Proposed target | Tells you |
|---|---|---|
| Missed visible instances in an audited sample | < 2 % | Whether labels teach the model to ignore products. The failure the geometry filters cannot see |
| Wrong-SKU rate in an audited sample | < 1 % | Whether product identity is trustworthy |
| Merged / split instance rate | < 1 % | Tracks the counting bug directly. Measure on frames that deliberately contain repeated units |
| Approved frames per reviewer hour | baseline it | Whether the workflow change actually saved time. **The number that decides whether to scale from two videos to ten** |

### System behaviour — does the product work?

| Measure | Proposed target | Tells you |
|---|---|---|
| mAP@50-95, mAP@50, per-SKU AP | on held-out sessions | Localisation and recognition. Meaningful only once the split is grouped |
| Per-SKU precision / recall at the **deployed** threshold | agree per SKU | False charges vs. missed items in operation — not at the threshold that flatters the model |
| Confusion matrix, background included | reviewed each run | Which package variants are confused, and what is invented from nothing |
| **Basket exact-match rate** | the headline number | Every SKU and quantity right, no extras. Brutal: 20 items each independently 98 % correct give a fully correct basket only ~67 % of the time |
| Signed and absolute euro error per basket | track both | Signed reveals systematic under/over-charging; absolute reveals volatility. Undercounting is the predicted failure of the current labels |
| Unknown-item referral rate | measure, don't drop | Whether unfamiliar products are surfaced. Today they are discarded (`FLOW.md:330`) |
| Automatic-acceptance coverage | report with accuracy | A system can look excellent by referring most carts to a human. Correctness among auto-accepted baskets is meaningless without the fraction auto-accepted |
| p50 / p95 end-to-end latency on target hardware | agree a budget | Real user-facing speed including pre- and post-processing |

> ⚠️ **Sample size is measured in sessions, not frames.** Zero failures in 100
> independent cart trials still leaves roughly a 3 % upper bound on the true
> failure rate. Hundreds of near-identical frames do not supply hundreds of
> independent trials.

## B4 · Detector — baseline, then one challenger

| Now | Should be |
|---|---|
| `yolov8n.pt`, `imgsz=1280`, 50 epochs, batch 16, `device=0` hardcoded, no seed passed to `train()` | Re-establish `yolov8n` as a **measured** baseline on a clean split first. Then one challenger at a time. Pass a seed; repeat close comparisons across seeds before believing a small gain |

Published figures, as a shortlist and **not** a ranking of what wins on carts —
COCO accuracy does not transfer to this domain:

| Model | COCO AP | Params | Latency | Licence |
|---|---|---|---|---|
| RF-DETR Nano | 48.4 | 30.5 M | 2.3 ms | Apache 2.0 |
| RF-DETR Small | 53.0 | 32.1 M | 3.5 ms | Apache 2.0 |
| RF-DETR Medium | 54.7 | 33.7 M | 4.4 ms | Apache 2.0 |
| RF-DETR Large | 56.5 | 33.9 M | 6.8 ms | Apache 2.0 |
| RF-DETR XLarge / 2XL | 58.6 / 60.1 | ~126 M | 11.5 / 17.2 ms | PML 1.0 |
| Ultralytics YOLO26 *(Jan 2026)* | — | — | — | AGPL-3.0 or paid |
| `yolov8n` *(current)* | — | ~3.2 M | — | AGPL-3.0 or paid |

RF-DETR figures are vendor-published on **NVIDIA T4, TensorRT FP16, batch 1**, not
independently reproduced. RF-DETR Large reportedly reaches 56.5 AP at 6.8 ms
against YOLOv11x at 50.9 AP at comparable latency, and is stronger on
domain-shift benchmarks — the property that matters most here, because the model
must work in a store it was not trained in. It accepts **COCO JSON or YOLO
format**, so existing labels feed it without conversion. Fine-tuning wants a CUDA
GPU with ≥ 8 GB VRAM.

> ⚠️ **"Nano" is not a like-for-like swap.** RF-DETR Nano is ~30.5 M parameters
> against `yolov8n`'s ~3.2 M — roughly ten times the model, at 2.3 ms on a T4. For
> a fixed station with a GPU that is fine and probably desirable, since nano
> capacity may well be the current limit. For a battery-powered edge device it is
> not.

Also worth settling: detection vs. instance segmentation vs. oriented boxes.
Products stack at angles. Use masks to derive boxes first; benchmark segmentation
at runtime only if overlap and counting errors justify the extra annotation and
inference cost.

## B5 · Licence — a decision, not a default

Ultralytics ships YOLO26 and everything before it under **AGPL-3.0 or a paid
Enterprise licence**. AGPL obligations attach to network-served software, and this
detector is served over HTTP to in-store stations for named supermarket chains.

| Option | Trade |
|---|---|
| Buy the Ultralytics Enterprise licence | Lowest disruption, keeps the known stack, recurring cost |
| Move to RF-DETR (Nano–Large, Apache 2.0) | No licence cost, better published accuracy, reads existing labels, new framework to learn |
| Benchmark both, licence cost as tiebreaker | Correct in principle — but only **after** B1, or the benchmark picks the wrong winner |

SAM's own licence matters far less: it runs in **annotation**, produces labels, and
never ships to a store. The output is yours. Spend the legal attention here.

## B6 · Augmentation — evidence, not settings

| Now | Should be |
|---|---|
| Offline balancing writes `__aug1` files into the dataset, **then** heavy online augmentation: `mosaic=1.0`, `mixup=0.2`, `scale=0.5`, `hsv_s=0.7`, flips off, `copy_paste=0.0` | Training-time only — the on-disk files are what created the leakage. Then re-tune for *this* task, changing one thing at a time against the baseline |

| Setting | Concern |
|---|---|
| `mixup=0.2` | Blends whole images. Poorly matched to counting overlapping products |
| `hsv_s=0.7` | Heavy saturation shift attacks colour — a primary SKU cue for packaging |
| `copy_paste=0.0` | Forgoes the one augmentation that actually synthesises occlusion |
| `imgsz=1280` | Test higher resolution specifically against small products; preserve realistic aspect ratios and match deployment preprocessing |
| epochs `50` | Choose from learning curves, not a round number |

## B7 · Experiments and versioning

| Now | Should be |
|---|---|
| Constants edited in source between runs (`NUM_CHECK_POINT = 1203` with `#TODO cambiar esto` beside it), output names reused, nothing recorded | One run = one immutable record: dataset version, config, environment, seed, weights, metrics |

Versioning and tracking solve **different** problems and both are needed:

| Tool | Job | Add it when |
|---|---|---|
| Manifest + hashes | image → video/frame → SKU/instance → review → dataset version | **Day one.** Cheapest thing here, highest value-to-effort ratio. DVC preserves versions but does not verify this chain is *correct* |
| DVC + Git | Recover the exact dataset a run used | Before the first comparison between two models, or the comparison means nothing |
| MLflow | Link runs to params, metrics, dataset version, artefacts | Same moment as DVC |

## B8 · Export and serving

| Now | Should be |
|---|---|
| Exports the in-memory model and passes a `path` argument; `best_model_path` is computed at `01_train_yolov8.py:98` and never used, so the exported artefact is probably not `best.pt` | Export the selected checkpoint explicitly, then evaluate **that exact artefact** for accuracy and latency on target hardware |
| Server loads a TF SavedModel by `detect` signature, class names from a pickle that must be exported alongside it. YOLO output has no path in. Thresholds duplicated across eight sites with four different values | One integration for the winning model — preprocessing, output format, class mapping — with the threshold defined **once** |

Export does not preserve behaviour for free. Verify preprocessing, rotation, class
ids, coordinate convention, thresholds and any quantisation effect. ONNX or
TensorRT are reasonable routes depending on hardware.

> 🔴 **Add an explicit unresolved state.** An unknown detection must surface, not
> vanish from the basket. A correct model prediction the application drops is
> still a product failure.

## B9 · Release gate

| Now | Should be |
|---|---|
| None. `98%` is a literal in the template | Promotion only when independently measured quality and latency meet agreed thresholds on held-out sessions |

The final test set is used **after** model and threshold selection. Used
continuously for tuning, it stops being a test. Review all its annotations
carefully, with a second reviewer resolving ambiguity.

---

## Tools

Grouped by job. *Verdict* is for this project specifically: one person, limited
time, ~13 SKUs today, a commercial product for named retailers, a Windows machine
with a GPU.

### Segmentation and tracking engines

Models, not interfaces. Pick an interface that hosts one.

| Engine | Gives you | Advantages | Constraints | Verdict |
|---|---|---|---|---|
| **SAM 3** *(2025-11-19)* | Promptable concept segmentation: text or exemplar prompt returns **every** matching instance with unique IDs, in images and video | Solves multi-instance natively — the fix for [A2](#a2--identify--the-one-rewrite). Open-vocabulary, so "toothbrush" works untrained. ~30 ms for 100+ objects on an H200 | Custom **SAM License**, not Apache/MIT. Commercial use permitted, not copyleft, no revenue thresholds — but modifications inherit the terms, and checkpoints are **gated** behind a Hugging Face access request. Trade-control clauses | **Primary** |
| **SAM 3.1** *(2026-03-27)* | SAM 3 plus object multiplexing — up to 16 objects per forward pass | ~Doubles video throughput (32 fps vs 16 on one H100). Sixteen objects maps onto a cart | Same licence and gating. Whether the chosen interface has adopted it yet is the practical question | **Prefer if hosted** |
| **SAM 2 / 2.1** | Point/box-guided video masks with memory propagation. One object per prompt | Mature, widely integrated — the only SAM generation CVAT's tracker officially supports today | One object per prompt means instance identity is still assigned by hand. Confirm its licence separately — SAM 3's terms were verified, SAM 2's were not | Fallback |
| **SAM 1 + XMem** *(current)* | What `sam_wrapper.py` wraps, via a Track-Anything fork | — | Two generations behind, and the wrapper carries a mask-inverting off-by-one plus a hardcoded centre-box retry | **Retire** |

### Annotation interfaces

The decision that determines how the days feel. The critical distinction for a
cart is **whether the tool propagates many objects at once, or one at a time**.

| Tool | Multi-object tracking | Advantages | Constraints | Cost |
|---|---|---|---|---|
| **CVAT** + SAM 2 Tracker | **Yes — all at once.** *Run Actions*, or `Ctrl+E` with nothing selected, applies the tracker to every visible polygon and mask | Best fit for a cart: annotate eight products on a keyframe, propagate all eight in one action. Mature review features — reference ground-truth jobs, consensus, quality dashboards. Re-run from a corrected frame | 🔴 **Community edition does not support it.** Nuclio variant Enterprise-only; AI Agent variant needs CVAT Online or Enterprise, v2.42.0+, Docker Compose on your hardware, GPU strongly recommended. Masks must be converted to polygons. Skeletons unsupported. Single agent, no concurrency; an agent crash loses tracking state | Paid tier or Enterprise |
| **X-AnyLabeling** + SAM 3 | **No — one target per session.** Docs are explicit: one target for visual prompting, one category per session for text | Free, fully local desktop, no cloud. Hosts SAM 3, so one text prompt per SKU still returns all instances of that SKU with separate IDs — most of what is needed | Eight SKUs means eight propagation sessions per clip: repetitive, though not per-item. Needs client v3.3.4+ and X-AnyLabeling-Server v0.0.4+, plus checkpoints and a BPE vocab file. ~15-frame warm-up; propagates forward from the current frame only; cancelling mid-task loses results | **Free** |
| **Roboflow** | Smart Polygon (SAM 2 one-click); Auto Label across whole datasets | Least setup of anything here, and end-to-end: annotate, train, deploy, dataset versioning built in. RF-DETR is theirs. Label Assist lets your own model pre-label the next batch — the active-learning loop | Cloud. Cart footage and catalogue leave the machine — a question to settle with the retail clients, not just internally. Video-tracking ergonomics less specialised than CVAT's | Free public tier with credits; private from ~$79/mo |
| **Label Studio** | Timeline video labelling; SAM video tracking weaker than CVAT's | Open source, free to self-host, genuinely multi-modal, plugin architecture | Not the strongest video-tracking story. More integration work — the thing being eliminated | Free self-host |
| **Supervisely** | Full-stack platform with tracking | Broad: annotation, training, deployment. Strong on 3D/LiDAR if that ever matters | Heavier than needed at this scale | Free tier; Pro from ~€199/mo |
| **The Gradio apps** *(current)* | One mask per product name — instances collapse | Fully understood by their author | Maintenance falls on this project, and the review GUI approves the wrong frame while the correction helper is wired to nothing | **Retire** |

> **The trade-off, stated plainly.** CVAT gives true multi-object propagation but
> is not free for this feature. X-AnyLabeling is free and local but needs one
> session per SKU. **Pilot X-AnyLabeling first** — free, local, hosts SAM 3 — and
> measure approved frames per hour. If per-SKU session overhead is what dominates,
> *then* the CVAT subscription has a number to justify it. Do not pay to fix a
> bottleneck that has not been measured. Data-residency obligations to the retail
> clients may make this decision instead.

### Curation, versioning, tracking

| Tool | Job | Advantages | Add it when |
|---|---|---|---|
| **FiftyOne** | Dataset inspection and curation. `compute_near_duplicates()` finds near-identical frames via embeddings; a hardness score surfaces likely annotation errors; embedding views expose clusters and failure modes | Directly answers the [A5](#a5--frame-selection--positional--informative) selective-export question and "which labels are probably wrong?". Free and open source. Also the natural place to choose the next batch to label | After the first export. Its suggestions go to **review**, never straight into labels |
| **DVC + Git** | Dataset versioning | Answers what the production model was trained on. Low ceremony, sits beside the existing Git | Before the first model comparison |
| **MLflow** | Experiment tracking | Replaces editing constants in source and hoping | With DVC |
| **Manifest + hashes** | The lineage chain | Cheapest, highest value-to-effort. DVC preserves versions but does not verify the chain is correct | Day one |
| **FFmpeg** | Clip trimming, frame extraction | Faster and batch-friendly; keeps source video and timestamps intact for lineage | Optional, low priority |

---

## Sequence

Dependency order, not appeal order. Effort figures are **estimates** for one
person who knows this codebase — planning numbers, not commitments.

| # | Step | Deliverable | Done when | Effort |
|---|---|---|---|---|
| 0 | [Rules and catalogue](#a0--rules-and-catalogue--before-any-tooling) · rotate the Azure key | SKU catalogue, annotation rules, capture plan | Written down and agreed | ½ day |
| 1 | **Stop manufacturing bad data** | Fix the review GUI frame offset and wire up `correct_box()`. Give annotations instance identity. Fix the off-by-one, add a max-area gate, fix or remove the centre-box retry. Relax the aspect-ratio and circularity filters | Every 🔴 and ⚠️ intake row above is closed | 2–4 days |
| 2 | **Make one number honest** | Group the split by session, augmentation at training time only, carve an untouched test set, repair the metrics call. Retrain `yolov8n` unchanged | A saved metrics file exists and the number is believable — expect it to **fall** | 2–3 days + compute |
| 3 | **Prove the loop on two clips** | Two short clips with repeated units and real occlusion, end to end through the new tool | Approved frames per hour is measured and clearly beats hand-labelling | 1–3 days |
| 4 | **Scale intake to ten videos** | Varied sessions, manifest written as you go, FiftyOne once review is the bottleneck | Ten sessions with full lineage | several days |
| 5 | **Lineage, then challengers** | DVC + MLflow. Then `yolov8n` baseline vs. a larger variant vs. RF-DETR Small, one change at a time. Settle the licence | A comparison on the same protocol, measured on the exported artefact | across the quarter |
| 6 | **Close the serving seam** | One integration for the winner, threshold defined once, explicit unresolved state | End-to-end predictions and quantities agree with approved test annotations | — |
| 7 | **Expand through observed failures** | Recordings targeted at missed SKUs and hard conditions | Successive dataset versions improve independently measured performance | ongoing |

> 🔴 **Do not start at step 5.** It is the most interesting step and it is
> worthless before step 2. Comparing RF-DETR against `yolov8n` on a validation set
> containing the training images produces a confident, precise, meaningless
> answer — and it will be acted on.

Steps 1 and 2 are independent of 3 and 4 and can proceed in parallel if the
review-GUI fix lands first.

---

## Decisions this plan cannot make

| # | Decision | The trade |
|---|---|---|
| 1 | **Free and repetitive, or paid and fast?** | X-AnyLabeling costs nothing and stays local but needs one session per SKU. CVAT propagates everything in one action but the feature is absent from the free edition. Pilot free, measure, let the hourly cost decide. Data-residency obligations may decide instead |
| 2 | **Buy the AGPL exemption, or move to Apache?** | Ultralytics Enterprise keeps the known stack at recurring cost. RF-DETR Nano–Large is Apache 2.0, reads existing YOLO labels, reports better accuracy — at ten times the parameter count. Needs legal input; the one item with a non-technical consequence |
| 3 | **Does the synthetic track survive?** | It trained the production model, its Azure dependency retires 2028-09-25, and its labels are pixel-exact on images that do not look like real carts. Keep only as a supplement that *measurably* helps on real held-out carts |

---

## The ceiling more data cannot raise

Three things on the target catalogue are not fully solvable by vision from one
overhead camera. Better to design around this than discover it in a store.

| Limit | Why |
|---|---|
| **Occlusion** | A cart is a pile. An item at the bottom is invisible from above, and no amount of training data recovers what the camera never captured. A physical constraint, not a model deficiency |
| **Near-identical packaging** | Whole vs. semi-skimmed milk in the same carton design; the same shampoo in 300 ml and 500 ml. May be genuinely indistinguishable from above at this resolution |
| **Produce varieties** | Fuji vs. Gala apples overlap in appearance. A lettuce is deformable with no packaging to read |

Each needs an explicit fallback rather than a confident wrong answer: a second
viewpoint, weight reconciliation, barcode where visible, a controlled placement
flow, or asking the shopper to confirm. The UI already gestures at this — there is
an *estimated weight* readout and an anti-fraud panel reading *pending scale*.
That scale is the mechanism that catches what the camera cannot see.

**The error asymmetry should set the thresholds:** a missed item costs the
retailer margin; **charging for an item the shopper does not have destroys
trust**, which is far more expensive.

> **Is any of this worth doing before the occlusion question is settled?** Yes.
> The ceiling is a real product risk but not a reason to pause — the fixes above
> are prerequisites for *measuring* it. Right now there is no way to distinguish
> "the model is weak" from "the item was invisible", because the metrics are
> contaminated and the labels omit occluded items by construction. Fix the gates,
> then measure, then the remaining error can be attributed to physics or to
> training.

---

## What is missing from this plan

1. **Nothing here has been measured on iaCarry data.** No dataset, weights,
   `label_map.pbtxt` or checkpoint exists on the machine this was written on, and
   there is no `E:` drive. Every effort figure is an estimate and every target is
   a proposal.
2. **Benchmark figures are vendor-published and not independently reproduced**,
   and COCO accuracy does not transfer to supermarket carts.
3. **SAM 2's licence was not verified** — only SAM 3's. Check it if SAM 2 becomes
   the chosen engine.
4. **Whether any interface has adopted SAM 3.1** was not established. Confirm
   before assuming the 16-object multiplexing is available.
5. **The metrics-API mismatch needs checking against the pinned Ultralytics
   version** in the environment that actually runs training, not from
   documentation.
6. **Pricing figures are as advertised** and change without notice. Confirm before
   budgeting.
7. **No cost is estimated for the physical changes** the occlusion ceiling may
   require — a scale, a second camera, or a changed placement flow. That is a
   product decision with a hardware budget, outside this plan.

---

## Sources

Checked 2026-09-09.

- [Meta AI — SAM 3.1: Faster and More Accessible Real-Time Video Detection and Tracking](https://ai.meta.com/blog/segment-anything-model-3/)
- [facebookresearch/sam3](https://github.com/facebookresearch/sam3) · [SAM License text](https://github.com/facebookresearch/sam3/blob/main/LICENSE)
- [Ultralytics Docs — SAM 3: Segment Anything with Concepts](https://docs.ultralytics.com/models/sam-3)
- [CVAT Docs — Segment Anything 2 Tracker](https://docs.cvat.ai/docs/annotation/auto-annotation/segment-anything-2-tracker/)
- [CVAT Blog — SAM2 Object Tracking via AI Agent Integration](https://www.cvat.ai/resources/blog/sam2-ai-agent-tracking)
- [X-AnyLabeling — SAM 3 interactive video object segmentation](https://github.com/CVHub520/X-AnyLabeling/blob/main/examples/interactive_video_object_segmentation/sam3/README.md) · [repository](https://github.com/CVHub520/X-AnyLabeling)
- [RF-DETR documentation — variants, benchmarks, licensing](https://rfdetr.roboflow.com/latest/) · [repository](https://github.com/roboflow/rf-detr) · [Apache 2.0 announcement](https://blog.roboflow.com/rf-detr-is-free-to-use-commercially/)
- [Ultralytics Docs — YOLO26](https://docs.ultralytics.com/models/yolo26) · [Ultralytics licensing](https://www.ultralytics.com/license)
- [Microsoft Learn — Migrate from Custom Vision Service](https://learn.microsoft.com/en-us/azure/ai-services/custom-vision-service/migration-options)
- [FiftyOne Brain — near-duplicate and mistakenness detection](https://docs.voxel51.com/brain/index.html)
- [Roboflow — annotation tool comparison, 2026](https://blog.roboflow.com/best-image-annotation-tools/) *(vendor-authored; treat comparative claims accordingly)*

Code claims were read in this repository at `f921ba7`, branch `_co_dev1`.
