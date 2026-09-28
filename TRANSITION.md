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
or (b) checked against a vendor or primary source on 2026-09-09 and again on
2026-09-28, listed in [Sources](#sources). Nothing was measured on iaCarry data —
no dataset, weights, `label_map.pbtxt` or checkpoint exists on the machine this
was written on.

---

## Second review — 2026-09-28

Every code claim was re-read at `0738725` and every tool claim re-checked against
a primary source. **Verdict: viable.** Click once, propagate, export many boxes
is the standard way to label video today, and the tools named here are the right
family. What changed:

1. **SAM 3 finds instances, not SKUs.** Its prompts are simple noun phrases
   (`can`, `box`), and its own paper says it struggles with fine-grained concepts
   zero-shot. Six of the 14 catalogue SKUs are 33 cl cans. A person names the SKU
   of each tracked instance, once per clip — see [A2](#a2--identify--the-one-rewrite).
2. **Pilot before repairing.** The first sequence spent 2–4 days fixing the Gradio
   apps and the SAM 1 wrapper, which A3 and A6 then retire. The new
   [Sequence](#sequence) freezes the old intake, pilots the new one, and repairs
   only what survives.
3. **The YOLO script dies at export, not at metrics.** `path=` is not an
   Ultralytics argument, so line 101 raises as soon as training ends. The weights
   it tried to export *were* `best.pt` — Ultralytics reloads it after `train()`.
4. **Four defects the first review missed, and ten of its rows corrected or
   extended** — see [Confirmed defects](#confirmed-defects). The phase READMEs
   were corrected where they described behaviour the code does not have.
5. **A step the plan lacked:** once a model is good enough, it pre-labels new
   video and people correct it. That loop, not the tracker, removes most
   labelling in the long run.

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

And one thing changed in the wider world that solves the hard half of the
deepest design flaw:

> ✅ **SAM 3 returns every instance of a concept at once, each with its own ID.**
> SAM 1 and SAM 2 predict one object per prompt, and the labelling GUI keys each
> prompt by product *name* (`02_label_gui.py:64`) — so two identical products
> collapse into one. Prompt `can` once and every can comes back separately.
> Instance separation, the root of the counting bug, now comes from the model;
> the schema that stores it ([A2](#a2--identify--the-one-rewrite)) is still yours
> to write. SAM 3 shipped 2025-11-19; SAM 3.1 followed 2026-03-27 and tracks
> objects in buckets of **16 per forward pass** — a speed gain, not a cap. A cart
> holds roughly 8–15 items.
>
> ⚠️ **What SAM 3 does not do is tell SKUs apart.** A concept prompt is a simple
> noun phrase, and the SAM 3 paper states the model struggles with fine-grained
> concepts zero-shot. `can` returns the Coca-Cola, the Fanta and both Mahous
> alike. Whether a brand-level prompt such as `Mahou 5 Estrellas can` separates
> them is untested. Plan on a person naming the SKU of each tracked ID, and let
> the pilot show whether prompts can take that over.

---

## Confirmed defects

Ranked by damage to final model quality. Every row was read in the source on
`_co_dev1`, and re-read at `0738725` for the second review; rows marked *(2nd)*
were added or corrected then. These are the reason for the transition, not a
wish list.

| Sev | Defect | Where | Effect |
|---|---|---|---|
| 🔴 | Masks keyed by product label — no instance identity *(2nd)* | `ingestion/video/02_label_gui.py:49-64`, `label_gui_utils.py:165` | Clicks accumulate per label into **one point prompt**, and the result replaces that label's mask. A click on a second unit of the same product joins the first unit's prompt — one mask, one `.npy` per label. Downstream keeps only the largest connected component (`track_masks_utils.py:154`). Quantity is unrepresentable — while the checkout screen displays `✓ × N` badges |
| 🔴 | Review GUI approves the wrong frame *(2nd)* | `ingestion/video/04_bbox_review.py:128`, `:187` | `accept_frame()` loads the *next* frame, saves it, marks it accepted, *then* displays it. `demo.load(fn=accept_frame)` auto-accepts frame 1 sight-unseen. After any Accept the frame on screen is **already saved**: Accept commits the frame after it, and Discard skips one frame and marks the one after that — the frame judged bad stays in the dataset. Only Undo removes the frame on screen, and it then marks the next one discarded unseen |
| 🔴 | Augmented copies split across train and val | `ingestion/video/06_rotate_stats_balance.py:131`, `training/yolo/01_train_yolov8.py:30` | `X__aug1.png` is written beside `X.png`; the trainer then splits basenames at random. The same photograph, colour-jittered, lands on both sides |
| 🔴 | Near-duplicate frames split across train and val | `training/yolo/01_train_yolov8.py:30` | Frames pooled from all clips and split individually. Frames 4 and 9 of one clip — same cart, 5/30 s apart — routinely land on opposite sides |
| ⚠️ | A lost product is dropped from a kept frame *(2nd)* | `ingestion/video/03_track_masks.py:108-110` | An empty mask is read as "not in view": the label is skipped and the frame is saved. When the tracker has lost a product that is still visible, the frame goes into the dataset with that product unboxed, teaching the model to ignore it. Nothing tells the two cases apart, and review shows only the boxes that exist |
| ⚠️ | Mask post-processing selects the background *(2nd)* | `ingestion/video/sam_wrapper.py:58`, `:69` | `labels` is shifted `+1`, then component `largest_idx + 1` is selected. For a single foreground blob that expression resolves to the background. The mask is inverted. Reached only through the tracker retry (`:155`); the click path stores SAM's mask untouched |
| ⚠️ | Inverted masks pass every quality gate *(2nd)* | `ingestion/video/track_masks_utils.py:89` | There is a `min_area` and **no maximum**. A near-full-frame mask of a 16:9 frame has aspect ratio 1.78, solidity ≈ 1.0, circularity ≈ 0.72, one external contour — it clears all five filters. The retry writes a 0/1 mask, so it lands on the product tagged first (id 1, `03_track_masks.py:104`): a whole-image box labelled as that product |
| ⚠️ | Tracker retry re-prompts with a hardcoded centre box *(2nd)* | `ingestion/video/sam_wrapper.py:142`, `:148` | Fires when the **combined** mask area of all tracked products drops below 500 px — near-total tracking loss, or an empty view — and re-prompts SAM with the middle 50 % of the frame, not the object. Live, via `03_track_masks.py:47` |
| ⚠️ | Geometry filters reject the target catalogue *(2nd)* | `ingestion/video/track_masks_utils.py:89` | `max_aspect_ratio=4.0`, `min_circularity=0.15`. Toothbrushes, toothpaste tubes edge-on, spaghetti boxes, thin visible slivers of occluded items — all systematically discarded. One failing product discards the **whole frame** (`03_track_masks.py:111-115`), taking every other product's label with it |
| ⚠️ | Image and label orientation can disagree *(2nd)* | `ingestion/video/bbox_review_utils.py:449-458`, `:467` | Review rotates the *pixels* by the clip's orientation metadata (read through the Windows Shell, `label_tools.py:200`), with filename exceptions for `_n2 (` clips, and rescales the boxes without rotating them. The boxes were computed in step 03, which reads frames with no rotation handling. For a clip review rotates, image and label agree only if the OpenCV that ran step 03 auto-rotated frames and the one that ran step 04 did not. Nothing checks it |
| ⚠️ | No held-out test set | `training/yolo/01_train_yolov8.py:122` | Train/val only, then `val_ids[:5]` reused as "test predictions". No honest final number even before the leakage |
| ⚠️ | No correction path in review *(2nd)* | `ingestion/video/bbox_review_utils.py:163`, `:258` | `correct_box()` exists and is wired to no control — it is an OpenCV `imshow` loop, so it could not run inside the Gradio page as written. `draw_key_legend()` advertises `s`/`c`/`d`/`q` keys that nothing binds; only ← and → work. A nearly-right box must be discarded |
| ⚠️ | The YOLO script stops at export *(2nd)* | `training/yolo/01_train_yolov8.py:101`, `:106` | `model.export(..., path=…)`: `path` is not an Ultralytics argument, so the call raises `SyntaxError` the moment training ends — nothing is exported, and the metrics block and sample predictions are never reached. Past that, `results.metrics` would raise `AttributeError`: `train()` returns a `DetMetrics` whose values live under `.box` (`.box.map50`, `.box.map`, `.box.mp`, `.box.mr`). Both checked in the Ultralytics 8.0.200 and 8.4.164 source. The trainer has already written `best.pt`, `results.csv` and its validation plots by then |
| ⚠️ | Unknown classes dropped from the basket | `serving/FLOW.md:330` | A product the model names but the front-end catalogue lacks is discarded and never charged — classified as "allowed". A correct prediction the application throws away is still a product failure |
| 🔴 | Azure keys in plaintext, public repo *(2nd)* | `ingestion/synthetic/05b_azure_upload_augmented.py:14`, `06_azure_download_coco.py:22-23` | Two keys: the Custom Vision training key, in both files, and a second key at `06:23` whose comment links to the Keys page of an Azure AI Search service, beside the subscription id. Rotate both — deleting them from the source leaves them in Git history. The only item here with a deadline set by someone else |
| ℹ️ | "98 % accuracy" is a literal *(2nd)* | `serving/templates/iacarry_checkout.html:194` | Hardcoded, as is "1.2s inference". Shown to customers. Measure it or remove it. (The first review said `README.md` quotes it; it never has) |
| ℹ️ | A third class ordering, hardcoded *(2nd)* | `ingestion/video/bbox_review_utils.py:29-44`, `:231` | The review step's `coco_annotations.json` takes category **ids** from the YOLO labels but category **names** from a hardcoded 14-SKU list. If that order differs from `classes.txt`, every name in the file is wrong. The YOLO labels are unaffected — feed trainers that read COCO, RF-DETR included, from those |
| ℹ️ | Step 03 cannot start from a clean clone *(2nd)* | `ingestion/video/track_masks_utils.py:12` | Imports `parse_label_map` from `utils_`, which does not exist — it lives in `label_tools.py`. `tools/check_imports.py` takes unknown names for third-party packages, so it reports the import as resolved |
| ✅ | The labelling GUI itself is **not** at fault | `ingestion/video/02_label_gui.py:57` | Clicks *are* passed to SAM correctly via `first_frame_click`. The broken centre-box `segment_objects()` is dead code, reachable only from `legacy/` |

> ⚠️ **The phase READMEs were reliable on structure and unreliable on
> behaviour.** The second review corrected them where they described behaviour
> the code does not have: post-processing in labelling, box correction in
> review, the orientation step 05 produces, and how far the YOLO script gets.
> Treat any behavioural claim not tied to a `file:line` as unverified.

---

# Part A — Intake

Raw video → a dataset you can trust. The workflow shape does not change. Almost
every step below is a repair or a tool swap; exactly one is a rewrite.

## The target loop

```
YOU                                         TOOL
① prompt a concept — "can", "box" —   ──►  ② every instance found, each with
  or draw one exemplar box                    its own ID (SAM 3)
③ name the SKU of each ID,            ◄──
  once per clip                       ──►  ④ propagate every ID through the clip
⑤ correct at the FIRST bad frame      ◄──
                                      ──►  ⑥ re-propagate that span only
                                           ⑦ export diverse frames → boxes
⑧ review the export — tens,           ◄──
  not thousands
```

Step ③ is the one the first version of this loop left out: SAM 3 separates
instances, it does not name SKUs (see [A2](#a2--identify--the-one-rewrite)).

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

**Effort:** half a day *(estimate)*. Also rotate both Azure keys here.

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

**Where SAM 3 fits, and where it stops.** A concept prompt (`can`, `bottle`,
`box`) or one exemplar box drawn round a single unit returns every matching
instance with its own ID. That fills the **instance** field. The **SKU** field is
a person's call: name each ID once per clip, and the name holds for every frame
that ID is tracked through. Six of the 14 SKUs are 33 cl cans (`aguila`,
`cocacola`, `estrellag`, `fanta`, `mahou00`, `mahou5`), and the catalogue will
grow — the fine-grained case the SAM 3 paper names as its weakness. Test
brand-level prompts in the pilot; do not build the workflow on them.

## A3 · Propagate — swap the engine

| Now | Should be |
|---|---|
| SAM 1 ViT-H + XMem through a Track-Anything fork. When the combined mask area collapses it re-prompts with a hardcoded centre box (`sam_wrapper.py:148`) and then inverts the mask (`:58`, `:69`) | A promptable video segmenter with native memory-based propagation, driven from an annotation tool where one fits, a thin script over SAM 3 where it does not — never again a bespoke tracking stack. Handle entry, exit and reappearance explicitly. Correct at the first bad frame and re-propagate that interval |

`ingestion/video/sam_wrapper.py` and `03_track_masks.py` are the files this step
retires. Two generations behind, and both defects above are live.

Two free ways to drive SAM 3 today: from an annotation tool (see
[Annotation interfaces](#annotation-interfaces)), or from a short script —
Ultralytics' `SAM3VideoSemanticPredictor` takes several concept prompts in one
pass. The official `sam3` package asks for Python 3.12+, PyTorch 2.7+ and
CUDA 12.6+, and documents a Linux setup; on the Windows workstation, budget for
WSL2 or a Linux GPU box and confirm it in the pilot. (Today's intake is the
opposite — Windows-only, through `win32com` at `label_tools.py:196`.) The
checkpoint is 3.45 GB (848 M parameters), and published timings are on
data-centre GPUs.

## A4 · Quality filtering — repair, do not delete

| Now | Should be |
|---|---|
| Generic geometry gates: `min_area=300`, `max_aspect_ratio=4.0`, `min_circularity=0.15`, `min_solidity=0.10`, `max_components=4`. No maximum area. A frame failing the check is dropped whole. An empty mask drops the product and keeps the frame | Filters appropriate to product geometry — a toothbrush is legitimately long and thin. **Add a maximum-area gate** so an inverted mask cannot pass. Route suspicious masks to human review rather than deleting them silently. Flag **missing** instances, the error class the current gates cannot see at all |

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

## A6 · Human validation — freeze, then swap

| Now | Should be |
|---|---|
| A Gradio app that saves and marks accepted the frame it is *about to show*. Loading the page accepts frame 1. After any Accept, the frame on screen is already saved, so Discard cannot remove it. No correction control. Accept or discard only | **Do not approve data with the current app.** If it survives the [pilot](#sequence), the first fix is that the decision applies to the frame **on screen**. Then: accept, correct, reject, **and uncertain**; an explicit "is every visible in-scope product labelled?" check; recorded reviewer identity and timestamp |

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
| YOLO and COCO files across hand-configured folders, with an undocumented rename between steps 03 and 04. `classes.txt` read from two different directories that must not disagree — and one of them, `gui_04_bbox_clean/`, is written by no step. A third, hardcoded ordering names the COCO categories | One versioned dataset release with a manifest carrying, per image: source video, frame index, instance and SKU ids, annotation origin (proposed vs. approved), reviewer decision, content hash |

Store training frames **clean** — no painted masks, no drawn boxes. Keep overlays
separately. (Today's review step already does this: it re-reads frames from the
source video.) Validate that every exported label matches its image's actual
orientation and dimensions — today it may not (see the orientation row in
[Confirmed defects](#confirmed-defects)).

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
> have a transition plan in place by **2026-09-25**. That date passed three days
> before the second review; the service itself keeps running until retirement.
> The dependency was going to be retired anyway; this makes it scheduled.

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
| A print block the script never reaches — it stops at the export on line 101 — and that would raise `AttributeError` if it did, with hand-written `ideal >` thresholds beside each line. The trainer's own `results.csv`, PR curves and confusion matrix *are* written to the run folder, but scored on the leaky split. Track A evaluates by eye on JPEGs. None of it is in the repo | Saved, versioned metrics. Ultralytics already provides mAP, PR curves and confusion matrices — read them from `results.box`, keep the run folder beside the dataset version. No new architecture is needed to start measuring |

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
| Ultralytics YOLO26 n / s / m *(Jan 2026)* | 40.9 / 48.6 / 53.1 | 2.4 / 9.5 / 20.4 M | 1.7 / 2.5 / 4.7 ms | AGPL-3.0 or paid |
| `yolov8n` *(current)* | 37.3 | 3.2 M | — | AGPL-3.0 or paid |

RF-DETR and YOLO26 figures are vendor-published on **NVIDIA T4, TensorRT FP16,
batch 1**, not independently reproduced; `yolov8n`'s come from Ultralytics' own
table. YOLO26n is the only row with a published CPU figure — 38.9 ms through
ONNX — which matters if a store station has no GPU.

RF-DETR Large reportedly reaches 56.5 AP at 6.8 ms against YOLOv11x at 50.9 AP
at comparable latency, and is stronger on domain-shift benchmarks — the property
that matters most here, because the model must work in a store it was not
trained in. It accepts **COCO JSON or YOLO format**, so existing labels feed it
without conversion — use the YOLO labels, not the review step's COCO file (see
the class-ordering row in [Confirmed defects](#confirmed-defects)). Fine-tuning
wants a CUDA GPU with ≥ 8 GB VRAM.

> ⚠️ **"Nano" is not a like-for-like swap.** RF-DETR Nano is ~30.5 M parameters
> against `yolov8n`'s ~3.2 M — roughly ten times the model, at 2.3 ms on a T4. For
> a fixed station with a GPU that is fine and probably desirable, since nano
> capacity may well be the current limit. For a battery-powered edge device it is
> not.

On custom data the gaps shrink. A fine-tuning comparison published by JetBrains
in August 2026, on Roboflow's RF100-VL datasets, found no family winning
everywhere; on its soda-bottle set, five of six models — RF-DETR, YOLOv12 and
YOLO26 variants — finished within two mAP points of each other (0.622–0.642).
Vendor-adjacent, and not carts, but the direction is clear: expect licence and
target hardware, more than accuracy, to decide, and measure on held-out sessions.

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
never ships to a store. The output is yours. The same holds for Ultralytics' SAM 3
wrapper if the scripted route is used — AGPL, but internal and never shipped.
Spend the legal attention here.

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
| Passes `path=` to `model.export()`, which Ultralytics rejects, so nothing is exported. (The in-memory model *would* have been `best.pt` — `train()` reloads it; `best_model_path` at `01_train_yolov8.py:98` is simply unused) | Export the selected checkpoint explicitly — `YOLO("…/best.pt").export(format=…)` — then evaluate **that exact artefact** for accuracy and latency on target hardware |
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
time, 14 SKUs today (`bbox_review_utils.py:9`), a commercial product for named
retailers, a Windows machine with a GPU.

### Segmentation and tracking engines

Models, not interfaces. Pick an interface that hosts one.

| Engine | Gives you | Advantages | Constraints | Verdict |
|---|---|---|---|---|
| **SAM 3** *(2025-11-19)* | Promptable concept segmentation: text or exemplar prompt returns **every** matching instance with unique IDs, in images and video | Solves multi-instance natively — the fix for [A2](#a2--identify--the-one-rewrite). Open-vocabulary, so "toothbrush" works untrained. ~30 ms for 100+ objects on an H200 | Prompts are simple noun phrases, and its paper says it struggles with fine-grained concepts zero-shot — it separates cans, it does not name them. 848 M parameters, 3.45 GB checkpoint; official setup Python 3.12+, PyTorch 2.7+, CUDA 12.6+, Linux-style. Custom **SAM License**, not Apache/MIT. Commercial use permitted, not copyleft, no revenue thresholds — but modifications inherit the terms, and checkpoints are **gated** behind a Hugging Face access request. Trade-control clauses | **Primary** — for instances |
| **SAM 3.1** *(2026-03-27)* | SAM 3 plus object multiplexing — 16 objects per forward pass, more in further buckets | ~Doubles video throughput (32 fps vs 16 on one H100). Sixteen objects maps onto a cart | Same licence and gating. On 2026-09-28 only the official `sam3` code runs it on video: Ultralytics' video predictors still need `sam3.pt`, CVAT has no SAM 3 video at all, and X-AnyLabeling's support is unconfirmed | **Prefer if hosted** |
| **SAM 2 / 2.1** | Point/box-guided video masks with memory propagation. One object per prompt | Mature, widely integrated — the only SAM generation CVAT's tracker officially supports today | One object per prompt means instance identity is still assigned by hand. Confirm its licence separately — SAM 3's terms were verified, SAM 2's were not | Fallback |
| **SAM 1 + XMem** *(current)* | What `sam_wrapper.py` wraps, via a Track-Anything fork | — | Two generations behind, and the wrapper carries a mask-inverting off-by-one plus a hardcoded centre-box retry | **Retire** |

### Annotation interfaces

The decision that determines how the days feel. The critical distinction for a
cart is **whether the tool propagates many objects at once, or one at a time**.

| Tool | Multi-object tracking | Advantages | Constraints | Cost |
|---|---|---|---|---|
| **CVAT** + SAM 2 Tracker | **Yes — all at once.** *Run Actions*, or `Ctrl+E` with nothing selected, applies the tracker to every visible polygon and mask | Best fit for a cart: annotate eight products on a keyframe, propagate all eight in one action. Mature review features — reference ground-truth jobs, consensus, quality dashboards. Re-run from a corrected frame | 🔴 **Community edition does not support it.** Nuclio variant Enterprise-only; AI Agent variant needs CVAT Online or Enterprise, v2.42.0+, Docker Compose on your hardware, GPU strongly recommended. Masks must be converted to polygons. Skeletons unsupported. Single agent, no concurrency; an agent crash loses tracking state. SAM 3 is in CVAT for **images** only (visual prompts Jan 2026, label-text prompts Mar 2026); SAM 3 video tracking is announced, not shipped — the tracker is still SAM 2, one object per prompt | Paid tier or Enterprise |
| **X-AnyLabeling** + SAM 3 | **No — one target per session.** Docs are explicit: one target for visual prompting, one category per session for text | Free, fully local desktop, no cloud. Hosts SAM 3, so one concept prompt still returns every instance with separate IDs — most of what is needed | A prompt per SKU relies on SAM 3 telling brands apart (see above); a prompt per concept (`can`) returns all cans under one label, and each ID still needs its SKU. **Check in the pilot whether one edit renames an ID across all its frames** — if not, the per-ID step becomes per-frame. Needs client v3.3.4+ and X-AnyLabeling-Server v0.0.4+, plus checkpoints and a BPE vocab file. ~15-frame warm-up; propagates forward from the current frame only; cancelling mid-task loses results | **Free** |
| **A script** on `sam3` or Ultralytics' `SAM3VideoSemanticPredictor` | **Yes** — several concept prompts in one pass, each instance with its own ID (Ultralytics API; in the official repo, several text prompts in one session is an open issue) | Free, local, no per-concept sessions. Writes straight into the A2/A8 manifest schema | The review UI is yours to build or borrow (FiftyOne, X-AnyLabeling) — a smaller share of the maintenance being retired. Ultralytics' wrapper does not yet run SAM 3.1 on video | **Free** |
| **Roboflow** | Smart Polygon (SAM 2 one-click); Auto Label across whole datasets | Least setup of anything here, and end-to-end: annotate, train, deploy, dataset versioning built in. RF-DETR is theirs. Label Assist lets your own model pre-label the next batch — the active-learning loop | Cloud. Cart footage and catalogue leave the machine — a question to settle with the retail clients, not just internally. Video-tracking ergonomics less specialised than CVAT's | Free public tier with credits; private from ~$79/mo |
| **Label Studio** | Timeline video labelling; SAM video tracking weaker than CVAT's | Open source, free to self-host, genuinely multi-modal, plugin architecture | Not the strongest video-tracking story. More integration work — the thing being eliminated | Free self-host |
| **Supervisely** | Full-stack platform with tracking | Broad: annotation, training, deployment. Strong on 3D/LiDAR if that ever matters | Heavier than needed at this scale | Free tier; Pro from ~€199/mo |
| **The Gradio apps** *(current)* | One mask per product name — instances collapse | Fully understood by their author | Maintenance falls on this project, and the review GUI approves the wrong frame while the correction helper is wired to nothing | **Retire** |

> **The trade-off, stated plainly.** CVAT gives true multi-object propagation but
> is not free for this feature, and its tracker is still SAM 2. X-AnyLabeling is
> free and local but needs one session per concept. **Pilot X-AnyLabeling first**
> — free, local, hosts SAM 3 — and measure approved frames per hour. If session
> overhead is what dominates, try **the scripted route** next: also free, and it
> propagates every concept in one pass. CVAT earns a subscription only once its
> SAM 3 video tracking ships, or if SAM 2's one-object-per-prompt proves fast
> enough on a cart. Do not pay to fix a bottleneck that has not been measured.
> Data-residency obligations to the retail clients may make this decision
> instead.

### Curation, versioning, tracking

| Tool | Job | Advantages | Add it when |
|---|---|---|---|
| **FiftyOne** | Dataset inspection and curation. `compute_near_duplicates()` finds near-identical frames via embeddings; a hardness score surfaces likely annotation errors; embedding views expose clusters and failure modes | Directly answers the [A5](#a5--frame-selection--positional--informative) selective-export question and "which labels are probably wrong?". Free and open source. Also the natural place to choose the next batch to label | After the first export. Its suggestions go to **review**, never straight into labels |
| **DVC + Git** | Dataset versioning | Answers what the production model was trained on. Low ceremony, sits beside the existing Git | Before the first model comparison |
| **MLflow** | Experiment tracking | Replaces editing constants in source and hoping | With DVC |
| **Manifest + hashes** | The lineage chain | Cheapest, highest value-to-effort. DVC preserves versions but does not verify the chain is correct | Day one |
| **FFmpeg** | Clip trimming, frame extraction | Faster and batch-friendly; keeps source video and timestamps intact for lineage | Optional, low priority |
| **Your own detector, as pre-labeller** | Proposes boxes and SKUs on new video; people correct | The step that removes most of the remaining labelling: later batches are corrected, not created. Also names SKUs, which SAM 3 cannot. Roboflow's Label Assist is the hosted version | Once a model measured on held-out sessions makes correcting faster than creating — compare approved frames per hour both ways. Proposals go to **review**, never straight into labels |

---

## Sequence

Dependency order, not appeal order. Effort figures are **estimates** for one
person who knows this codebase — planning numbers, not commitments.

| # | Step | Deliverable | Done when | Effort |
|---|---|---|---|---|
| 0 | [Rules and catalogue](#a0--rules-and-catalogue--before-any-tooling) · rotate **both** Azure keys | SKU catalogue, annotation rules, capture plan | Written down and agreed | ½ day |
| 1 | **Freeze the old intake, pilot the new one** | Stop producing data with the Gradio apps and SAM 1 + XMem — free, and it stops the damage at once. Run two short clips with repeated units and real occlusion through SAM 3 (X-AnyLabeling, or the scripted route). Answer four questions: are instances separated? can prompts name SKUs, or must a person? approved frames per hour? does it run on this workstation? | All four answered with numbers, and one intake tool chosen | 1–3 days |
| 2 | **Make one number honest** | Group the split by session, augmentation at training time only, carve an untouched test set. Fix the export call and read metrics from `results.box`. Retrain `yolov8n` unchanged | A saved metrics file exists and the number is believable — expect it to **fall** | 2–3 days + compute |
| 3 | **Repair only what survives** | Engine-independent gates on the chosen path: max-area gate, product-appropriate geometry, missing-instance warnings, an image/label orientation check. The Gradio and SAM 1 fixes — frame offset, `correct_box()`, off-by-one, centre-box retry — only if the pilot sends you back to them | No 🔴 or ⚠️ intake row applies to the chosen path | 1–4 days |
| 4 | **Scale intake to ten sessions** | Varied sessions, manifest written as you go, FiftyOne once review is the bottleneck | Ten sessions with full lineage | several days |
| 5 | **Lineage, then challengers** | DVC + MLflow. Then `yolov8n` baseline vs. YOLO26 vs. RF-DETR Small, one change at a time. Settle the licence | A comparison on the same protocol, measured on the exported artefact | across the quarter |
| 6 | **Pre-label with your own model** | The best model proposes boxes and SKUs on new video; people correct, and the corrections feed the next dataset version | Approved frames per hour clearly beats step 1's figure | ongoing, from the first good model |
| 7 | **Close the serving seam** | One integration for the winner, threshold defined once, explicit unresolved state | End-to-end predictions and quantities agree with approved test annotations | — |
| 8 | **Expand through observed failures** | Recordings targeted at missed SKUs and hard conditions | Successive dataset versions improve independently measured performance | ongoing |

> 🔴 **Do not start at step 5.** It is the most interesting step and it is
> worthless before step 2. Comparing RF-DETR against `yolov8n` on a validation set
> containing the training images produces a confident, precise, meaningless
> answer — and it will be acted on.

Steps 1 and 2 are independent and can run in parallel: the pilot needs new
clips, the honest number needs only the existing labels and the training script.
Step 2 scores the existing labels, intake defects included — its number is
honest about the split, not about the labels.

> **Why this order changed.** The first version repaired the Gradio apps and the
> SAM 1 wrapper (2–4 days) before the pilot, although A3 and A6 retire both.
> Freezing costs nothing and stops bad data just as surely; the pilot then
> decides which repairs are worth making.

---

## Decisions this plan cannot make

| # | Decision | The trade |
|---|---|---|
| 1 | **Free and repetitive, or paid and fast?** | X-AnyLabeling costs nothing and stays local but needs one session per concept. A script on SAM 3 is free and propagates every concept at once, but its review UI is yours to maintain. CVAT propagates everything in one action but the feature is absent from the free edition, and its tracker is SAM 2 until SAM 3 video ships there. Pilot free, measure, let the hourly cost decide. Data-residency obligations may decide instead |
| 2 | **Buy the AGPL exemption, or move to Apache?** | Ultralytics Enterprise keeps the known stack at recurring cost. RF-DETR Nano–Large is Apache 2.0, reads existing YOLO labels, reports better accuracy — at ten times the parameter count, and without YOLO26n's CPU option. Needs legal input; the one item with a non-technical consequence |
| 3 | **Does the synthetic track survive?** | It trained the production model, its Azure dependency retires 2028-09-25, and its labels are pixel-exact on images that do not look like real carts. Keep only as a supplement that *measurably* helps on real held-out carts |
| 4 | **One detector, or locate then identify?** | One detector with a class per SKU is simplest and fastest, and retrains for every new SKU. A class-agnostic "product" detector plus an SKU classifier matches what SAM 3 already produces, adds a SKU with reference crops rather than a retrain, and can give near-identical packaging its own high-resolution check — at the cost of a second model. Decide from step 5's per-SKU confusion matrix, not before; the A2 schema serves both |

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
   there is no `E:` drive — still true at the second review. Every effort figure
   is an estimate and every target is a proposal.
2. **Benchmark figures are vendor-published and not independently reproduced**,
   and COCO accuracy does not transfer to supermarket carts.
3. **SAM 2's licence was not verified** — only SAM 3's. Check it if SAM 2 becomes
   the chosen engine.
4. **SAM 3.1 on video is available only from the official `sam3` code** as of
   2026-09-28 — not in Ultralytics' video predictors, not in CVAT. Whether
   X-AnyLabeling hosts it was not established.
5. **The Ultralytics API claims were checked in the 8.0.200 and 8.4.164 source**,
   not run. `requirements.txt` pins no Ultralytics version; confirm against the
   one in the environment that actually trains.
6. **Whether SAM 3 can tell the 14 SKUs apart by name was not tested**, and
   neither was SAM 3 on Windows. The pilot answers both.
7. **Pricing figures are as advertised** and change without notice. Confirm before
   budgeting.
8. **No cost is estimated for the physical changes** the occlusion ceiling may
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

Added at the second review, checked 2026-09-28.

- [SAM 3: Segment Anything with Concepts — paper](https://arxiv.org/abs/2511.16719) · limitations, Appendix B · [facebookresearch/sam3 issue #206](https://github.com/facebookresearch/sam3/issues/206) *(several text prompts in one video session)*
- [LearnOpenCV — SAM 3.1 Object Multiplex](https://learnopencv.com/sam-3-whats-new/)
- [CVAT — SAM 3, Part 1: image segmentation](https://www.cvat.ai/resources/changelog/sam-3-image-segmentation) · [Part 2: label-based text prompts](https://www.cvat.ai/resources/changelog/sam3-text-prompts)
- [Ultralytics Docs — YOLO26 performance table](https://docs.ultralytics.com/models/yolo26)
- [JetBrains — Fine-tuning SOTA object detection models on real-world datasets, Aug 2026](https://blog.jetbrains.com/pycharm/2026/08/fine-tuning-sota-object-detection-models-on-real-world-datasets/) *(Roboflow datasets and tooling; vendor-adjacent)*
- Ultralytics source, [8.0.200](https://pypi.org/project/ultralytics/8.0.200/) and [8.4.164](https://pypi.org/project/ultralytics/8.4.164/) from PyPI — `engine/model.py` (`train()` reloads `best.pt`), `cfg/__init__.py` (`check_dict_alignment` rejects unknown arguments), `utils/metrics.py` (`DetMetrics`)

Code claims were read in this repository at `f921ba7`, and re-read at `0738725`
for the second review, branch `_co_dev1`.
