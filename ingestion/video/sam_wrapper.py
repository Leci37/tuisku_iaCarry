from track_anything import TrackingAnything, parse_augment
import numpy as np
import torch
import os
import cv2


def init_model():
    import torch

    print("CUDA Available:", torch.cuda.is_available())
    print("Total CUDA Devices:", torch.cuda.device_count())

    for i in range(torch.cuda.device_count()):
        print(f"Device {i}: {torch.cuda.get_device_name(i)}")
    args = parse_augment()

    # Force use of most accurate SAM model
    args.sam_model_type = 'vit_h'  # Override whatever is passed in
    args.device = "cuda:0"
    args.mask_save = False
    args.port = 12212

    # Add these lines to attach checkpoints to args
    SAM_checkpoint_dict = {
        'vit_h': "sam_vit_h_4b8939.pth",
        'vit_l': "sam_vit_l_0b3195.pth",
        "vit_b": "sam_vit_b_01ec64.pth"
    }
    args.SAM_checkpoint = os.path.join("./checkpoints", SAM_checkpoint_dict[args.sam_model_type])
    args.xmem_checkpoint = os.path.join("./checkpoints", "XMem-s012.pth")
    args.e2fgvi_checkpoint = os.path.join("./checkpoints", "E2FGVI-HQ-CVPR22.pth")

    model = TrackingAnything(args.SAM_checkpoint, args.xmem_checkpoint, args.e2fgvi_checkpoint, args)

    return model

def apply_mask_postprocessing(mask, min_area=500):
    """
    Post-process the SAM mask:
    - Keep only the largest connected component.
    - Apply morphological closing to clean small holes/gaps.

    Args:
        mask (np.ndarray): Raw binary mask (H, W).
        min_area (int): Minimum area threshold to keep a component.

    Returns:
        np.ndarray: Cleaned binary mask.
    """
    mask = mask.astype(np.uint8)

    # Find all connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    # Skip background (label 0)
    areas = stats[1:, cv2.CC_STAT_AREA]
    labels = labels + 1  # To align with mask labeling starting from 1

    # If no valid regions found, return empty mask
    if len(areas) == 0:
        return np.zeros_like(mask)

    # Keep the largest component above area threshold
    largest_idx = np.argmax(areas)
    if areas[largest_idx] < min_area:
        return np.zeros_like(mask)

    cleaned_mask = (labels == (largest_idx + 1)).astype(np.uint8)

    # Morphological closing to clean edges
    kernel = np.ones((5, 5), np.uint8)
    closed = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel)

    return closed


def segment_objects(model, image, object_names):
    """
    Use box prompts instead of point prompts for SAM segmentation.
    Applies post-processing: largest component filtering + morphology cleanup.

    Args:
        model: Initialized model with SAM controller.
        image: Input RGB frame.
        object_names: List of label strings to segment.

    Returns:
        dict: {label_name: binary_mask}
    """
    model.samcontroler.sam_controler.reset_image()
    model.samcontroler.sam_controler.set_image(image)

    h, w = image.shape[:2]
    masks = {}

    for i, name in enumerate(object_names):
        # Simple center-region box (can be replaced with detector output)
        box = np.array([[w // 4, h // 4, 3 * w // 4, 3 * h // 4]])

        try:
            mask, _, _ = model.first_frame_box(
                image=image,
                boxes=box,
                multimask=True
            )
        except Exception as e:
            print(f"[ERROR] SAM failed on label '{name}': {e}")
            continue

        # Post-process mask to filter noise
        cleaned = apply_mask_postprocessing(mask)
        masks[name] = cleaned

    return masks





def track_with_mask_refinement(model, frames, initial_mask, min_area=500, retry=True):
    """
    Perform tracking with optional mask quality checking and fallback to SAM-based refinement.

    Args:
        model: Initialized TrackingAnything model
        frames: List of RGB frames
        initial_mask: Binary mask from frame 0
        min_area: Minimum valid area for tracking masks
        retry: If True, re-run SAM if mask quality drops

    Returns:
        List of tracked masks and list of painted frames
    """
    model.xmem.clear_memory()

    masks, logits, painted = model.generator(images=frames, template_mask=initial_mask)

    # Smooth or refine
    for i in range(1, len(masks)):
        mask = masks[i]
        area = np.sum(mask > 0)
        if area < min_area and retry:
            print(f"[WARN] Low mask area at frame {i}, retrying SAM refinement")
            model.samcontroler.sam_controler.reset_image()
            model.samcontroler.sam_controler.set_image(frames[i])
            h, w = frames[i].shape[:2]
            box = np.array([[w // 4, h // 4, 3 * w // 4, 3 * h // 4]])
            try:
                mask_retry, _, _ = model.first_frame_box(
                    image=frames[i],
                    boxes=box,
                    multimask=True
                )
                masks[i] = apply_mask_postprocessing(mask_retry, min_area)
            except Exception as e:
                print(f"[ERROR] Retry SAM failed at frame {i}: {e}")

    model.xmem.clear_memory()
    return masks, painted
