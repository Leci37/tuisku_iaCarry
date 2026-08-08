from track_anything import TrackingAnything, parse_augment
import numpy as np
import torch
import os

def init_model():
    args = parse_augment()
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

    model = TrackingAnything(args.SAM_checkpoint, args.xmem_checkpoint, args.e2fgvi_checkpoint , args)

    return model

def segment_objects(model, image, object_names):
    # This sets the first frame in SAM controller
    model.samcontroler.sam_controler.reset_image()
    model.samcontroler.sam_controler.set_image(image)

    masks = {}
    for i, name in enumerate(object_names):
        h, w = image.shape[:2]
        point = np.array([[w // 2, h // 2]])
        labels = np.array([1])
        mask, _, _ = model.first_frame_click(
            image=image,
            points=point,
            labels=labels,
            multimask="True"
        )
        masks[name] = mask
    return masks
