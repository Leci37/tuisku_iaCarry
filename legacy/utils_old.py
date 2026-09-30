import os
import cv2
import numpy as np

def parse_label_map(pbtxt_path):
    label_map = {}
    with open(pbtxt_path, 'r', encoding='utf-8') as f:  # ✅ force UTF-8
        content = f.read()
    items = content.split('item {')
    for item in items[1:]:
        id_line = [line for line in item.split('\n') if 'id:' in line]
        name_line = [line for line in item.split('\n') if 'name:' in line]
        if id_line and name_line:
            idx = int(id_line[0].split(':')[-1].strip())
            name = name_line[0].split(':')[-1].strip().replace('"', '')
            label_map[idx] = name
    return label_map



def segment_objects(model, image, object_names):
    masks = {}
    for name in object_names:
        # TODO: auto-generate a point/bbox prompt based on name (currently non-trivial)
        # For now, mock or use placeholder point
        prompt_point = np.array([[image.shape[1]//2, image.shape[0]//2]])
        labels = np.array([1])
        mask, _, _ = model.first_frame_click(image=image, points=prompt_point, labels=labels, multimask="True")
        masks[name] = mask
    return masks
