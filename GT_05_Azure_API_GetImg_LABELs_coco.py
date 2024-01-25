import os
import json
import requests
from PIL import Image
from io import BytesIO
import cv2
import numpy as np
import matplotlib.pyplot as plt

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
dict_category_index = label_map_util.create_category_index_from_labelmap(
    r"AZURE_models/TensorFlow._pb_eroski/label_map.pbtxt", use_display_name=True)
import utils_bbox
from GT_Utils import bcolors

print("Aqui salen los datos : ", "https://www.customvision.ai/projects/bb92520b-5d96-436e-ae98-84af06f0dee0#/settings")
import http.client, urllib.request, urllib.parse, urllib.error, base64

headers = {
    # Request headers
    'Training-Key': '0e7f39006dc24953988124b4d2c3a6bc',
    'Training-key': 'RREOc30QaN4N8xWYOGHB7WxcfrfqXZlWQfyQJF1UIUAzSeAF2I6r', #api key portal https://portal.azure.com/#@lllecinanagmail.onmicrosoft.com/resource/subscriptions/5f95e484-c770-45d5-9e08-defeacdcca21/resourceGroups/test_resurce_group/providers/Microsoft.Search/searchServices/luislecinanaserviceapi/Keys
}

list_data_raw = []
for num_skip in range(0,5300,255):
    params = urllib.parse.urlencode({
        'iterationId': '{string}',
        'tagIds': '{array}',
        'orderBy': '{string}',
        'take': '255',
        'skip': str(num_skip), #desde que imagen empieza
    })
    try:
        conn = http.client.HTTPSConnection('westeurope.api.cognitive.microsoft.com')
        conn.request("GET", "/customvision/v3.0/training/projects/bb92520b-5d96-436e-ae98-84af06f0dee0/images/tagged?%s" % params, "{body}", headers)
        response = conn.getresponse()
        data_raw = response.read()
        data_raw_my_json = data_raw.decode('utf8').replace("'", '"')
        _list_data_img = json.loads(data_raw_my_json)
        list_data_raw = [*list_data_raw, *_list_data_img]
        print("\t",num_skip , "   Download data len:\t", len( data_raw))
        conn.close()
    except Exception as e:
        print("[Errno {0}] {1}".format(e.errno, e.strerror))
# my_json = list_data_raw[0]
# # Load the JSON to a Python list & dump it back out as formatted JSON
# data_img = json.loads(my_json)
# str_json = json.dumps(data_img, indent=4, sort_keys=True)
# Create an empty list to store the annotations COCO https://medium.com/@manuktiwary/coco-format-what-and-how-5c7d22cf5301

annotations_coco = []
imgs_coco = []
count_id_anotation = 100
SAVED_IMG = r"E:\iaCarry_img_eroski\Aug_A_down"





print("\n", bcolors.HEADER +" Will download num images:  " +str(len(list_data_raw)) + bcolors.ENDC )
print("It will save in the folder: ",SAVED_IMG, "\n" )

def get_annotation_from_download_img_from_azure(img, dh, dw):
    global  count_id_anotation
    boxes_util = []
    detection_scores = []
    detection_scores_cat_index = []
    img['regions']#YOLO FORMAT  'left': , 'top': , 'width': , 'height':
    for ir in range(0, len(img['regions'])):
        count_id_anotation = count_id_anotation + 1
        reg = img['regions'][ir]
        y = reg['top'];
        x = reg['left'];
        w = reg['width'];
        h = reg['height']
        boxes_util.append(np.asarray((y, x, y + h, x + w)))
        tag_index = [(key, value) for key, value in dict_category_index.items() if value['name'] == reg['tagName']][0]
        detection_scores_cat_index.append(tag_index[0])
        detection_scores.append((1))  # ES el GT siempre es 100% , por poner algo
        # Annotate the image with a bounding box and label
        annotation = {
            "id": count_id_anotation,  # Use a unique identifier for the annotation
            "image_id": i,  # Use the same identifier for the image
            "category_id": tag_index[0],  # Assign a category ID to the object
            "bbox": [int(x * dw), int(y * dh), int(w * dw), int(h * dh)],
            # Specify the bounding box in the format [x, y, width, height]
            "area": int(w * dw) * int(h * dh),  # Calculate the area of the bounding box
            "iscrowd": 0,  # Set iscrowd to 0 to indicate that the object is not part of a crowd
        }
        annotations_coco.append(annotation)
    return boxes_util, detection_scores, detection_scores_cat_index


for i in range(0, len(list_data_raw )):
    img = list_data_raw[i]
    print("\nImage:", bcolors.OKCYAN + img['id']+ bcolors.ENDC ,"Shape: ", img['width'],"x",img['height'],"\tTags: ", ",".join( [x['tagName']  for x in img['tags'] ]),"\tUrl: ", img['originalImageUri']  )
    image = Image.open(BytesIO(requests.get(img['originalImageUri'] ).content))
    dh, dw =image.size

    boxes_util, detection_scores, detection_scores_cat_index = get_annotation_from_download_img_from_azure(img, dh, dw)

    name_to_coco_img_saved = SAVED_IMG + "/igor_"+str(len(detection_scores_cat_index))+"_{:05d}_".format(i)+".png"
    name_to_coco_img_saved = os.path.normpath(name_to_coco_img_saved)
    print("\t",i,".\t",name_to_coco_img_saved)
    image.save(name_to_coco_img_saved)
    image_np = cv2.imread(name_to_coco_img_saved)
    # dh, dw, _ = image_np.shape

    SAVED_IMG_label = SAVED_IMG +  "/check_label/igor_"+str(len(detection_scores_cat_index))+"_{:05d}_".format(i)+ "_"+  '.'.join(str(v) for v in detection_scores_cat_index) +".png"
    SAVED_IMG_label = os.path.normpath(SAVED_IMG_label)
    np_img_labeled = vis_util.visualize_boxes_and_labels_on_image_array(
            image_np,
            np.asarray(boxes_util),
            detection_scores_cat_index,  # output_dict['detection_classes'],
            detection_scores,
            dict_category_index,
            min_score_thresh=0.66,
            # instance_masks=output_dict.get('detection_masks'),
            use_normalized_coordinates=True,
            line_thickness=2)
    cv2.imwrite(SAVED_IMG_label, np_img_labeled)
    print("\t",i,".\t  ",SAVED_IMG_label)
    # cv2.imshow('Object detector', image_np)
    # cv2.waitKey(0)
    img_coco ={
                "id": i,  # Use the same identifier as the annotation
                "width":  dw,  # Set the width of the image
                "height": dh,  # Set the height of the image
                "file_name": name_to_coco_img_saved,  # Set the file name of the image
                # "license": 1,  # Set the license for the image (optional)
            }
    imgs_coco.append(img_coco)

# Create the COCO JSON object
coco_data = {
    "info": {
        "description": "My COCO dataset",  # Add a description for the dataset
        "url": "https://iacarry.tuisku.eu/",  # Add a URL for the dataset (optional)
        "version": "1.0",  # Set the version of the dataset
        "year": 2024,  # Set the year the dataset was created
        "contributor": "@Leci37",  # Add the name of the contributor (optional)
        "date_created": "2024-01-01T00:00:00",  # Set the date the dataset was created
    },
    "licenses": [
    {
        "id": 1,
        "url": "https://iacarry.tuisku.eu/",
        "name": "tuisku.eu Domain"
    }
    ],  # Add a list of licenses for the images in the dataset (optional)
    "images": imgs_coco,
    "annotations": annotations_coco,  # Add the list of annotations to the JSON object
    "categories" : list(dict_category_index.values())
    # "categories": [{"id": 1, "name": "object"}],  # Add a list of categories for the objects in the dataset
}

# Save the COCO JSON object to a file
PATH_COCO_DATA = SAVED_IMG + "/cocod.json"
with open(PATH_COCO_DATA, "w") as f:
    json.dump(coco_data, f , indent=4)
print(PATH_COCO_DATA)
from Utils import COCO_json_format_validator
COCO_json_format_validator.coco_json_validator(PATH_COCO_DATA)
print("COCO json passed the test : ", PATH_COCO_DATA)


import os

#from https://github.com/Gradiant/pyodi/blob/master/pyodi/apps/coco/coco_split.py
output_files = utils_bbox.random_split_coco_json(annotations_file= PATH_COCO_DATA, output_filename=SAVED_IMG + "/cocod", val_percentage = 0.33, seed = 123)
print(output_files)
# random_split_coco_json solo admite partir si el nombre acaba en _val
try: # clean in case
    os.remove(output_files[1].replace("_val", "_test"))
except OSError:
    pass
os.rename(output_files[1], output_files[1].replace("_val", "_test"))

output_files = utils_bbox.random_split_coco_json(annotations_file= output_files[0], output_filename=SAVED_IMG + "/cocod", val_percentage = 0.08, seed = 123)
print(bcolors.HEADER +  "Files generated:"+ bcolors.ENDC,   ", ".join(output_files), ", ", output_files[1].replace("_val", "_test") )

