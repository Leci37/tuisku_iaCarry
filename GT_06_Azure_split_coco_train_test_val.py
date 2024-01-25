import json
import utils_bbox
import os

from GT_Utils import bcolors
from Utils import COCO_json_format_validator

SAVED_IMG = r"E:\iaCarry_img_eroski\A_igor_img_etiqutadas"
SAVED_IMG = r"E:\iaCarry_img_eroski\Aug_A_down"

# Save the COCO JSON object to a file
PATH_COCO_DATA = SAVED_IMG + "/cocod.json"
f = open(PATH_COCO_DATA, )
coco_data = json.load(f)

#TODO to remove in case file dont exists
# list_img_coco_exists = [i for i in coco_data['images'] if os.path.isfile(  i['file_name'] ) ]
# print(bcolors.OKCYAN +  "Load paths from coco "+ bcolors.ENDC, "num Files " , len(coco_data['images']), "\tfiles remove (dont exists) num: ",  len(coco_data['images']) - len(list_img_coco_exists)  )
# coco_data['images'] = list_img_coco_exists
# PATH_COCO_DATA = SAVED_IMG + "/cocod.json"
# with open(PATH_COCO_DATA, "w") as f:
#     json.dump(coco_data, f , indent=4)


# TODO CUIDADO  cocosplit.py" --multi-class not splitting images with multiple boxes correctly https://github.com/akarazniewicz/cocosplit/issues/12
# $ python cocosplit.py --having-annotations --multi-class -s 0.8 /path/to/your/coco_annotations.json train.json test.json

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

for jpath in ["_train", "_test","_val" ]:
    path_coco = PATH_COCO_DATA.replace(".json", jpath +".json" )
    COCO_json_format_validator.coco_json_validator(path_coco) #TODO no removed importan
    print("COCO json passed the test : ", path_coco)
