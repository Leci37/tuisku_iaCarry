import glob
import random

from PIL import Image
import numpy as np
import pandas as pd
import os
import os.path
from tensorflow import keras

from GT_Utils import bcolors, get_scalated_00_x_y_w_h, get_scalated_00_xmin_ymin_xmax_ymax

os.environ["VISION_TRAINING_ENDPOINT"] = "https://westeurope.api.cognitive.microsoft.com/"
os.environ["VISION_TRAINING_KEY"] = "0e7f39006dc24953988124b4d2c3a6bc"
# os.environ["VISION_PREDICTION_KEY"] =
# os.environ["VISION_PREDICTION_RESOURCE_ID"] =

# retrieve environment variables
ENDPOINT = os.environ["VISION_TRAINING_ENDPOINT"]
training_key = os.environ["VISION_TRAINING_KEY"]
# prediction_key = os.environ["VISION_PREDICTION_KEY"]
# prediction_resource_id = os.environ["VISION_PREDICTION_RESOURCE_ID"]

from azure.cognitiveservices.vision.customvision.training import CustomVisionTrainingClient
from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient
from azure.cognitiveservices.vision.customvision.training.models import ImageFileCreateBatch, ImageFileCreateEntry, Region
from msrest.authentication import ApiKeyCredentials
import os, time, uuid

def is_unique(s):
    a = s.to_numpy() # s.values (pandas<0.24)
    return (a[0] == a).all()

credentials = ApiKeyCredentials(in_headers={"Training-key": training_key})
trainer = CustomVisionTrainingClient(ENDPOINT, credentials)

PATH_BBOX = r"E:\iaCarry_img_eroski\Aug_A\df_size_Aug.csv"
df = pd.read_csv(PATH_BBOX, sep='\t', index_col=0)
print("Load data Shape: ", df.shape, " Path: ", PATH_BBOX)
df_gr = df.groupby(["type_prod"])['path'].count() #df.groupby(["type_prod"]).last().reset_index().sort_values(["type_prod"], ascending=True)
print(df_gr)

# Make two tags in the new project
PROJECT_ID = "bb92520b-5d96-436e-ae98-84af06f0dee0"
# for id_label, count in df_gr.items():
#     try:
#         _ = trainer.create_tag(PROJECT_ID, id_label)
#         print("\tCreated label: ", id_label )
#     except Exception:
#         print("\tYa created : ", id_label )


LIST_TAGS = trainer.get_tags(PROJECT_ID)
print("loaded Tags Num: ", len(LIST_TAGS), " Names: ", [x.name for x in LIST_TAGS])
list_images_with_regions = []
count_upload = 0
series_mix = df[:].groupby(["id_aug"])['path'].count()

df_hys_path = df[df['type_prod'] == "mahou5_33cl"]['path'].values
series_mix = df[df['path'].isin(df_hys_path)].groupby(["id_aug"])['path'].count().sort_values()[880:990:2]

for id, num_box in series_mix.items(): #[450::-1]
    df_box = df[df['id_aug'] == id]
    print(bcolors.HEADER + id + bcolors.ENDC , 'value: ', num_box , " shape: ", df_box.shape)
    print('index: ', id, 'value: ', num_box)

    path_img = df_box['path'].values[0]
    if not is_unique(df_box['path']) :
        raise ("Las images del mismo bbox apuntan a dos imagenes ", path_img)
    if not os.path.isfile(df_box['path'].values[0]):
        print("\t File does not exist" , df_box['path'].values[0])
        continue
    ram_opt = random.randint(0, 100)
    if ram_opt >=70:
        continue

    regions_id_xywh = []
    regions_label = []
    for index, row_box in df_box.iterrows():
        regions_label.append(row_box['type_prod'])
        name_and_id = [(x.name, x.id) for x in LIST_TAGS if x.name == row_box['type_prod']][0]
        x, y, w, h = get_scalated_00_xmin_ymin_xmax_ymax(row_box)
        regions_id_xywh.append(Region(tag_id=name_and_id[1], left=x, top=y, width=w, height=h))

    with open(path_img, mode="rb") as image_contents:
        img_azure_upload = ImageFileCreateEntry(name=path_img.split("\\")[-1], contents=image_contents.read(), regions=regions_id_xywh)
        list_images_with_regions.append(img_azure_upload)
        print("\t  Name:", path_img.split("\\")[-1], "\tRegions_num:", len(regions_id_xywh), "\tRegions_labels:", "', '".join(regions_label) )

    count_upload = count_upload + 1
    if count_upload % 2 == 0:
        # DOCU https://learn.microsoft.com/en-us/azure/ai-services/custom-vision-service/quickstarts/object-detection?tabs=windows%2Cvisual-studio&pivots=programming-language-python b
        try:
            print("Uploading imagenes ....    Num: ", len(list_images_with_regions) )# , bcolors.HEADER + name_tag + bcolors.ENDC)
            upload_result = trainer.create_images_from_files(PROJECT_ID, ImageFileCreateBatch(images=list_images_with_regions))
            if not upload_result.is_batch_successful:
                print("Image batch upload failed.")
                for image in upload_result.images:
                    print("Image status: ", image.status, "Path: ", image.source_url)
                print("FAIL upload ") #exit(-1)
            else:
                print("successful upload ")
            list_images_with_regions = []
        except Exception as e:
            print(bcolors.FAIL +   "EXCEPTION: ", str(e)+ bcolors.ENDC  )


print(df)