import glob
from PIL import Image
import numpy as np
import pandas as pd
import os
from tensorflow import keras

from GT_Utils import bcolors

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



credentials = ApiKeyCredentials(in_headers={"Training-key": training_key})
trainer = CustomVisionTrainingClient(ENDPOINT, credentials)

PATH_BBOX = r"E:\iaCarry_img_eroski\df_size_Rota_img.csv"
df = pd.read_csv(PATH_BBOX, sep='\t', index_col=0)
print("Load data Shape: ", df.shape, " Path: ", PATH_BBOX)
df_gr = df.groupby(["type_prod"])['path'].count() #df.groupby(["type_prod"]).last().reset_index().sort_values(["type_prod"], ascending=True)

# Make two tags in the new project
PROJECT_ID = "bb92520b-5d96-436e-ae98-84af06f0dee0"
# for id_label, count in df_gr.items():
#     try:
#         _ = trainer.create_tag(PROJECT_ID, id_label)
#         print("\tCreated label: ", id_label )
#     except Exception:
#         print("\tYa created : ", id_label )


list_tags = trainer.get_tags(PROJECT_ID)
print("loaded Tags Num: ", len(list_tags), " Names: ", [x.name for x in list_tags ])
# base_image_location = os.path.join (os.path.dirname(__file__), r"E:\iaCarry_img_eroski\frames_rota_fon")

count_upload = 0

for tag in  list_tags:
    name_tag = tag.name
    print("\n" + bcolors.HEADER + name_tag + bcolors.ENDC +  "\ttag_id: "+tag.id )
    list_images_with_regions = []

    df_n = df[df["type_prod"] == name_tag ]
    print("Label loaded Name: ", name_tag, " Shape: ", df_n.shape)

    for index, row in df_n[-50:-30].iterrows():
        count_upload = count_upload +1
        x, y, w, h = get_scalated_00_x_y_w_h(row)
        regions = [Region(tag_id=tag.id, left=x, top=y, width=w, height=h)]

        with open(row['path'], mode="rb") as image_contents:
            list_images_with_regions.append(ImageFileCreateEntry(name=row['path'].split("\\")[-1], contents=image_contents.read(), regions=regions))
        print("\t  Name:",row['path'].split("\\")[-1] ,"\tTag:" ,name_tag ,  "\tregion(x,y):", str(x)+"x"+str(y),  "\tregion(w,h):" ,str(w)+" "+str(h))

        if count_upload % 5 == 0:
            # DOCU https://learn.microsoft.com/en-us/azure/ai-services/custom-vision-service/quickstarts/object-detection?tabs=windows%2Cvisual-studio&pivots=programming-language-python b
            print("Uploading imagenes ....    Num: ", len(list_images_with_regions) , bcolors.HEADER + name_tag + bcolors.ENDC)
            upload_result = trainer.create_images_from_files(PROJECT_ID, ImageFileCreateBatch(images=list_images_with_regions))
            if not upload_result.is_batch_successful:
                print("Image batch upload failed.")
                for image in upload_result.images:
                    print("Image status: ", image.status, "Path: ", image.source_url)
                print("FAIL upload ") #exit(-1)
            else:
                print("successful upload ")
            list_images_with_regions = []


# for file_name in fork_image_regions.keys():
#     x,y,w,h = fork_image_regions[file_name]
#     regions = [ Region(tag_id=fork_tag.id, left=x,top=y,width=w,height=h) ]
#
#     with open(os.path.join (base_image_location, "fork", file_name + ".jpg"), mode="rb") as image_contents:
#         tagged_images_with_regions.append(ImageFileCreateEntry(name=file_name, contents=image_contents.read(), regions=regions))
#
#
# upload_result = trainer.create_images_from_files(PROJECT_ID, ImageFileCreateBatch(images=image_list))
# if not upload_result.is_batch_successful:
#     print("Image batch upload failed.")
#     for image in upload_result.images:
#         print("Image status: ", image.status)
#     exit(-1)

print(df)