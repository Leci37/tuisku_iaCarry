import tensorflow as tf
import numpy as np

from Utils_Detect_Signature import img_proccess, draw_boxes_s

"""# Load test images and run inference with new model!"""
import Ultils_model_creation

category_index = Ultils_model_creation.get_category_index()

PATH_MODELS_CHECK = 'model_mobil_v2_C'
PATH_FOLDER_IMG_TEST = "img_cat_dogs_test_valida" # NO TOCAR
PATH_FOLDER_IMG_TEST_RESULT = "img_cat_mobiv2_C"


detection_model =  tf.saved_model.load( PATH_MODELS_CHECK + "/save_model_detect")
signatures = detection_model.signatures
print('Signature:', signatures)
detector = detection_model.signatures['detect']

list_paths = ["cat.2000.jpg", "cat.2001.jpg", "cat.2002.jpg", "cat.2003.jpg", "cat.2004.jpg", "cat.2005.jpg", "cat.2006.jpg", "cat.2007.jpg", "cat.2008.jpg", "cat.2009.jpg", "cat.2010.jpg", "cat.2011.jpg", "cat.2012.jpg", "cat.2013.jpg", "cat.2014.jpg", "cat.2015.jpg", "cat.2016.jpg", "cat.2017.jpg", "cat.2018.jpg", "cat.2019.jpg", "cat.2020.jpg", "cat.2021.jpg", "cat.2022.jpg", "cat.2023.jpg", "cat.2024.jpg", "cat.2025.jpg", "cat.2026.jpg", "cat.2027.jpg", "cat.2028.jpg", "cat.2029.jpg", "cat.2030.jpg", "cat.2031.jpg", "dog.2000.jpg", "dog.2001.jpg", "dog.2002.jpg", "dog.2003.jpg", "dog.2004.jpg", "dog.2005.jpg", "dog.2006.jpg", "dog.2007.jpg", "dog.2008.jpg", "dog.2009.jpg", "dog.2010.jpg", "dog.2011.jpg", "dog.2012.jpg", "dog.2013.jpg", "dog.2014.jpg", "dog.2015.jpg", "dog.2016.jpg", "dog.2017.jpg", "dog.2018.jpg", "dog.2019.jpg", "dog.2020.jpg", "dog.2021.jpg", "dog.2022.jpg", "dog.2023.jpg", "dog.2024.jpg", "dog.2025.jpg", "dog.2026.jpg", "dog.2027.jpg", "dog.2028.jpg", "dog.2029.jpg", "dog.2030.jpg", "dog.2031.jpg", "dog.2032.jpg", "dog.2033.jpg", "dog.2034.jpg", "dog.2035.jpg", "zombie (1).jpg", "zombie (10).jpg", "zombie (11).jpg", "zombie (12).jpg", "zombie (13).jpg", "zombie (14).jpg", "zombie (15).jpg", "zombie (16).jpg", "zombie (17).jpg", "zombie (18).jpg", "zombie (19).jpg", "zombie (2).jpg", "zombie (20).jpg", "zombie (21).jpg", "zombie (22).jpg", "zombie (23).jpg", "zombie (24).jpg", "zombie (25).jpg", "zombie (26).jpg", "zombie (27).jpg", "zombie (28).jpg", "zombie (29).jpg", "zombie (3).jpg", "zombie (30).jpg", "zombie (31).jpg", "zombie (32).jpg", "zombie (33).jpg", "zombie (34).jpg", "zombie (4).jpg", "zombie (5).jpg", "zombie (6).jpg", "zombie (7).jpg", "zombie (8).jpg", "zombie (9).jpg"]


category_index = Ultils_model_creation.get_category_index()
label_id_offset = 1


for path_test_img_path in list_paths:
    print("TEST image: ", path_test_img_path)
    path_test_img = img_proccess(PATH_FOLDER_IMG_TEST + "/"+path_test_img_path)

    detections= detector( input_tensor = path_test_img["detection_img_tensor"] )
    path_saved = PATH_FOLDER_IMG_TEST_RESULT + "/"+ path_test_img_path
    print("SAVED image: ",path_saved)
    draw_boxes_s(
        path_test_img['origin_img_np'],
        detections['detection_classes'][0],
        detections['detection_boxes'][0],
        detections["detection_scores"][0],
        0.4, path_saved,label_map = category_index )

