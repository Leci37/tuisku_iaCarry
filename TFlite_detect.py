# https://github.com/tensorflow/models/blob/master/research/object_detection/colab_tutorials/centernet_on_device.ipynb

import tensorflow as tf
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageColor
import re, glob
import numpy as np


import Ultils_model_creation

#FROM https://github.com/TannerGilbert/Tensorflow-Lite-Object-Detection-with-the-Tensorflow-Object-Detection-API/blob/master/Convert_custom_object_detection_model_to_TFLITE.ipynb
from Utils_Detect_Signature import   img_proccess, draw_bounding_box_on_image
from Utils_detect_TFlite import detect_objects_TFlite

MODEL_TFLITE_PATH = 'model_mobil_v2_C/frozen/model_simple_sigNo_Mdata.tflite'
PATH_IMG_SOURCE = 'img_cat_dogs_test_valida/*.jpg'
interpreter = tf.lite.Interpreter(model_path=MODEL_TFLITE_PATH)
interpreter.allocate_tensors()
_, input_height, input_width, _ = interpreter.get_input_details()[0]['shape']

# "options": {"mean": [127.5],"std": [127.5]}
input_mean = 127.5
input_std = 127.5
HUMBRAL_PREDTIC = 0.5
CATEGORY_INDEX = Ultils_model_creation.get_category_index()
print("Type input data: ", interpreter.get_input_details()[0]['dtype'])

for i, image_path in enumerate(glob.glob(PATH_IMG_SOURCE)):
  print("load image to detect: ", image_path)
  image = Image.open(image_path)
  image_pred = image.resize((input_width ,input_height), Image.ANTIALIAS)

  path_test_img = img_proccess(image_path)

  if interpreter.get_input_details()[0]['dtype'] == np.float32:
    image_pred = (np.float32(image_pred) - input_mean) / input_std
  results = detect_objects_TFlite(interpreter, image_pred, HUMBRAL_PREDTIC)

  PATH_saved_img = "img_cat_TFlite/" + str(i) + ".jpg"

  image_np = path_test_img['origin_img_np']
  now_image_np = np.zeros((image_np.shape))
  np.copyto(now_image_np, image_np)
  for re  in results:
    if (float(re['score']) > HUMBRAL_PREDTIC):
      box, score, class_id_name = re['bounding_box'], re['score'], re['class_id']
      class_name = CATEGORY_INDEX[class_id_name + 1]['name']  # LUIS label_map[class_name]
      ymin, xmin, ymax, xmax = tuple(box)
      colors = list(ImageColor.colormap.values())

      font = ImageFont.load_default()
      display_str = (str(class_name) + ":" + str(int(score*100))+"%")
      color = colors[hash(str(class_name)) % len(colors)]

      image_pil = Image.fromarray(np.uint8(now_image_np)).convert("RGB")

      draw_bounding_box_on_image(
        image_pil,ymin,xmin,ymax,xmax,color,font,
        display_str_list=[display_str])

      np.copyto(now_image_np, np.array(image_pil))

  image_arry = Image.fromarray(np.uint8(np.array(now_image_np))).convert("RGB")
  print("draw_boxes_s saved with detection Path: ", PATH_saved_img)
  image_arry.save(PATH_saved_img)
