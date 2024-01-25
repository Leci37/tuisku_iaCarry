import glob
import cv2
import re
import os
import numpy as np
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder

from GT_Utils import bcolors
from Transfer_L_Mediun_utils import load_image_into_numpy_array, plot_detections, get_model_detection_function
from utils_transfer_learning import compute_resource_use_GPU

compute_resource_use_GPU("N")

def get_np_img_from_folder(path_glob_to_load_img  ):
    test_images_np = []
    TEST_PATHS = glob.glob(path_glob_to_load_img)
    for filename in TEST_PATHS:
        if filename.endswith(".png"):
            image = cv2.imread(filename)
            cv2.imwrite(filename.replace(".png", ".jpg"), image, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
            test_images_np.append(np.expand_dims(load_image_into_numpy_array(filename.replace(".png", ".jpg")), axis=0))
        else:
            test_images_np.append(np.expand_dims(load_image_into_numpy_array(filename), axis=0))
    print("Files to eval load from Path: ", path_glob_to_load_img, " Number :", len(test_images_np))
    return test_images_np

LABEL_MAP_PATH = r"E:/iaCarry_img_eroski/A_igor_img_etiqutadas/img_tfrecord/label_map.pbtxt"
def get_label_map_cat_index_13_DEMO():
    label_map = label_map_util.load_labelmap(LABEL_MAP_PATH)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=label_map_util.get_max_label_map_index(label_map), use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
    print("Read: label_map.pbtxt:", LABEL_MAP_PATH, "Categories: ", categories)
    return label_map, category_index

def restore_model_from_ckp(model_config, Path_model_ckp):
    # global Trained_model
    if  re.match(r"ckpt-\d{1,5}$", Path_model_ckp): #Path_model_ckp.endswith(".index") or Path_model_ckp.endswith(".meta"):
        raise ValueError("IMPORTANT: Please don't set the path to include the .index extension in the checkpoint file name.If you do set it to ckpt-0.index, there won't be any immediate error message, but later during training, youll notice that your models loss doesnt improve, which means that the pre-trained weights were not restored properly")
    Trained_model = model_builder.build(model_config=model_config, is_training=False)
    ckpt_trained = tf.compat.v2.train.Checkpoint(model=Trained_model)
    # Generate weights by running with dummy inputs.
    image, shapes = Trained_model.preprocess(tf.zeros([1, 640, 640, 3]))
    prediction_dict = Trained_model.predict(image, shapes)
    _ = Trained_model.postprocess(prediction_dict, shapes)
    print("Load model form ckp:  ", bcolors.OKBLUE + Path_model_ckp + bcolors.ENDC )
    ckpt_trained.restore(Path_model_ckp)
    print('Restored!')
    return Trained_model

# Again, uncomment this decorator if you want to run inference eagerly
@tf.function
def detect(input_tensor):
  preprocessed_image, shapes = Trained_model.preprocess(input_tensor)
  prediction_dict = Trained_model.predict(preprocessed_image, shapes)
  return Trained_model.postprocess(prediction_dict, shapes)


MODEL_TO_LOAD = "model_efi_d2_aug"
PIPELINE_CONFIG = MODEL_TO_LOAD + "/pipeline.config"
NUM_CHECK_POINT = 1203 #TODO cambiar esto para RUN
configs = config_util.get_configs_from_pipeline_file(PIPELINE_CONFIG)
model_config = configs['model']
print("Read PIPELINE_CONFIG: ", PIPELINE_CONFIG)
label_map, category_index = get_label_map_cat_index_13_DEMO()

Test_images_np = get_np_img_from_folder(r"C:\Users\Luis\Desktop\tuisku_ML\model_ssd_2Aug\test_iaCarry_outside*")
# CKP_PATHS = glob.glob(MODEL_TO_LOAD +'/ckpt-*.index')
# CKP_PATHS = [x.rstrip('.index') for x in CKP_PATHS ]

PATH_MODEL_TRANSFER_CHECK = MODEL_TO_LOAD +'/ckpt-'+str(NUM_CHECK_POINT)
#To Load to detect() function
Trained_model = restore_model_from_ckp(model_config=model_config, Path_model_ckp=PATH_MODEL_TRANSFER_CHECK)


MIN_SCORE = 0.4
LABEL_ID_OFFSET = 1
for i in range(len(Test_images_np)):
  input_tensor = tf.convert_to_tensor(Test_images_np[i], dtype=tf.float32)
  detections = detect(input_tensor)

  list_good_score = [x for x in detections['detection_scores'][0] if x > MIN_SCORE]
  if not os.path.exists(MODEL_TO_LOAD +"/test_imgA_"+str(NUM_CHECK_POINT)):
      os.makedirs(MODEL_TO_LOAD +"/test_imgA_"+str(NUM_CHECK_POINT))
  PATH_IMG_SAVE_TESTED = MODEL_TO_LOAD + "/test_imgA_" + str(NUM_CHECK_POINT) + "/img_" + ('%02d' % i) + ".jpg"
  print("\t",bcolors.OKCYAN + PATH_IMG_SAVE_TESTED + bcolors.ENDC)
  for j  in  range(0, len(list_good_score)):
      print("\tProbability: ",'{:.2f}'.format(detections['detection_scores'][0][j].numpy()),
            "\tClasses: ", detections['detection_classes'][0][j].numpy().astype(np.uint32) + LABEL_ID_OFFSET,
            "\tBoxes: ", detections['detection_boxes'][0][j].numpy())

  plot_detections(
      Test_images_np[i][0],
      detections['detection_boxes'][0].numpy(),
      detections['detection_classes'][0].numpy().astype(np.uint32)
      + LABEL_ID_OFFSET,
      detections['detection_scores'][0].numpy(),
      category_index, min_score_thresh = MIN_SCORE, figsize=(15, 20), image_name=PATH_IMG_SAVE_TESTED)

