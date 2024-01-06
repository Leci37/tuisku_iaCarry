import glob
import matplotlib
import matplotlib.pyplot as plt
import os
import numpy as np
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder

from Transfer_L_Mediun_utils import load_image_into_numpy_array, plot_detections, get_model_detection_function


MODEL_TO_LOAD = "ssd_cat_dog_zombie_A1"
pipeline_config = MODEL_TO_LOAD + "/pipeline.config"
PATH_MODEL_TRANSFER_CHECK = MODEL_TO_LOAD +'/checkpoint/ckpt-1'

configs = config_util.get_configs_from_pipeline_file(pipeline_config)
model_config = configs['model']
print("Read: ", pipeline_config)

LABEL_MAP_PATH = './ssd_mobilenet_v2_fpnlite_640x640/mscoco_label_map_cat_dog.pbtxt'
label_map = label_map_util.load_labelmap(LABEL_MAP_PATH)
categories = label_map_util.convert_label_map_to_categories(
     label_map, max_num_classes=label_map_util.get_max_label_map_index(label_map),use_display_name=True)
category_index = label_map_util.create_category_index(categories)
print("Read: ", LABEL_MAP_PATH , "Categories: ", categories)

trained_model = model_builder.build(model_config=model_config, is_training=False)
ckpt_trained = tf.compat.v2.train.Checkpoint(model=trained_model)
# Generate weights by running with dummy inputs.
image, shapes = trained_model.preprocess(tf.zeros([1, 640, 640, 3]))
prediction_dict = trained_model.predict(image, shapes)
_ = trained_model.postprocess(prediction_dict, shapes)
print("Load: ",PATH_MODEL_TRANSFER_CHECK)
ckpt_trained.restore(PATH_MODEL_TRANSFER_CHECK)
print('Restored!')


test_images_np = []
TEST_PATHS = glob.glob(r"C:\Users\Luis\Desktop\Object_detec\cat_dogs_test_valida" + '/*.jpg')
for test_path in TEST_PATHS:
   test_images_np.append(np.expand_dims(load_image_into_numpy_array(test_path), axis=0))

# Again, uncomment this decorator if you want to run inference eagerly
@tf.function
def detect(input_tensor):
  """
  Returns:
    A dict containing 3 Tensors (`detection_boxes`, `detection_classes`,
      and `detection_scores`).
  """
  preprocessed_image, shapes = trained_model.preprocess(input_tensor)
  prediction_dict = trained_model.predict(preprocessed_image, shapes)
  return trained_model.postprocess(prediction_dict, shapes)

# Note that the first frame will trigger tracing of the tf.function, which will
# take some time, after which inference should be fast.
MIN_SCORE = 0.5
label_id_offset = 1
for i in range(len(test_images_np)):
  input_tensor = tf.convert_to_tensor(test_images_np[i], dtype=tf.float32)
  detections = detect(input_tensor)

  list_good_score = [x for x in detections['detection_scores'][0] if x > MIN_SCORE]
  print("ssd_cat_dog_zombie_A1/test_img/gif_frame_" + ('%02d' % i) + ".jpg")
  for j  in  range(0, len(list_good_score)):
      print("\tProbability: ",'{:.2f}'.format(detections['detection_scores'][0][j].numpy()),
            "\tClasses: ", detections['detection_classes'][0][j].numpy().astype(np.uint32) + label_id_offset,
            "\tBoxes: ", detections['detection_boxes'][0][j].numpy())

  plot_detections(
      test_images_np[i][0],
      detections['detection_boxes'][0].numpy(),
      detections['detection_classes'][0].numpy().astype(np.uint32)
      + label_id_offset,
      detections['detection_scores'][0].numpy(),
      category_index, min_score_thresh = MIN_SCORE, figsize=(15, 20), image_name="ssd_cat_dog_zombie_A1/test_img/gif_frame_" + ('%02d' % i) + ".jpg")
# print(detections)
# The output result below. Although omitted, there are 100 each
#'detection_boxes''detection_classes''detection_scores' is the final result
#'detection_anchor_indices'' raw_detection_boxes'' raw_detection_scores' is the data in the middle used to calculate the final result (I think, maybe)
