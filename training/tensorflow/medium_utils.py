# # https://rockyshikoku.medium.com/how-to-use-tensorflow-object-detection-api-with-the-colab-sample-notebooks-477707fadf1b

import matplotlib
import matplotlib.pyplot as plt
import os
import scipy.misc
import numpy as np
from six import BytesIO
from PIL import Image, ImageDraw, ImageFont
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder


""" 4. The functions for reading the images """
def load_image_into_numpy_array(path):
   """ Read the images and put them to the numpy array
    Puts image into numpy array to feed into tensorflow graph.
    Note that by convention we put it into a numpy array with shape
   (height, width, channels), where channels=3 for RGB.
    Args:
     path: the file path to the image
    Returns:
     uint8 numpy array with shape (img_height, img_width, 3)
   """
   img_data = tf.io.gfile.GFile(path, 'rb').read()
   image = Image.open(BytesIO(img_data))
   (im_width, im_height) = image.size
   return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)

def get_keypoint_tuples(eval_config):
   """Return a tuple list of keypoint edges from the eval config.
    Args:
     eval_config: an eval config containing the keypoint edges
    Returns:
     a list of edge tuples, each in the format (start, end)
   """
   tuple_list = []
   kp_list = eval_config.keypoint_edge
   for edge in kp_list:
     tuple_list.append((edge.start, edge.end))
   return tuple_list

"""5. The functions visualize the results."""
def plot_detections(image_np,
                    boxes,
                    classes,
                    scores,
                    category_index,min_score_thresh,
                    figsize=(12, 16),
                    image_name=None):
   """Wrapper function to visualize detections.

     Args:
       image_np: uint8 numpy array with shape (img_height, img_width, 3)
       boxes: a numpy array of shape [N, 4]
       classes: a numpy array of shape [N]. Note that class indices are 1-based,
         and match the keys in the label map.
       scores: a numpy array of shape [N] or None.  If scores=None, then
         this function assumes that the boxes to be plotted are groundtruth
         boxes and plot all boxes as black with no classes or scores.
       category_index: a dict containing category dictionaries (each holding
         category index `id` and category name `name`) keyed by category indices.
       figsize: size for the figure.
       image_name: a name for the image file.
     """
   image_np_with_annotations = image_np.copy()
   viz_utils.visualize_boxes_and_labels_on_image_array(
      image_np_with_annotations,
      boxes,
      classes,
      scores,
      category_index,
      use_normalized_coordinates=True,
      min_score_thresh=min_score_thresh)
   if image_name:
      plt.imsave(image_name, image_np_with_annotations)
   else:
      plt.imshow(image_np_with_annotations)

"""7. prepare the inferencing function"""
def get_model_detection_function(model):
   """Get a tf.function for detection."""
   @tf.function
   def detect_fn(image):
      """Detect objects in image."""
      image, shapes = model.preprocess(image)
      prediction_dict = model.predict(image, shapes)
      detections = model.postprocess(prediction_dict, shapes)
      return detections, prediction_dict, tf.reshape(shapes, [-1])
   return detect_fn
