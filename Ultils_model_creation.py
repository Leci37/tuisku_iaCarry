import os

import matplotlib.pyplot as plt
import numpy as np
from six import BytesIO
from PIL import Image, ImageDraw, ImageFont


import tensorflow as tf

from object_detection.utils import config_util
from object_detection.utils import visualization_utils as viz_utils




def load_image_into_numpy_array(path):
  """Load an image from file into a numpy array.

  Puts image into numpy array to feed into tensorflow graph.
  Note that by convention we put it into a numpy array with shape
  (height, width, channels), where channels=3 for RGB.

  Args:
    path: a file path.

  Returns:
    uint8 numpy array with shape (img_height, img_width, 3)
  """
  img_data = tf.io.gfile.GFile(path, 'rb').read()
  image = Image.open(BytesIO(img_data))
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

def plot_detections(image_np,
                    boxes,
                    classes,
                    scores,
                    category_index,
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
      min_score_thresh=0.5)
  if image_name:
    plt.imsave(image_name, image_np_with_annotations)
  else:
    plt.imshow(image_np_with_annotations)


def get_category_index():
    zombie_CLASS_ID = 1 #Since you are just predicting one class (zombie), please assign 1 to the zombie class ID.
    cat_CLASS_ID = 2
    dog_CLASS_ID = 3
    category_index = {zombie_CLASS_ID :
                         {'id'  : zombie_CLASS_ID,'name': 'zombie'},
                    cat_CLASS_ID :
                         {'id'  : cat_CLASS_ID,'name': 'cat'},
                    dog_CLASS_ID :
                         {'id'  : dog_CLASS_ID,'name': 'dog'}
                      }
    return category_index

# Again, uncomment this decorator if you want to run inference eagerly
@tf.function
def detect(detection_model, input_tensor):
  """Run detection on an input image.

  Args:
    input_tensor: A [1, height, width, 3] Tensor of type tf.float32.
      Note that height and width can be anything since the image will be
      immediately resized according to the needs of the model within this
      function.

  Returns:
    A dict containing 3 Tensors (`detection_boxes`, `detection_classes`,
      and `detection_scores`).
  """
  preprocessed_image, shapes = detection_model.preprocess(input_tensor)
  prediction_dict = detection_model.predict(preprocessed_image, shapes)
  return detection_model.postprocess(prediction_dict, shapes)

def predict_and_save_imagen(detection_model, images_np, category_index, path):
    label_id_offset = 1
    input_tensor = tf.convert_to_tensor(images_np, dtype=tf.float32)
    # input_tensor = tf.expand_dims(tf.convert_to_tensor(test_images_np[i], dtype=tf.float32), axis=0)
    print(path + " shape: ", input_tensor.shape , " img.shape: ",images_np.shape)

    # preprocessed_image, shapes = detection_model.preprocess(input_tensor)
    # prediction_dict = detection_model.predict(preprocessed_image, shapes)
    # , detection_model.postprocess(prediction_dict, shapes)
    detections = detect(detection_model, input_tensor)

    print(path  , "Probability: ",
          detections['detection_scores'][0][0].numpy(),
          "Classes: ", detections['detection_classes'][0][0].numpy().astype(np.uint32) + label_id_offset,
          "Boxes: ", detections['detection_boxes'][0][0].numpy())
    plot_detections(
        images_np[0],
        detections['detection_boxes'][0].numpy(),
        detections['detection_classes'][0].numpy().astype(np.uint32)
        + label_id_offset,
        detections['detection_scores'][0].numpy(),
        category_index, figsize=(15, 20), image_name=path )

# 1.0 LOAD the imagenes with bbox or segmentation
# assign the name (string) of the directory containing the training images
def get_valitation_images():
    global train_image_dir, list_test_images_np, cat_dog, i, path_img_to_train
    train_image_dir = 'img_cat_dogs_test_valida'
    # declare an empty list
    list_test_images_np = []
    # DOG vs CAT
    # for cat_dog in ['cat', 'dog']:
    #     for i in range(0, 10):  # Luis Code extra
    #         path_img_to_train = os.path.join(train_image_dir, cat_dog + '.200' + str(i) + '.jpg')
    #         print(path_img_to_train)
    #         # test_images_np.append( load_image_into_numpy_array(path_img_to_train))
    #         list_test_images_np.append(
    #             np.expand_dims( load_image_into_numpy_array(path_img_to_train), axis=0))
    #     for i in range(10, 30):  # Luis Code extra
    #         path_img_to_train = os.path.join(train_image_dir, cat_dog + '.20' + str(i) + '.jpg')
    #         list_test_images_np.append(
    #             np.expand_dims( load_image_into_numpy_array(path_img_to_train), axis=0))

    for i in range(1,3): # 11):  # Luis Code extracat_dogs_test_valida/zombie (32).jpg
        path_img_to_train = os.path.join(train_image_dir,  'zombie (' + str(i) + ').jpg')
        print(path_img_to_train)
        # test_images_np.append( load_image_into_numpy_array(path_img_to_train))
        list_test_images_np.append(
            np.expand_dims(load_image_into_numpy_array(path_img_to_train), axis=0))
    print("Images are loaded for train len(list) : ", len(list_test_images_np))
    return list_test_images_np


def save_detecion_pd_checkpoint(detection_model, PATH_MODELS_CHECK, configs):
    # Save new pipeline config
    new_pipeline_proto = config_util.create_pipeline_proto_from_configs(configs)
    config_util.save_pipeline_config(new_pipeline_proto, PATH_MODELS_CHECK)
    exported_ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
    ckpt_manager = tf.train.CheckpointManager(
        exported_ckpt, directory=PATH_MODELS_CHECK + "/checkpoint/", max_to_keep=1)
    print('Done fine-tuning!')
    ckpt_manager.save()
    print('Checkpoint saved! path: ',PATH_MODELS_CHECK + "/checkpoint/")
    tf.saved_model.save(detection_model, PATH_MODELS_CHECK + "/saved_model/", signatures=None, options=None)
    print('Save .pb ', PATH_MODELS_CHECK + "/saved_model/")

    # new_model = tf.saved_model.load(PATH_MODELS_CHECK + "/saved_model_sig/")
    # detections = new_model.signatures['detect']("your_detection_img_tensor")

    # try:
    #     tf.saved_model.save(detection_model, PATH_MODELS_CHECK + "/saved_model/", signatures=None, options=None)
    # except:
    #     print("Fallo tf.saved_model.save")
    #
    # #NEW
    # try:
    #     detection_model.save(PATH_MODELS_CHECK +'/modelH5.h5')
    # except:
    #     print("Fallo detection_model.save(PATH_MODELS_CHECK +'/modelH5.h5')")
    #
    # try:
    #     detection_model.save(PATH_MODELS_CHECK +'/model')
    # except:
    #     print("Fallo NO H5 detection_model.save(PATH_MODELS_CHECK +'/model')")
    #
    # #https://www.tensorflow.org/guide/keras/save_and_serialize?hl=es-419#exportar_a_savedmodel
    # # Exportar el modelo a 'SavedModel'
    # try:
    #     tf.keras.experimental.export_saved_model(detection_model, PATH_MODELS_CHECK +'/expri')
    # except:
    #     print("tf.keras.experimental.export_saved_model(detection_model, PATH_MODELS_CHECK +'/expri')")


def plot_parameter_setup_config(plt):
    plt.rcParams['axes.grid'] = False
    plt.rcParams['xtick.labelsize'] = False
    plt.rcParams['ytick.labelsize'] = False
    plt.rcParams['xtick.top'] = False
    plt.rcParams['xtick.bottom'] = False
    plt.rcParams['ytick.left'] = False
    plt.rcParams['ytick.right'] = False
    plt.rcParams['figure.figsize'] = [14, 7]
    return plt