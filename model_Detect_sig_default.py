import tensorflow as tf
import numpy as np

from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

# from Utils_Detect_Signature import img_proccess, draw_boxes_s

"""# Load test images and run inference with new model!"""
import Ultils_model_creation

#https://techzizou.com/build-android-app-for-custom-object-detection-using-tf2/
# 16) Test your trained model
# Export inference graph
# Current working directory is /content/models/research/object_detection
print("REQUERIERE ")
print("REQUERIERE !python exporter_main_v2.py --trained_checkpoint_dir=/mydrive/customTF2/training --pipeline_config_path=/content/gdrive/MyDrive/customTF2/data/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.config --output_directory /mydrive/customTF2/data/inference_graph")
# category_index = rubber_duck_ultils.get_category_index()
PATH_TO_LABELS = 'model_101_C/mscoco_label_map.pbtxt'
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

PATH_MODELS_CHECK = 'model_101_C/frozen_graph'
detection_model =  tf.saved_model.load(PATH_MODELS_CHECK +"/saved_model") # PATH_MODELS_CHECK + "/frozen/saved_model")

signatures = detection_model.signatures
print('Signature:', signatures)
print("Para que el .pb cargado se genere las signatures['serving_default'] hay que Congelarle Frozen  con export_tflite_graph_tf2.py ")
print("Structured_outputs: ", detection_model.signatures["serving_default"].structured_outputs)
print("Structured_ Types: ", detection_model.signatures["serving_default"].output_dtypes )
print("Structured_ Shapes: ", detection_model.signatures["serving_default"].output_shapes )

#TUTORIAL FOR serving_default https://www.mygreatlearning.com/blog/object-detection-using-tensorflow/
detector = detection_model.signatures["serving_default"]
print("FUNCION_detector: ", str(detector) )

list_paths = ["cat.2000.jpg", "cat.2001.jpg", "cat.2002.jpg", "cat.2003.jpg", "cat.2004.jpg", "cat.2005.jpg", "cat.2006.jpg", "cat.2007.jpg", "cat.2008.jpg", "cat.2009.jpg", "cat.2010.jpg", "cat.2011.jpg", "cat.2012.jpg", "cat.2013.jpg", "cat.2014.jpg", "cat.2015.jpg", "cat.2016.jpg", "cat.2017.jpg", "cat.2018.jpg", "cat.2019.jpg", "cat.2020.jpg", "cat.2021.jpg", "cat.2022.jpg", "cat.2023.jpg", "cat.2024.jpg", "cat.2025.jpg", "cat.2026.jpg", "cat.2027.jpg", "cat.2028.jpg", "cat.2029.jpg", "cat.2030.jpg", "cat.2031.jpg", "dog.2000.jpg", "dog.2001.jpg", "dog.2002.jpg", "dog.2003.jpg", "dog.2004.jpg", "dog.2005.jpg", "dog.2006.jpg", "dog.2007.jpg", "dog.2008.jpg", "dog.2009.jpg", "dog.2010.jpg", "dog.2011.jpg", "dog.2012.jpg", "dog.2013.jpg", "dog.2014.jpg", "dog.2015.jpg", "dog.2016.jpg", "dog.2017.jpg", "dog.2018.jpg", "dog.2019.jpg", "dog.2020.jpg", "dog.2021.jpg", "dog.2022.jpg", "dog.2023.jpg", "dog.2024.jpg", "dog.2025.jpg", "dog.2026.jpg", "dog.2027.jpg", "dog.2028.jpg", "dog.2029.jpg", "dog.2030.jpg", "dog.2031.jpg", "dog.2032.jpg", "dog.2033.jpg", "dog.2034.jpg", "dog.2035.jpg", "zombie (1).jpg", "zombie (10).jpg", "zombie (11).jpg", "zombie (12).jpg", "zombie (13).jpg", "zombie (14).jpg", "zombie (15).jpg", "zombie (16).jpg", "zombie (17).jpg", "zombie (18).jpg", "zombie (19).jpg", "zombie (2).jpg", "zombie (20).jpg", "zombie (21).jpg", "zombie (22).jpg", "zombie (23).jpg", "zombie (24).jpg", "zombie (25).jpg", "zombie (26).jpg", "zombie (27).jpg", "zombie (28).jpg", "zombie (29).jpg", "zombie (3).jpg", "zombie (30).jpg", "zombie (31).jpg", "zombie (32).jpg", "zombie (33).jpg", "zombie (34).jpg", "zombie (4).jpg", "zombie (5).jpg", "zombie (6).jpg", "zombie (7).jpg", "zombie (8).jpg", "zombie (9).jpg"]
PATH_FOLDER_IMG_TEST_LOAD = "img_cat_dogs_test_valida"
PATH_FOLDER_IMG_TEST_EVALED = "img_cat_101_C_frozen"

# category_index = rubber_duck_ultils.get_category_index()
label_id_offset = 1

#https://www.mygreatlearning.com/blog/object-detection-using-tensorflow/
def run_inference_for_single_image(model, image):
    image = np.asarray(image)
    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(image ) #, tf.float32 )
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis, ...]

    # Run inference
    model_fn = model.signatures['serving_default']
    output_dict = model_fn(input_tensor)

    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.
    num_detections = int(output_dict.pop('num_detections'))
    output_dict = {key: value[0, :num_detections].numpy()
                   for key, value in output_dict.items()}
    output_dict['num_detections'] = num_detections

    # detection_classes should be ints.
    output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)

    # Handle models with masks:
    if 'detection_masks' in output_dict:
        # Reframe the the bbox mask to the image size.
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            output_dict['detection_masks'], output_dict['detection_boxes'],
            image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5,
                                           tf.uint8)
        output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()

    return output_dict

for path_test_img_path in list_paths:
    print("TEST image: ", path_test_img_path)
    image_np = np.array(Image.open(PATH_FOLDER_IMG_TEST_LOAD + "/" + path_test_img_path))

    output_dict = run_inference_for_single_image(detection_model , image_np )

    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    # detections= detector(img_pro_test_img["detection_img_tensor"])
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        output_dict['detection_boxes'],
        output_dict['detection_classes'],
        output_dict['detection_scores'],
        category_index,
        instance_masks=output_dict.get('detection_masks_reframed', None),
        use_normalized_coordinates=True,
        line_thickness=6)

    path_saved = PATH_FOLDER_IMG_TEST_EVALED + "/" +path_test_img_path
    print("SAVED image: ",path_saved)
    Image.fromarray(image_np).save(path_saved)

    # draw_boxes_s(
    #     path_test_img['origin_img_np'],
    #     detections['detection_classes'][0],
    #     detections['detection_boxes'][0],
    #     detections["detection_scores"][0],#.numpy(),
    #     0.4, path_saved,label_map = category_index )

