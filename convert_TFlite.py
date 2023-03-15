# https://techzizou.com/build-android-app-for-custom-object-detection-using-tf2/#tf2_step21

'''*******************************
   FOR FLOATING-POINT INFERENCE
**********************************'''

import tensorflow as tf
import numpy as np

import Utils_TFlite_see_info
from convert_TFlite_mdata import mdata_write_all_in_tflite_SIMPLE, log_metadata_info_tflite


def Add_metadata(model_path):
    save_path_sim = model_path.replace(".tflite", "_Mdata.tflite")
    print("ADD metadata: ", save_path_sim)
    try:
        mdata_write_all_in_tflite_SIMPLE(MODEL_PATH_LITE=model_path, SAVE_TO_PATH_LITE=save_path_sim, LABEL_FILE=_LABEL_FILE)
        log_metadata_info_tflite(save_path_sim)
    except Exception as e:
        print("Expection: ", e)
    print("\n --------------\n")

_LABEL_FILE = "model_101_C/label_map.txt"
PATH_MODEL = "model_101_C/frozen" #  "ssd_resnet50_v1_fpn_640x640_coco17_tpu-8"
SIGNATURE_KEY = 'serving_default'# 'detect' #'serving_default'


'''*******************************
   FOR FLOATING-POINT INFERENCE
**********************************'''

import tensorflow as tf

print("REQUERIERE ")
print("REQUERIERE !python export_tflite_graph_tf2.py  --pipeline_config_path=model_101_C/pipeline.config  --trained_checkpoint_dir model_101_C/checkpoint  --output_directory model_101_C/frozen")


TF_LITE_PATH = PATH_MODEL+'/model_simple_sigNo.tflite'

print("Load model: ", PATH_MODEL+"/saved_model")
converter = tf.lite.TFLiteConverter.from_saved_model(PATH_MODEL+"/saved_model"  )
tflite_model = converter.convert()
open(TF_LITE_PATH, "wb").write(tflite_model)
Utils_TFlite_see_info.Log_TFlite_info_file(TF_LITE_PATH)
Add_metadata(TF_LITE_PATH)



'''**************************************************
#  FOR FLOATING-POINT INFERENCE WITH OPTIMIZATIONS
#**************************************************'''
print("Load model: ", PATH_MODEL+"/saved_model")
converter_F = tf.lite.TFLiteConverter.from_saved_model(PATH_MODEL+"/saved_model",signature_keys=[SIGNATURE_KEY])
converter_F.optimizations = [tf.lite.Optimize.DEFAULT]
converter_F.experimental_new_converter = True
converter_F.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,
                                       tf.lite.OpsSet.SELECT_TF_OPS]
tflite_model = converter_F.convert()

TF_LITE_PATH_2 = PATH_MODEL+'/model_Build_sigDef.tflite'
with tf.io.gfile.GFile(TF_LITE_PATH_2, 'wb') as f:
  f.write(tflite_model)
Utils_TFlite_see_info.Log_TFlite_info_file(TF_LITE_PATH_2)
Add_metadata(TF_LITE_PATH_2)


'''**********************************
    FOR DYNAMIC RANGE QUANTIZATION 
*************************************
 The model is now a bit smaller with quantized weights, but other variable data is still in float format.'''
print("Load model: ", PATH_MODEL+"/saved_model")
converter_D = tf.lite.TFLiteConverter.from_saved_model(PATH_MODEL+"/saved_model",signature_keys=[SIGNATURE_KEY])
converter_D.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_quant_model = converter_D.convert()

TF_LITE_PATH = PATH_MODEL+'/model_Default_sigDef.tflite'
with tf.io.gfile.GFile(TF_LITE_PATH, 'wb') as f:
  f.write(tflite_quant_model)
Utils_TFlite_see_info.Log_TFlite_info_file(TF_LITE_PATH)
Add_metadata(TF_LITE_PATH)



'''**********************************
    normalized_input_image_tensor
*************************************'''
print("REQUERIERE ")
print("REQUERIERE !python exporter_main_v2.py --trained_checkpoint_dir=/mydrive/customTF2/training --pipeline_config_path=/content/gdrive/MyDrive/customTF2/data/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.config --output_directory /mydrive/customTF2/data/inference_graph")
#https://stackoverflow.com/questions/59679645/converting-saved-model-to-tflite-model-using-tf-2-0
#Diferente la forma de cargar tf.compat.v1.lite.TFLiteConverter.from_frozen_graph
converter = tf.compat.v1.lite.TFLiteConverter.from_frozen_graph(PATH_MODEL+"/saved_model",input_shapes = {'normalized_input_image_tensor':[1,640,640,3]},
    input_arrays = ['normalized_input_image_tensor'],output_arrays = ['TFLite_Detection_PostProcess','TFLite_Detection_PostProcess:1',
    'TFLite_Detection_PostProcess:2','TFLite_Detection_PostProcess:3'])
converter.allow_custom_ops=True
# Convert the model to quantized TFLite model.
converter.optimizations =  [tf.lite.Optimize.DEFAULT]
tflite_model_D = converter.convert()
TF_LITE_PATH_D = PATH_MODEL+'/model_PostProcess_sigNo.tflite'
with tf.io.gfile.GFile(TF_LITE_PATH_D, 'wb') as f:
  f.write(tflite_model_D)
Utils_TFlite_see_info.Log_TFlite_info_file(TF_LITE_PATH_D)
Add_metadata(TF_LITE_PATH_D)