# https://techzizou.com/build-android-app-for-custom-object-detection-using-tf2/#tf2_step21


# '''*********************************
#   FOR FULL INTEGER QUANTIZATION
# ************************************
# The internal quantization remains the same as previous float fallback quantization method,
# but you can see the input and output tensors here are also now integer format'''
import os
from tflite_support.metadata_writers import object_detector
from tflite_support import metadata
import flatbuffers
from tensorflow_lite_support.metadata import metadata_schema_py_generated as _metadata_fb
from tensorflow_lite_support.metadata.python import metadata as _metadata
from tensorflow_lite_support.metadata.python.metadata_writers import writer_utils

def mdata_write_all_in_tflite_SIMPLE(MODEL_PATH_LITE, SAVE_TO_PATH_LITE, LABEL_FILE ):
    writer = object_detector.MetadataWriter.create_for_inference(
        writer_utils.load_file(MODEL_PATH_LITE), input_norm_mean=[127.5],
        input_norm_std=[127.5], label_file_paths=[LABEL_FILE])
    writer_utils.save_file(writer.populate(), SAVE_TO_PATH_LITE)

    print("saved! SIMPLE SAVE_TO_PATH: ", SAVE_TO_PATH_LITE)

ObjectDetectorWriter = object_detector.MetadataWriter

def mdata_write_all_in_tflite_FULL(MODEL_PATH_LITE, SAVE_TO_PATH_LITE, LABEL_FILE ):
    print("MODEL_PATH: ", MODEL_PATH_LITE)
    writer = ObjectDetectorWriter.create_for_inference(
        writer_utils.load_file(MODEL_PATH_LITE), [127.5], [127.5], [LABEL_FILE])
    writer_utils.save_file(writer.populate(), SAVE_TO_PATH_LITE)
    # Verify the populated metadata and associated files.
    displayer = metadata.MetadataDisplayer.with_model_file(SAVE_TO_PATH_LITE)
    print("Metadata populated:")
    print(displayer.get_metadata_json())
    print("Associated file(s) populated:")
    print(displayer.get_packed_associated_file_list())
    model_meta = _metadata_fb.ModelMetadataT()
    model_meta.name = "SSD_Detector " + MODEL_PATH_LITE
    model_meta.description = (
        "Identify which of a known set of objects might be present and provide "
        "information about their positions within the given image or a video "
        "stream.")
    # Creates input info.
    input_meta = _metadata_fb.TensorMetadataT()
    input_meta.name = "image"
    input_meta.content = _metadata_fb.ContentT()
    input_meta.content.contentProperties = _metadata_fb.ImagePropertiesT()
    input_meta.content.contentProperties.colorSpace = (
        _metadata_fb.ColorSpaceType.RGB)
    input_meta.content.contentPropertiesType = (
        _metadata_fb.ContentProperties.ImageProperties)
    input_normalization = _metadata_fb.ProcessUnitT()
    input_normalization.optionsType = (
        _metadata_fb.ProcessUnitOptions.NormalizationOptions)
    input_normalization.options = _metadata_fb.NormalizationOptionsT()
    input_normalization.options.mean = [127.5]
    input_normalization.options.std = [127.5]
    input_meta.processUnits = [input_normalization]
    input_stats = _metadata_fb.StatsT()
    input_stats.max = [255]
    input_stats.min = [0]
    input_meta.stats = input_stats
    # Creates outputs info.
    output_location_meta = _metadata_fb.TensorMetadataT()
    output_location_meta.name = "location"
    output_location_meta.description = "The locations of the detected boxes."
    output_location_meta.content = _metadata_fb.ContentT()
    output_location_meta.content.contentPropertiesType = (
        _metadata_fb.ContentProperties.BoundingBoxProperties)
    output_location_meta.content.contentProperties = (
        _metadata_fb.BoundingBoxPropertiesT())
    output_location_meta.content.contentProperties.index = [1, 0, 3, 2]
    output_location_meta.content.contentProperties.type = (
        _metadata_fb.BoundingBoxType.BOUNDARIES)
    output_location_meta.content.contentProperties.coordinateType = (
        _metadata_fb.CoordinateType.RATIO)
    output_location_meta.content.range = _metadata_fb.ValueRangeT()
    output_location_meta.content.range.min = 2
    output_location_meta.content.range.max = 2
    output_class_meta = _metadata_fb.TensorMetadataT()
    output_class_meta.name = "category"
    output_class_meta.description = "The categories of the detected boxes."
    output_class_meta.content = _metadata_fb.ContentT()
    output_class_meta.content.contentPropertiesType = (
        _metadata_fb.ContentProperties.FeatureProperties)
    output_class_meta.content.contentProperties = (
        _metadata_fb.FeaturePropertiesT())
    output_class_meta.content.range = _metadata_fb.ValueRangeT()
    output_class_meta.content.range.min = 2
    output_class_meta.content.range.max = 2
    label_file = _metadata_fb.AssociatedFileT()
    label_file.name = os.path.basename("tflite_label_map.txt")
    label_file.description = "Label of objects that this model can recognize."
    label_file.type = _metadata_fb.AssociatedFileType.TENSOR_VALUE_LABELS
    output_class_meta.associatedFiles = [label_file]
    output_score_meta = _metadata_fb.TensorMetadataT()
    output_score_meta.name = "score"
    output_score_meta.description = "The scores of the detected boxes."
    output_score_meta.content = _metadata_fb.ContentT()
    output_score_meta.content.contentPropertiesType = (
        _metadata_fb.ContentProperties.FeatureProperties)
    output_score_meta.content.contentProperties = (
        _metadata_fb.FeaturePropertiesT())
    output_score_meta.content.range = _metadata_fb.ValueRangeT()
    output_score_meta.content.range.min = 2
    output_score_meta.content.range.max = 2
    output_number_meta = _metadata_fb.TensorMetadataT()
    output_number_meta.name = "number of detections"
    output_number_meta.description = "The number of the detected boxes."
    output_number_meta.content = _metadata_fb.ContentT()
    output_number_meta.content.contentPropertiesType = (
        _metadata_fb.ContentProperties.FeatureProperties)
    output_number_meta.content.contentProperties = (
        _metadata_fb.FeaturePropertiesT())
    # Creates subgraph info.
    group = _metadata_fb.TensorGroupT()
    group.name = "detection result"
    group.tensorNames = [
        output_location_meta.name, output_class_meta.name,
        output_score_meta.name
    ]
    subgraph = _metadata_fb.SubGraphMetadataT()
    subgraph.inputTensorMetadata = [input_meta]
    subgraph.outputTensorMetadata = [
        output_location_meta, output_class_meta, output_score_meta,
        output_number_meta
    ]
    subgraph.outputTensorGroups = [group]
    model_meta.subgraphMetadata = [subgraph]
    b = flatbuffers.Builder(0)
    b.Finish(
        model_meta.Pack(b),
        _metadata.MetadataPopulator.METADATA_FILE_IDENTIFIER)
    metadata_buf = b.Output()
    print("saved! FULL SAVE_TO_PATH: ", SAVE_TO_PATH_LITE)



def log_metadata_info_tflite(save_path):
    displayer = metadata.MetadataDisplayer.with_model_file(save_path)
    str_matadata = displayer.get_metadata_json()
    with open(save_path.replace(".tflite", "_mdata.json"), 'w') as filetowrite:
        filetowrite.write(str_matadata)
    print("Metadata populated in :", save_path.replace(".tflite", "_mdata.json"))

# PATH_MODEL = "escarabajo"
# _MODEL_PATH = PATH_MODEL+"/model_G3.tflite"
# _LABEL_FILE = "model_cat_dog_zombie/mscoco_label_map.pbtxt"
# # _SAVE_TO_PATH = _MODEL_PATH.replace("model_G3", "model_mdata_G3")
#


 #"model_cat_dog_zombie_mobil/model_with_metadata_G3.tflite"

# LIST_TFlite = []
# import glob
#
# # root_dir needs a trailing slash (i.e. /root/dir/)
# for filename in glob.iglob("model_cat_dog_zombie/frozen" + '**/**', recursive=True):
#
#     if filename.endswith(".tflite"):
#         print(filename)
#         LIST_TFlite.append(filename)
#
#
#
# for l in LIST_TFlite:
#     model_path =   l
#     save_path_sim = model_path.replace(".tflite", "_mdataSimple.tflite")
#     print(save_path_sim)
#     try:
#         mdata_write_all_in_tflite_SIMPLE(MODEL_PATH_LITE=model_path, SAVE_TO_PATH_LITE=save_path_sim, LABEL_FILE=_LABEL_FILE)
#         log_metadata_info_tflite(save_path_sim)
#     except Exception as e:
#         print("Expection: ", e)
#
#     save_path_full = model_path.replace(".tflite", "_mdataFULL.tflite")
#     print(save_path_full)
#     try:
#         mdata_write_all_in_tflite_SIMPLE(MODEL_PATH_LITE=model_path, SAVE_TO_PATH_LITE=save_path_full, LABEL_FILE=_LABEL_FILE)
#         log_metadata_info_tflite(save_path_full)
#     except Exception as e:
#         print("Expection: ", e)





