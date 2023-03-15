
import tensorflow as tf




# As mentioned in the TensorFlow Lite docs, you need to use a tf.lite.Interpreter to parse a .tflite model.
def Log_TFlite_info_file(SAVED_MODEL_PATH_LITE, SAVED_MODEL_FILE_INFO_JSON = None):

    if SAVED_MODEL_FILE_INFO_JSON is None:
        SAVED_MODEL_FILE_INFO_JSON = SAVED_MODEL_PATH_LITE + ".json"

    print("Load: ", SAVED_MODEL_PATH_LITE)
    # Load TFLite model and allocate tensors.
    interpreter = tf.lite.Interpreter(model_path=SAVED_MODEL_PATH_LITE)
    # interpreter.allocate_tensors()
    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    # https://stackoverflow.com/questions/70424582/how-to-visualize-feature-maps-of-a-tensorflow-lite-model
    # Gives you all tensor details index-wise. you will find all the quantization_parameters here
    details = interpreter.get_tensor_details()
    # Allocate tensors
    interpreter.allocate_tensors()
    # Print the input and output details of the model
    print()
    print("Input details:")
    print(input_details)
    print()
    print("Output details:")
    print(output_details)
    print()
    # Print the signatures from the converted model
    signatures = interpreter.get_signature_list()
    print('Signature:', signatures)
    interpreter.get_tensor_details()
    signatures = interpreter.get_signature_list()
    print(signatures)
    json_info_data = "INPUT:\n" + str(input_details) + "\n\n" + "OUTPUT:\n" + str(
        output_details) + "\n\n" + "SIGNATURES:\n" + str(signatures) + "\n\n\n\n" + "DETAILS:\n" + str(details) + "\n\n"
    json_info_data = json_info_data.replace(", '", ",\n\t'")
    # print("save info: ", SAVED_MODEL_FILE_INFO_JSON)
    with open(SAVED_MODEL_FILE_INFO_JSON, 'w') as filetowrite:
        filetowrite.write(json_info_data)
    print("saved info: ", SAVED_MODEL_FILE_INFO_JSON)




# list_tflite = ['model_cat_dog_zombie/frozen_tflite/TFLite_Detection.tflite']#,
#                #  'model_cat_dog_zombie/ANdroid_tflite/detect.tflite',
#                # 'model_cat_dog_zombie/ANdroid_tflite/efficientdet-lite2.tflite',
#                # 'model_cat_dog_zombie/ANdroid_tflite/ssd_mobilenet_v1.tflite']
#
# for l in list_tflite:
#     Log_TFlite_info_file(l)