# https://pravash-techie.medium.com/python-singleton-pattern-for-effective-object-management-49d62ec3bd9b
import glob
import pickle
import time
from datetime import datetime
import numpy as np
import pandas as pd
from PIL import Image

import tensorflow as tf

import utils_tuisku_server
from utils_tuisku_server import bcolors

import logging

from threading import Lock
inc_lock = Lock()

class Detector_model:
    SIGNATURE_REF = "detect"
    PATH_TO_SAVED_MODEL_INTERFACE_GRAPH = "model_efi_d1C/save_model_sig_54"  # TODO "model_efi_d1C/save_model_detect29"
    PATH_PICKLE_CAT_INDEX = 'model_efi_d1C/P_Category_index.pickle'

    _instance = None
    detector = None
    Category_index = None
    _test_images_np = []

    # def __new__(cls): #singletone
    #     if cls._instance is None:
    #         logging.info("[SGTON] Singletone instance is None. Load it. \tPath: "+cls.PATH_TO_SAVED_MODEL_INTERFACE_GRAPH + " Signature_ref: "+ cls.SIGNATURE_REF)
    #         cls._instance = super(Detector_model, cls).__new__(cls)
    #         cls._instance._initialize()
    #     else:
    #         logging.info("[SGTON] Singletone instance is NOT None. RELoad existing it. \tPath: "+cls.PATH_TO_SAVED_MODEL_INTERFACE_GRAPH + " Signature_ref: "+ cls.SIGNATURE_REF)
    #         logging.debug("[SGTON] DEBUG Instance. Name: "+cls._instance.detector._as_name_attr_list.name +  " Descriptor: "+ str(cls._instance.detector._as_name_attr_list.DESCRIPTOR ) )
    #     return cls._instance

    def __init__(self):
        # self.log_file = open("log.txt", "a")
        logging.info("\n" + bcolors.OKBLUE + self.PATH_TO_SAVED_MODEL_INTERFACE_GRAPH + bcolors.ENDC)
        logging.info('[SGTON] Loading model...')
        detection_model = tf.saved_model.load(self.PATH_TO_SAVED_MODEL_INTERFACE_GRAPH)
        logging.info('[SGTON] Done!')
        signatures = detection_model.signatures
        logging.info('\n [SGTON]' + bcolors.OKCYAN + ' Signature:' + bcolors.ENDC+str( signatures ) )
        logging.debug("\t [SGTON]Para que el .pb cargado se genere las signatures['" + self.SIGNATURE_REF + "'] hay que Congelarle Frozen  con exporter_main_v2.py ")
        logging.debug("\t [SGTON]" + bcolors.OKCYAN + ' Structured_outputs:' + bcolors.ENDC +str(detection_model.signatures[self.SIGNATURE_REF].structured_outputs) )
        logging.debug("\t [SGTON]" + bcolors.OKCYAN + ' Structured_ Types:' + bcolors.ENDC+str(detection_model.signatures[self.SIGNATURE_REF].output_dtypes) )
        logging.debug("\t [SGTON]" + bcolors.OKCYAN + ' Structured_ Shapes:' + bcolors.ENDC+str(detection_model.signatures[self.SIGNATURE_REF].output_shapes) )
        # TUTORIAL FOR serving_default https://www.mygreatlearning.com/blog/object-detection-using-tensorflow/
        self.detector = detection_model.signatures[self.SIGNATURE_REF]
        logging.info("[SGTON]" + bcolors.OKCYAN + ' FUNCION_detector:' + bcolors.ENDC+ str(self.detector) + "\n")

        _test_images_np = utils_tuisku_server.load_image_into_numpy_array_by_Image_open([r"..\iacarry-evaluation\zTest_img.jpg"])
        logging.debug("\n[SGTON] Do a test 1º prediction to full load Img_shape: " + str(_test_images_np[0].shape))
        input_tensor = tf.convert_to_tensor(_test_images_np[0],dtype=tf.float32)  # necesario si  np.array(Image.open(path))
        input_tensor = input_tensor[tf.newaxis, ...]  # necesario si  np.array(Image.open(path))  Es necesario para pb exporter_main_v2.py , Crepo que es por usar esta estructura de carga y no cv2:  np.array(Image.open(path))
        detections_xxx = self.detector(input_tensor=input_tensor)
        logging.info("[SGTON] Do a test prediction Completed Num_preditc "+ str([d for d in list( detections_xxx['detection_scores'].numpy()[0] ) if d >0.5 ]))

        logging.debug("\n[SGTON] Category_index.pickle Path: " + self.PATH_PICKLE_CAT_INDEX)
        with open(self.PATH_PICKLE_CAT_INDEX, 'rb') as handle:
            self.Category_index = pickle.load(handle)
        str_cat = [str(x[1]['id'])+":"+str(x[1]['name'])  for x in self.Category_index.items() ]
        logging.info("\n[SGTON] Category_index.pickle loaded \tNum_keys: "+str(len(str_cat)) + " \tNames: "+ ", ".join(str_cat))

    MIN_SCORE = 0.5
    PATH_TO_SAVED = "..\iacarry-evaluation\_upload_img_bbox_results" # "model_efi_d1C"
    NUM_CHECK_POINT_OR_NAME_FOLDER = datetime.now().strftime("%Y_%m_%d")
    PATH_REGISTER_RESULT_MULTI_REAL_TIME = PATH_TO_SAVED + "/predi_MULTI_real_time_" + datetime.now().strftime("%Y_%m_%d") + ".csv"
    def do_prediction_from_list_paths(self, path_img, rid = ""):
        """Run the detector over one frame.

        `rid` is the request id minted by RUN_server_py_upload.upload_file(). It
        is stamped on every line here so a single purchase can be followed from
        the browser console, through the Flask route, into the model.
        """
        tag = "[SGTON][%s]" % (rid or "-")
        t_start = time.perf_counter()
        df = pd.DataFrame()
        _test_images_np = np.array(Image.open(path_img))
        name_save_with_boxes = str(path_img.split("\\")[-1].split(".")[0])
        logging.info("%s START inference img=%s shape=%s dtype=%s",
                     tag, path_img, _test_images_np.shape, _test_images_np.dtype)
        # _test_images_np = utils_tuisku_server.load_image_into_numpy_array_by_Image_open(list_img)
        # for i in range(len(_test_images_np)):#TODO seguro que queremos una lista (una lista de uno en muchos casos ¿?)
        # The lock serialises inference across requests. Time spent WAITING here
        # is queueing, not compute; the two look identical from the browser and
        # must not be confused when someone reports "the screen is slow".
        t_wait = time.perf_counter()
        with inc_lock:# CRITICAL SECTION in server¿?
            t_held = time.perf_counter()
            logging.info("%s lock acquired after %.3fs of queueing", tag, t_held - t_wait)
            input_tensor = tf.convert_to_tensor(_test_images_np,dtype=tf.float32)  # necesario si  np.array(Image.open(path))
            input_tensor = input_tensor[tf.newaxis, ...]  # necesario si  np.array(Image.open(path))  Es necesario para pb exporter_main_v2.py , Crepo que es por usar esta estructura de carga y no cv2:  np.array(Image.open(path))
            t_infer = time.perf_counter()
            logging.debug("%s tensor ready %s in %.3fs", tag, input_tensor.shape, t_infer - t_held)
            detections = self.detector(input_tensor=input_tensor)
            t_done = time.perf_counter()
            logging.info("%s inference done in %.3fs (lock held %.3fs)",
                         tag, t_done - t_infer, t_done - t_held)

        img_format_viz = np.array([_test_images_np])  # necesario si  np.array(Image.open(path))
        t_plot = time.perf_counter()
        df, path_img_box = utils_tuisku_server.manage_plot_and_save_img_predicted(detections, df, self.NUM_CHECK_POINT_OR_NAME_FOLDER,PATH_TO_SAVED=self.PATH_TO_SAVED,Category_index=self.Category_index,
                                                                    test_images_np=img_format_viz, i_name_path_png=name_save_with_boxes,MIN_SCORE=self.MIN_SCORE, LABEL_ID_OFFSET=1)
        # This render is pure server-side bookkeeping: the browser draws its own
        # boxes and never fetches path_img_box, so its cost is latency the
        # customer pays for nothing they will see.
        logging.info("%s figure rendered and saved in %.3fs -> %s",
                     tag, time.perf_counter() - t_plot, path_img_box)

        if df.shape[0] != 0:
            path_to_csv = self.PATH_TO_SAVED + "/img_" + str(self.NUM_CHECK_POINT_OR_NAME_FOLDER) + "/df_test_img.csv"
            utils_tuisku_server.register_MULTI_in_zTelegram_Registers(df_r=df, PATH_REGISTER_RESULT_REAL_TIME=path_to_csv)
            logging.info("%s appended %d row(s) to %s", tag, df.shape[0], path_to_csv)
        else:
            logging.warning("%s no detection above MIN_SCORE=%s - nothing written to the CSV register",
                            tag, self.MIN_SCORE)

        #casto to numpy
        for k in detections.keys():
            detections[k] = detections[k].numpy()[0]  # The [0] is only for ONE
        # remove useless columns
        KEYS_COLUMNS_USELESS_DETECTOR = ['detection_anchor_indices', 'detection_multiclass_scores', 'raw_detection_boxes', 'raw_detection_scores']
        for key in KEYS_COLUMNS_USELESS_DETECTOR:
            detections.pop(key, None)

        list_cat_names = []
        n_unmapped = 0
        for j in range(0, len(detections['detection_classes']  )):
            int_class = int(detections['detection_classes'][j]) + 1
            match = [self.Category_index[x] for x in self.Category_index if self.Category_index[x]['id'] == int_class]
            if match:
                cat_name = match[0]['name']
            else:
                # Previously this indexed [0] straight into an empty list and
                # raised, turning one unmapped class id into a 500 and a blank
                # cart. The browser already skips tags it does not know, so a
                # synthetic name degrades into one skipped product instead.
                cat_name = "unknown_class_%d" % int_class
                n_unmapped += 1
            list_cat_names.append(cat_name)
        detections['detection_classes_name'] = list_cat_names
        if n_unmapped:
            logging.error("%s %d detection(s) had a class id absent from Category_index "
                          "(%d known ids) - sent as unknown_class_* and skipped by the client",
                          tag, n_unmapped, len(self.Category_index))

        # The one line that says what the model actually saw.
        scores = detections['detection_scores']
        n_above = int(sum(1 for s in scores if s > self.MIN_SCORE))
        top = sorted(zip(list_cat_names, [float(s) for s in scores]), key=lambda p: -p[1])[:8]
        logging.info("%s RESULT raw=%d above MIN_SCORE=%s: %d | top: %s | total %.3fs",
                     tag, len(scores), self.MIN_SCORE, n_above,
                     ", ".join("%s=%.2f" % (n, s) for n, s in top) or "(none)",
                     time.perf_counter() - t_start)

        return img_format_viz, _test_images_np, detections, df, path_img_box

    def close(self):
        logging.info("[SGTON] Close instance")
        # self.log_file.close()


# class DetectorFactory:
#     def create_instance_detector(self):
#         return Detector_model()


# Usage
# if __name__ == "__main__":
#     logging.basicConfig(level=logging.DEBUG, filename=r"C:\web-servers\iacarry-evaluation\server_predict_model_singletone.log", filemode="a+", format="%(asctime)-15s %(levelname)-8s %(message)s")
#     # PATHS_IMG = glob.glob(r'E:\iaCarry_img_eroski\_EVAL_img\*')
#     # logging.info("Load Images Num: "+ len(PATHS_IMG)+ " from: "+ r'C:\Users\tuisku\PycharmProjects\_EVAL_img\*')
#
#     detector_factory = DetectorFactory()
#
#     detector1 = detector_factory.create_instance_detector()
#     detector2 = detector_factory.create_instance_detector()
#
#     assert detector1 == detector2 , "[SGTON] The DetectorFactory() did not instance the same object in Singletone pathern"
#
#     detector1.do_prediction_from_list_paths(r"C:\web-servers\iacarry-evaluation\zTest_img.jpg")
#
#     logging.info("end")
