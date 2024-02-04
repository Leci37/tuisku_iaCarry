# https://pravash-techie.medium.com/python-singleton-pattern-for-effective-object-management-49d62ec3bd9b
import glob
import pickle
from datetime import datetime
import numpy as np
import pandas as pd
from PIL import Image

import tensorflow as tf

import utils_tuisku_server
from utils_tuisku_server import bcolors

import logging


class Class_Count:
    SIGNATURE_REF = "detect"
    # PATH_TO_SAVED_MODEL_INTERFACE_GRAPH = "model_efi_d1C/save_model_sig_54"  # TODO "model_efi_d1C/save_model_detect29"
    # PATH_PICKLE_CAT_INDEX = 'model_efi_d1C/P_Category_index.pickle'
    #
    # _instance = None
    # detector = None
    # Category_index = None
    # _test_images_np = []

    counter_call = 0
    counter_detect = 0
    list_elements = []


    def __init__(self):
        self.counter_call = 0
        self.counter_detect = 0
        self.list_elements = []
        logging.info("Class_Count() STATUS counter_call:" + str(self.counter_call) + " counter_detect: " + str(
            self.counter_detect) + " list_elements: " + str(self.list_elements))

    def plus_counter_call(self, n):
        logging.info("plus_counter_call() CREATED: " + str(n))
        self.counter_call = self.counter_call +n
        logging.info("Class_Count() STATUS counter_call:" + str(self.counter_call) + " counter_detect: " + str(self.counter_detect) + " list_elements: " + str(self.list_elements) )

    def plus_counter_detect(self, n, element):
        logging.info("plus_counter_detect() ADDED: " + str(n)+ " Element: "+element)
        self.counter_detect = self.counter_detect +n
        self.list_elements.append(element)
        logging.info("Class_Count() STATUS counter_call:" + str(self.counter_call) + " counter_detect: " + str(
            self.counter_detect) + " list_elements: " + str(self.list_elements))



    # def __new__(cls): #singletone
    #     if cls._instance is None:
    #         logging.info("[SGTON] Singletone instance is None. Load it. \tPath: "+cls.PATH_TO_SAVED_MODEL_INTERFACE_GRAPH + " Signature_ref: "+ cls.SIGNATURE_REF)
    #         cls._instance = super(Detector_model, cls).__new__(cls)
    #         cls._instance._initialize()
    #     else:
    #         logging.info("[SGTON] Singletone instance is NOT None. RELoad existing it. \tPath: "+cls.PATH_TO_SAVED_MODEL_INTERFACE_GRAPH + " Signature_ref: "+ cls.SIGNATURE_REF)
    #         logging.debug("[SGTON] DEBUG Instance. Name: "+cls._instance.detector._as_name_attr_list.name +  " Descriptor: "+ str(cls._instance.detector._as_name_attr_list.DESCRIPTOR ) )
    #     return cls._instance
