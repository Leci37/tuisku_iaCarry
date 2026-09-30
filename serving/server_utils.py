import logging
import os
import uuid

import numpy as np
import pandas as pd
from PIL import Image

from visualization_utils import plot_detections


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

import json
import numpy as np
class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def load_image_into_numpy_array_by_Image_open(list_paths):
    Test_images_np = []
    for path in list_paths:
        Test_images_np.append( np.array(Image.open(path)) )
    return Test_images_np

def register_MULTI_in_zTelegram_Registers(  df_r, PATH_REGISTER_RESULT_REAL_TIME ):
    if os.path.isfile(PATH_REGISTER_RESULT_REAL_TIME):
        df_r.to_csv(PATH_REGISTER_RESULT_REAL_TIME, sep="\t", mode='a', header=False)
    else:
        df_r.to_csv(PATH_REGISTER_RESULT_REAL_TIME, sep="\t")
        print("Created File Registers : " + PATH_REGISTER_RESULT_REAL_TIME)


def change_format_dict_json_to_client(detections, img_np_raw, path_img_box, MIN_SCORE_TO_CLIENT = 0.5, rid = ""):
    """Shape the detector output into the JSON contract the checkout page reads.

    `rid` is the request id minted in app.upload_file(); it ties
    these lines to the route's and the detector's for the same frame.
    """
    tag = "[SHAPE][%s]" % (rid or "-")
    list_predict = []
    n_raw = 0
    n_dropped = 0
    tally = {}
    for (bbox, score, int_class, str_class) in zip(detections['detection_boxes'], detections['detection_scores'],detections['detection_classes'],detections['detection_classes_name']):
        n_raw += 1
        if score > MIN_SCORE_TO_CLIENT:  # min score to sent to the cliente
            # print(bbox, score, int_class, str_class)
            # NOTE: these key names are wrong and the whole stack depends on it.
            # TensorFlow gives [ymin, xmin, ymax, xmax], so 'width' holds the
            # RIGHT edge and 'height' the BOTTOM edge, both normalised 0-1. The
            # browser subtracts to get real sizes; do not "fix" one side alone.
            dict_bbox = {'left': bbox[1], 'top': bbox[0], 'width': bbox[3],'height': bbox[2]}  # cuaidado con mover los numeros
            dict_predictor = {'probability': score, 'tagInt': int_class, 'tagName': str_class, 'boundingBox': dict_bbox}
            list_predict.append(dict_predictor)
            tally[str_class] = tally.get(str_class, 0) + 1
        else:
            n_dropped += 1
    dict_to_respond = {'path_server': path_img_box, 'shape_img': img_np_raw.shape, "predictions": list_predict}

    logging.info("%s shaped response: %d raw -> %d sent, %d below MIN_SCORE_TO_CLIENT=%s",
                 tag, n_raw, len(list_predict), n_dropped, MIN_SCORE_TO_CLIENT)
    logging.info("%s per class sent: %s", tag,
                 ", ".join("%s x%d" % (k, v) for k, v in sorted(tally.items())) or "(none)")
    # There are TWO thresholds in this system and only one of them is here. The
    # browser filters again at its own DETECTION_MIN_SCORE, so anything scoring
    # between the two is sent and never shown. Logging both numbers in one place
    # is the only way that gap is visible without reading two codebases.
    logging.info("%s note: the browser re-filters these at its own, higher threshold; "
                 "counts on screen can legitimately be lower than %d", tag, len(list_predict))
    if not list_predict:
        logging.warning("%s NOTHING above MIN_SCORE_TO_CLIENT=%s - the screen will show "
                        "'no products recognised' and block payment", tag, MIN_SCORE_TO_CLIENT)
    return dict_to_respond

from werkzeug.utils import secure_filename
def save_img_loaded(f, path_upload , date_name_folder, rid = "" ):
    tag = "[SAVE][%s]" % (rid or "-")
    folder_path = path_upload + "/img_" + date_name_folder
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        logging.info("%s created upload folder %s", tag, folder_path)
    filename = secure_filename(f.filename)
    if filename != f.filename:
        # secure_filename rewrites spaces and parentheses, so the name on disk
        # need not match what the browser sent — worth seeing when hunting a file.
        logging.info("%s filename sanitised: %r -> %r", tag, f.filename, filename)
    save_path = os.path.join(folder_path, filename)
    f.save(save_path)
    logging.info("%s wrote %s", tag, save_path)
    return filename, save_path

def manage_plot_and_save_img_predicted(detections, df, NUM_CHECK_POINT_OR_NAME_FOLDER, PATH_TO_SAVED, Category_index, test_images_np, i_name_path_png, MIN_SCORE=0.4, LABEL_ID_OFFSET =1):
    list_good_score = [x for x in detections['detection_scores'][0] if x > MIN_SCORE]
    if not os.path.exists(PATH_TO_SAVED + "/img_" + str(NUM_CHECK_POINT_OR_NAME_FOLDER)):
        os.makedirs(PATH_TO_SAVED + "/img_" + str(NUM_CHECK_POINT_OR_NAME_FOLDER))
    PATH_IMG_SAVE_TESTED = PATH_TO_SAVED + "/img_" + str(NUM_CHECK_POINT_OR_NAME_FOLDER) + "/"+i_name_path_png+"__" + str(len(list_good_score)) + "_.jpg"
    print("\t", bcolors.OKGREEN + PATH_IMG_SAVE_TESTED + bcolors.ENDC)
    id_img =  str(uuid.uuid4().fields[-1])[:6]
    for j in range(0, len(list_good_score)):
        print("\t\tProbability: ", '{:.2f}'.format(detections['detection_scores'][0][j].numpy()),
              "\tClasses: ", detections['detection_classes'][0][j].numpy().astype(np.uint32) + LABEL_ID_OFFSET,
              "\tBoxes: ", detections['detection_boxes'][0][j].numpy())
        int_class = detections['detection_classes'][0][j].numpy().astype(np.uint32) + LABEL_ID_OFFSET
        dict_imgA = {"id_img" : id_img, "Path": PATH_IMG_SAVE_TESTED,
                     "Probability": ('{:.2f}'.format(detections['detection_scores'][0][j].numpy())),
                     "Classes": int_class, "Type_prod": [Category_index[x] for x in Category_index if Category_index[x]['id'] == int_class][0]['name'],
                     "Boxes": detections['detection_boxes'][0][j].numpy(), "shape": test_images_np.shape}
        df = pd.concat([df, pd.DataFrame([dict_imgA])], ignore_index=True)
    plot_detections(
        test_images_np[0],
        detections['detection_boxes'][0].numpy(),
        detections['detection_classes'][0].numpy().astype(np.uint32) + LABEL_ID_OFFSET,
        detections['detection_scores'][0].numpy(),
        Category_index, min_score_thresh=MIN_SCORE, figsize=(15, 20), image_name=PATH_IMG_SAVE_TESTED)
    return df, PATH_IMG_SAVE_TESTED
