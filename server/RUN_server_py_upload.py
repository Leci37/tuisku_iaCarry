import os
import logging
import pickle
from datetime import datetime
# from gevent.pywsgi import WSGIServer

import  utils_tuisku_server

logging.basicConfig(level=logging.DEBUG, filename=r"..\iacarry-evaluation\zlog_"+datetime.now().strftime("%Y_%m_%d"), filemode="a+",
                    format="s%(asctime)-15s %(process)d:%(threadName)s %(levelname)-8s %(message)s") # para thread id no necesario :%(thread)d
# set up logging to console
console = logging.StreamHandler()
console.setLevel(logging.DEBUG)
# set a format which is simpler for console use
formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
console.setFormatter(formatter)
# add the handler to the root logger
logging.getLogger('').addHandler(console)
logger = logging.getLogger(__name__)
logging.info("-------------START-------------")
import psutil;   #os.getppid() Return the parent’s process id.
logging.info("-------------Parent's PID: "+str(psutil.Process(os.getpid()))+"-------------")
logging.info("-------------GrantPar PID: "+str(psutil.Process().parent())+"-------------")

from werkzeug.serving import is_running_from_reloader
if is_running_from_reloader():
    print(f"################### Restarting @ {datetime.utcnow()} ###################")


from flask import Flask, request, redirect, url_for, send_from_directory, render_template, after_this_request
from utils_log import logging
# import logging
from werkzeug.utils import secure_filename

# with open('zzdo_prediction_from_list_paths.pickle', 'wb') as handle:
#     pickle.dump( (img_np_boxes, img_np_raw, detections, df_d, path_img_box), handle, protocol=pickle.HIGHEST_PROTOCOL)
# with open('zzdo_prediction_from_list_paths.pickle', 'rb') as handle:
#     all_load = pickle.load(handle)
# img_np_boxes, img_np_raw, detections, df_d, path_img_box =    all_load

TEMPLATE_FOLDER = r"..\iacarry-evaluation"
DATE_NAME_FOLDER = datetime.now().strftime("%Y_%m_%d")
UPLOAD_FOLDER = os.path.abspath(TEMPLATE_FOLDER +r"\_uploads_img_for_test_from_web"  )
logging.info("UPLOAD_FOLDER: "+UPLOAD_FOLDER)
ALLOWED_EXTENSIONS = set(["png", "jpg", "jpge"])
logging.info("ALLOWED_EXTENSIONS: "+str(ALLOWED_EXTENSIONS))
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

PATH_TO_SAVED = "..\iacarry-evaluation\_upload_img_bbox_results" # "model_efi_d1C"



logging.info("Folder:\t" +__name__ + ":"+str(TEMPLATE_FOLDER))
app = Flask(__name__, template_folder=TEMPLATE_FOLDER)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
from server_Detector_model  import Detector_model
# @app.before_first_request
# def before_first_request():
logging.info(f"########### Restarted, First request @ {datetime.utcnow()} ############")
# #https://stackoverflow.com/questions/53162448/how-do-i-load-a-file-on-initialization-in-a-flask-application
app.detector1 = Detector_model()
from flask import current_app


#https://stackoverflow.com/questions/53162448/how-do-i-load-a-file-on-initialization-in-a-flask-application


#RETIRAR
PATH_PICKLE_CAT_INDEX = 'model_efi_d1C/P_Category_index.pickle'
with open(PATH_PICKLE_CAT_INDEX, 'rb') as handle:
    Category_index = pickle.load(handle)


HOST = "localhost"
PORT = 8000
MIN_SCORE_TO_CLIENT = 0.1

logging.info("HOST:\t" +HOST + ":"+str(PORT))


@app.route("/")
def home_index():
    logging.info("@app.route(/) Load index.html IP_remote: "+ str({'ip': request.remote_addr}) )
    return render_template("iaCarry_Local_JS_1_Clouding.html")

import json
from flask import jsonify, make_response
@app.route("/upload", methods=["GET", "POST"])
def upload_file():
    @after_this_request
    def add_header(response):
        response.headers['Access-Control-Allow-Origin'] = '*'
        return response
    logging.info("@app.route(\"/upload\":  IP_remote: "+ str({'ip': request.remote_addr}) )
    if request.method == "POST":
        if not "file" in request.files:
            return "No file part in the form."
        # print("file Detect ", request.files['file'].filename )
        f = request.files["file"]
        if f.filename == "":
            return "No file selected."
        logging.info("\tFiles Detect POST: "+ f.filename)
        if f and allowed_file(f.filename):
            filename, save_path = utils_tuisku_server.save_img_loaded(f, app.config["UPLOAD_FOLDER"], DATE_NAME_FOLDER)

            img_np_boxes, img_np_raw, detections, df_d, path_img_box = current_app.detector1.do_prediction_from_list_paths(save_path) #singletone current_app
            logging.info("save on file: "+os.path.join(app.config["UPLOAD_FOLDER"], filename))

            dict_to_respond = utils_tuisku_server.change_format_dict_json_to_client(detections, img_np_raw, path_img_box, MIN_SCORE_TO_CLIENT)
            jrespon = json.dumps(dict_to_respond, cls=utils_tuisku_server.NumpyEncoder)
            # print(jrespon)
            return jrespon #jsonify(jrespon)#make_response(jsonify(jrespon), 200)
            # return "file successfully upload Saved in: <br> "+ save_path  ,detections, df_d, path_img_box
        return "File not allowed. Only allowned: " + ", ".join(["png", "jpg", "jpge"])
    return "Upload file route"




if __name__ == "__main__":
    # app.run(debug=True, host=HOST, port=PORT)
    app.run(debug=True, host=HOST, port=PORT, use_reloader=False,  threaded=True, processes=1) #
    # from waitress import serve
    # serve(app, host=HOST, port=PORT)