import os
import logging
import pickle
import time
import uuid
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
# Directory holding this file (server/). It is the versioned home of the template
# served at GET /, so the file under version control is the file Flask renders.
SERVER_DIR = os.path.dirname(os.path.abspath(__file__))
logging.info("SERVER_DIR (primary template folder): " + SERVER_DIR)
DATE_NAME_FOLDER = datetime.now().strftime("%Y_%m_%d")
UPLOAD_FOLDER = os.path.abspath(TEMPLATE_FOLDER +r"\_uploads_img_for_test_from_web"  )
logging.info("UPLOAD_FOLDER: "+UPLOAD_FOLDER)
ALLOWED_EXTENSIONS = set(["png", "jpg", "jpge"])
logging.info("ALLOWED_EXTENSIONS: "+str(ALLOWED_EXTENSIONS))
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

PATH_TO_SAVED = "..\iacarry-evaluation\_upload_img_bbox_results" # "model_efi_d1C"



logging.info("Folder:\t" +__name__ + ":"+str(TEMPLATE_FOLDER))
app = Flask(__name__, template_folder=SERVER_DIR)
# Templates are looked up in server/ first, then in the legacy deploy folder
# (..\iacarry-evaluation). Editing the versioned file therefore takes effect on
# reload, while deployments that still copy the template across keep working.
from jinja2 import ChoiceLoader, FileSystemLoader
app.jinja_loader = ChoiceLoader([FileSystemLoader(SERVER_DIR), FileSystemLoader(TEMPLATE_FOLDER)])
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
    # Which file is actually served matters: the template is looked up in
    # server/ first and only then in the legacy deploy folder, and editing the
    # wrong copy is the classic "my change did nothing" hour.
    try:
        src = app.jinja_env.get_template("iaCarry_Local_JS_1_Clouding.html").filename
        logging.info("serving template from: %s", src)
    except Exception as e:                       # never let logging break the page
        logging.warning("could not resolve template path for logging: %s", e)
    return render_template("iaCarry_Local_JS_1_Clouding.html")

import json
from flask import jsonify, make_response
@app.route("/upload", methods=["GET", "POST"])
def upload_file():
    # One id per frame. Taken from the browser when it sends one (the checkout
    # page always does), otherwise minted here. It is written into every log line
    # below, passed down into the detector, and returned to the browser in both
    # the body and a header — so one purchase can be followed across all three
    # layers by grepping a single string.
    rid = request.headers.get("X-Request-Id") or uuid.uuid4().hex[:12]

    @after_this_request
    def add_header(response):
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['X-Request-Id'] = rid
        # Without this the browser CANNOT read X-Request-Id on a cross-origin
        # call — it is dropped silently and the correlation id looks absent.
        # The page posts same-origin, but this route is CORS-open by design.
        response.headers['Access-Control-Expose-Headers'] = 'X-Request-Id'
        return response

    t_req = time.perf_counter()
    logging.info("[%s] /upload %s from %s", rid, request.method, request.remote_addr)
    if request.method == "POST":
        # The three rejection paths below used to return a bare string and log
        # nothing at all, so a client failing every request looked identical to
        # no traffic at all.
        if not "file" in request.files:
            logging.warning("[%s] REJECTED: no 'file' part in the form (parts=%s)",
                            rid, list(request.files.keys()))
            return "No file part in the form."
        f = request.files["file"]
        if f.filename == "":
            logging.warning("[%s] REJECTED: empty filename", rid)
            return "No file selected."
        logging.info("[%s] frame received name=%s content_type=%s", rid, f.filename, f.mimetype)
        if f and allowed_file(f.filename):
            filename, save_path = utils_tuisku_server.save_img_loaded(
                f, app.config["UPLOAD_FOLDER"], DATE_NAME_FOLDER, rid=rid)
            try:
                size_b = os.path.getsize(save_path)
            except OSError:
                size_b = -1
            logging.info("[%s] saved %s (%s bytes)", rid, save_path, size_b)

            t_pred = time.perf_counter()
            img_np_boxes, img_np_raw, detections, df_d, path_img_box = \
                current_app.detector1.do_prediction_from_list_paths(save_path, rid=rid)  # singletone current_app
            t_pred = time.perf_counter() - t_pred

            dict_to_respond = utils_tuisku_server.change_format_dict_json_to_client(
                detections, img_np_raw, path_img_box, MIN_SCORE_TO_CLIENT, rid=rid)
            dict_to_respond["request_id"] = rid          # additive; older clients ignore it
            jrespon = json.dumps(dict_to_respond, cls=utils_tuisku_server.NumpyEncoder)
            logging.info("[%s] /upload DONE predictions=%d shape=%s bytes=%d "
                         "detector=%.3fs total=%.3fs",
                         rid, len(dict_to_respond["predictions"]),
                         dict_to_respond["shape_img"], len(jrespon),
                         t_pred, time.perf_counter() - t_req)
            return jrespon #jsonify(jrespon)#make_response(jsonify(jrespon), 200)
        # NOTE: ALLOWED_EXTENSIONS contains "jpge", almost certainly a typo for
        # "jpeg" — a genuine .jpeg frame lands here and is refused.
        logging.warning("[%s] REJECTED: extension not allowed name=%s allowed=%s",
                        rid, f.filename, sorted(ALLOWED_EXTENSIONS))
        return "File not allowed. Only allowned: " + ", ".join(["png", "jpg", "jpge"])
    logging.info("[%s] /upload reached with GET — nothing to do", rid)
    return "Upload file route"




if __name__ == "__main__":
    # app.run(debug=True, host=HOST, port=PORT)
    app.run(debug=True, host=HOST, port=PORT, use_reloader=False,  threaded=True, processes=1) #
    # from waitress import serve
    # serve(app, host=HOST, port=PORT)