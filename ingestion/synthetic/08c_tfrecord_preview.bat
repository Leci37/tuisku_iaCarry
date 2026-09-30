@echo off

rem from https://github.com/sulc/tfrecord-viewer

@REM SET "PATH_TFRECORD=img_tfrecord/coco_train.record*"
SET "PATH_TFRECORD=img_tfrecord/coco_testdev.record*"
SET "PATH_SCRIPT_VIEWER=C:\Users\Luis\Desktop\tuisku_ML\Utils\tfrecord_viewer"
SET "PY_SCRIPT_VIEWER=C:\Users\Luis\Desktop\tuisku_ML\Utils\tfrecord_viewer\tfviewer.py"
rem pip install -U tensorflow-datasets==4.8.3

echo %PATH_TFRECORD%
echo %PATH_SCRIPT_VIEWER%
echo %PY_SCRIPT_VIEWER%
echo --------------
cd %PATH_SCRIPT_VIEWER%
start http://127.0.0.1:8080

python %PY_SCRIPT_VIEWER%   %PATH_TFRECORD%

