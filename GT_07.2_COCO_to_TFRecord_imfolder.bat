@echo off

rem from https://github.com/sulc/tfrecord-viewer
rem Partir los datos en funcion de la etiqueta
SET "PATH_SCRIPT_VIEWER=C:\Users\Luis\Desktop\tuisku_ML\Utils\tfrecord_viewer"
SET "PY_SCRIPT_VIEWER=C:\Users\Luis\Desktop\tuisku_ML\Utils\tfrecord_viewer\tfrecord_to_imfolder.py"

SET "PATH_TFRECORD=img_tfrecord/coco_train.record*"
SET "PATH_OUTPUT=img_tfrecord/img_from_tfrecord_train"


echo %PATH_TFRECORD%
echo %PATH_SCRIPT_VIEWER%
echo %PY_SCRIPT_VIEWER%
echo %PATH_OUTPUT%
echo --------------
cd %PATH_SCRIPT_VIEWER%

python %PY_SCRIPT_VIEWER%   %PATH_TFRECORD% --output_path %PATH_OUTPUT%
echo check_path:  %PATH_OUTPUT%


SET "PATH_TFRECORD=img_tfrecord/coco_testdev.record*"
SET "PATH_OUTPUT=img_tfrecord/img_tfrecord_test"
rem pip install -U tensorflow-datasets==4.8.3

echo %PATH_TFRECORD%
echo %PATH_SCRIPT_VIEWER%
echo %PY_SCRIPT_VIEWER%
echo %PATH_OUTPUT%
echo --------------
cd %PATH_SCRIPT_VIEWER%

python %PY_SCRIPT_VIEWER%   %PATH_TFRECORD% --output_path %PATH_OUTPUT%
echo check_path:  %PATH_OUTPUT%


SET "PATH_TFRECORD=/img_tfrecord/coco_val.record*"
SET "PATH_OUTPUT=img_tfrecord/img_tfrecord_val"


echo %PATH_TFRECORD%
echo %PATH_SCRIPT_VIEWER%
echo %PY_SCRIPT_VIEWER%
echo %PATH_OUTPUT%
echo --------------
cd %PATH_SCRIPT_VIEWER%

python %PY_SCRIPT_VIEWER%   %PATH_TFRECORD% --output_path %PATH_OUTPUT%
echo check_path:  %PATH_OUTPUT%


