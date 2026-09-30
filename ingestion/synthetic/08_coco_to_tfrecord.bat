@echo off

rem from https://www.tensorflow.org/tfmodels/vision/object_detection

@REM SET "TRAIN_DATA_DIR=C:\Users\Luis\Desktop\tuisku_ML"
@REM SET "TRAIN_ANNOTATION_FILE_DIR=coco.json"
@REM SET "OUTPUT_TFRECORD_TRAIN=img_tfrecord/train_bat"
@REM SET "PATH_SCRIPT=Utils\tf_models_garden\official\vision\data\create_coco_tf_record.py"
@REM rem pip install -U tensorflow-datasets==4.8.3
@REM
@REM echo %TRAIN_DATA_DIR%
@REM echo %TRAIN_ANNOTATION_FILE_DIR%
@REM echo %OUTPUT_TFRECORD_TRAIN%
@REM echo %PATH_SCRIPT%
@REM echo --------------
@REM python %PATH_SCRIPT% --image_dir=%TRAIN_DATA_DIR% --object_annotations_file=%TRAIN_ANNOTATION_FILE_DIR%  --output_file_prefix=%OUTPUT_TFRECORD_TRAIN% --num_shards=1

@REM --logtostderr --image_dir=%TRAIN_DATA_DIR% --object_annotations_file=%TRAIN_ANNOTATION_FILE_DIR% --output_file_prefix=%OUTPUT_TFRECORD_TRAIN% --num_shards=4"


@REM DETALLES: https://www.tensorflow.org/tfmodels/vision/object_detection
@REM Example usage:
@REM     python create_coco_tf_record.py --logtostderr \
@REM       --train_image_dir="${TRAIN_IMAGE_DIR}" \
@REM       --val_image_dir="${VAL_IMAGE_DIR}" \
@REM       --test_image_dir="${TEST_IMAGE_DIR}" \
@REM       --train_annotations_file="${TRAIN_ANNOTATIONS_FILE}" \
@REM       --val_annotations_file="${VAL_ANNOTATIONS_FILE}" \
@REM       --testdev_annotations_file="${TESTDEV_ANNOTATIONS_FILE}" \
@REM       --output_dir="${OUTPUT_DIR}"
@REM set  PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

SET "PATH_SCRIPT=C:\Users\Luis\.conda\envs\tf_object\Lib\site-packages\object_detection\dataset_tools\create_coco_tf_record.py"
SET "PATH_IMG_DIR=E:\iaCarry_img_eroski\Aug_A_down"
SET "PATH_TRAIN_COCO=cocod_train.json"
SET "PATH_TEST_COCO=cocod_test.json"
SET "PATH_VAL_COCO=cocod_val.json"
SET "PATH_OUTPUT=img_tfrecord"

@REM pushd  %PATH_IMG_DIR%  @REM /D necesario para cambar de unidad
cd %PATH_IMG_DIR%
echo %PATH_SCRIPT%
echo %PATH_IMG_DIR%
echo %PATH_TRAIN_COCO%
echo %PATH_TEST_COCO%
echo %PATH_VAL_COCO%
echo %PATH_OUTPUT%
echo --------------
python %PATH_SCRIPT%  --num_shards 32 ^
        --train_image_dir=%PATH_IMG_DIR% ^
        --test_image_dir=%PATH_IMG_DIR% ^
        --val_image_dir=%PATH_IMG_DIR%  ^
        --train_annotations_file=%PATH_TRAIN_COCO% ^
        --testdev_annotations_file=%PATH_TEST_COCO% ^
        --val_annotations_file=%PATH_VAL_COCO% ^
        --logtostderr --output_dir=%PATH_OUTPUT%

