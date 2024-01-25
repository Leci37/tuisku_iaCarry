

### CREATE tfrecord with default correct format 
better use default object detection API scripts to create a tfrecord https://github.com/tensorflow/models/tree/master/research/object_detection/dataset_tools 
En local esta en
_"C:\Users\Luis\.conda\envs\tf_object\Lib\site-packages\object_detection\dataset_tools\create_coco_tf_record.py"

DETALLES: https://www.tensorflow.org/tfmodels/vision/object_detection 
Example usage:
    python create_coco_tf_record.py --logtostderr \
      --train_image_dir="${TRAIN_IMAGE_DIR}" \
      --val_image_dir="${VAL_IMAGE_DIR}" \
      --test_image_dir="${TEST_IMAGE_DIR}" \
      --train_annotations_file="${TRAIN_ANNOTATIONS_FILE}" \
      --val_annotations_file="${VAL_ANNOTATIONS_FILE}" \
      --testdev_annotations_file="${TESTDEV_ANNOTATIONS_FILE}" \
      --output_dir="${OUTPUT_DIR}"


TRAIN_DATA_DIR='C:\Users\Luis\Desktop\tuisku_ML'
TRAIN_ANNOTATION_FILE_DIR='coco.json'
OUTPUT_TFRECORD_TRAIN='img_tfrecord'

### Need to provide
  ##### 1. image_dir: where images are present
  ##### 2. object_annotations_file: where annotations are listed in json format
  ##### 3. output_file_prefix: where to write output convered TFRecords files
python -m official.vision.data.create_coco_tf_record --logtostderr \
  --image_dir=${TRAIN_DATA_DIR} \
  --object_annotations_file=${TRAIN_ANNOTATION_FILE_DIR} \
  --output_file_prefix=$OUTPUT_TFRECORD_TRAIN \
  --num_shards=4

#### Para esto usar el siguiente scrip
COCO_to_TFRecord.bat 

#### Para ver las etiquetas puestas en tfrcord y que las claves estan bien puestas 
COCO_to_TFRecord_show.bat
#### Para cojer las imagenes del tfrcord y dividirlas por estiquetas, todos los perros a una otos los gatos a otra
COCO_to_TFRecord_show.bat

#### No funcionana, los ficheros , ya que hacen el tfrecord pero con las etiquetas correctas
COCO_to_TFRecord.py
COCO_to_TFRecord_utils.py 