# https://rockyshikoku.medium.com/how-to-use-tensorflow-object-detection-api-with-the-colab-sample-notebooks-477707fadf1b
import random
import time

import cv2
import pandas as pd
import matplotlib.pyplot as plt
import os
import scipy.misc
import numpy as np
from keras.callbacks import EarlyStopping
from six import BytesIO
from PIL import Image, ImageDraw, ImageFont
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder

import GT_Utils
from Transfer_L_Mediun_utils import load_image_into_numpy_array, plot_detections, get_model_detection_function
from utils_transfer_learning import save_detecion_pd_checkpoint

"""5. Download the model
You can get the model you want from the Model Zoo.
https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md """
# !wget http://download.tensorflow.org/models/object_detection/tf2/20200713/centernet_hg104_512x512_coco17_tpu-8.tar.gz
# !tar -xf centernet_hg104_512x512_coco17_tpu-8.tar.gz
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
# os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
# def compute_resource(op):
#     if op.lower() == 'yes' or op.lower() == 'y':
#         # Allow memory growth for the GPU
#         physical_devices = tf.config.experimental.list_physical_devices('GPU')
#         tf.config.experimental.set_memory_growth(physical_devices[0], True)
#         # Assume that you have 12GB of GPU memory and want to allocate ~4GB:
#         gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.12)
#         sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))
#     else:
#         print("Going to use CPU ")
#         physical_devices = tf.config.experimental.list_physical_devices('CPU')
# compute_resource("y")
# gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.25, allow_growth=True)
# sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))


"""8. Prepare the labels
For the inferencing of the object detection, you need the labels of the objects had been used in the training.
You can find the labels in “models/research/object_detection/data/” in the repository. We use mscoco_label_map.pbtxt because our model have been trained by COCO Datasets."""
LABEL_MAP_PATH = './ssd_mobilenet_v2_fpnlite_640x640/label_map_eroski.pbtxt'
label_map = label_map_util.load_labelmap(LABEL_MAP_PATH)
categories = label_map_util.convert_label_map_to_categories(
     label_map, max_num_classes=label_map_util.get_max_label_map_index(label_map),use_display_name=True)
category_index = label_map_util.create_category_index(categories)
label_map_dict = label_map_util.get_label_map_dict(label_map, use_display_name=True)
NUM_CLASSES = len(category_index)
num_classes = NUM_CLASSES
NUM_ELE_PER_CLASS = 50
def get_id_from_name(type_prod_name):
    return [v['id'] for k, v in category_index.items() if v['name'] == type_prod_name][0]

PATH_BBOX = r"E:\iaCarry_img_eroski\df_size_Rota_img.csv"
df = pd.read_csv(PATH_BBOX, sep='\t', index_col=0)
# df = df.sort_values(['Date'], ascending=True)
# [v['id'] for k, v in category_index.items() if v['name'] == 'colgate_75ml' ]
print("Load data Shape: ", df.shape, " Path: ", PATH_BBOX)
types_prod = df.groupby(["type_prod"])['path'].count().index.values
print("Products count : ", len(types_prod) )

Train_image_filenames =[]
Train_labels = []
Gt_labels = []
Gt_boxes = []
for pro in  types_prod:
    df_p = df[df["type_prod"]== pro]
    df_p = df_p[-NUM_ELE_PER_CLASS:] #only the lastt 50
    Train_image_filenames = [*Train_image_filenames, *df_p['path'].values]
    Train_labels = [*Train_labels, *df_p['type_prod'].values]
    aux_Gt_labels = [np.array([get_id_from_name(name)]) for name in df_p['type_prod'].values]
    Gt_labels = [*Gt_labels, *aux_Gt_labels]
    # Numpy array of [ miny, minx, maxy, maxx ]
    for index, row in df_p.iterrows():
        x, y, w, h = GT_Utils.get_scalated_00_x_y_w_h(row)
        x, y, w, h = x, y, (x + w), (y + h)
        Gt_boxes.append(np.array([[y, x, h, w]], dtype=np.float32))
    print("" + GT_Utils.bcolors.OKBLUE + pro + GT_Utils.bcolors.ENDC + "\t Count: ", df_p.shape,"\t Id_label: ", str(aux_Gt_labels[0]))

# The path to the pipeline config.
pipeline_config = "./ssd_mobilenet_v2_fpnlite_640x640/pipeline.config"
# The to the checkpoint.
model_dir = "./ssd_mobilenet_v2_fpnlite_640x640/checkpoint"
# Reading the model configurations.
configs = config_util.get_configs_from_pipeline_file(pipeline_config)
model_config = configs['model']
# Build the model with the configurations read.
detection_model = model_builder.build(model_config=model_config, is_training=True)
# Restore the weights from the checkpoint.
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(model_dir, 'ckpt-0')).expect_partial()

"""7. prepare the inferencing function"""
detect_fn = get_model_detection_function(detection_model)


"""6. Prepare the images and the label map, the annotations data Things required.
1. Array of paths to the images.
2. Label map ( Dictionary which label point to which ID)
3. Array of IDs
4. Array of bounding boxes

Indexes in arrays of images, labels, boxes have to be same with each other.
"""
# import os
# import glob
# from PIL import Image
# src = glob.glob('./patos/train/*.jpg') # Set paths to original images.
# dst = './patos/train_r/' # Path to the destination directory for saving.
# width = 640 # width you want
# height = 640 # height
# for f in src:
#      img = Image.open(f)
#      img = img.resize((width,height))
#      print("\t", dst + os.path.basename(f))
#      img.save(dst + os.path.basename(f))

"""7. Put images in numpy array"""
train_images_np = []
for i in range(0, len(Train_image_filenames)):
    # img = Image.open(filename)
    filename = Train_image_filenames[i]
    print("\t",i, "\t",Gt_labels[i], "\t", filename , "\t\t Bbox: ", Gt_boxes[i].shape)
    # Load .png image
    image = cv2.imread(filename)
    cv2.imwrite(filename.replace(".png", ".jpg"), image, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
    train_images_np.append(load_image_into_numpy_array(filename.replace(".png", ".jpg")))

# Display
# plt.imshow(train_images_np[0])
# plt.show()

"""8. Put class labels in one hot tensor, put images in tensor, put boxes in tensor
“One hot” is the array of 0 and 1. It represent the number by pointing to the index with “1”."""
# Convert class labels to one-hot; convert everything to tensors.
# The `label_id_offset` here shifts all classes by a certain number of indices;
# we do this here so that the model receives one-hot labels where non-background
# classes start counting at the zeroth index.  This is ordinarily just handled
# automatically in our training binaries, but we need to reproduce it here.
LABEL_ID_OFFSET = 1
train_image_tensors = []
gt_classes_one_hot_tensors = []
gt_box_tensors = []
# One memory-costly process is the internal conversion from (usually) numpy to tf.tensor. If you are sure that your GPU should be able to handle the batches:
# Manually convert the data to tf tensors using the CPU RAM, and only then pass it to your model (eventually using GPU).
print("\nUse CPU, GPU dont have space to locate one_hot_tensors. Use for hot_encoders:", str(tf.config.experimental.list_physical_devices('CPU')[0]))
with tf.device('/cpu:0'): # Use CPU, GPU dont have space
    for (train_image_np, gt_box_np, gt_label_np) in zip(train_images_np, Gt_boxes, Gt_labels):
        print("|", end="")
        train_image_tensors.append(tf.expand_dims(tf.convert_to_tensor(train_image_np, dtype=tf.float32), axis=0)) # put images in tensor
        gt_box_tensors.append(tf.convert_to_tensor(gt_box_np, dtype=tf.float32)) # put box in Tensor
        zero_indexed_groundtruth_classes = tf.convert_to_tensor(gt_label_np - LABEL_ID_OFFSET) # put labels in Numpy array (min:0)
        gt_classes_one_hot_tensors.append(tf.one_hot(zero_indexed_groundtruth_classes, num_classes)) # label Tensor to one hot
print('Done prepping data. Num Data: ',len(train_images_np), "\n" )


"""9. Visualize ground truth boxes"""
DUMMY_100_SCORE = np.array([1.0], dtype=np.float32) # Temporarily put 100% scores
for i_step in range(0, len(train_images_np), NUM_ELE_PER_CLASS):
    ram_list = list(np.random.randint(i_step, i_step+NUM_ELE_PER_CLASS, 20))
    print(i_step, Train_labels[i_step], ram_list )
    plt.figure(figsize=(30, 15))
    count_index = 1
    for idx in ram_list:
       plt.subplot(4, 5, count_index)
       count_index = count_index+1
       plot_detections(
           image_np = train_images_np[idx],
           boxes = Gt_boxes[idx],
           classes = Gt_labels[idx],
           scores = DUMMY_100_SCORE, category_index = category_index)
    # plt.show()
    path_save_mosaico = "mosaico_GT_" + Train_labels[i_step]+".png"
    print("\t", path_save_mosaico)
    plt.savefig(path_save_mosaico, bbox_inches='tight')


"""10. Build the model and restore the weights
Restore weights except the last layer. Only the last layer is initialized with random weights for training.

In this article, we use ResNet back bone RetinaNet.Rewrite the class number in the config file to the class number of your own dataset.

Head specifies the layer to restore from the checkpoint. 
This time, we don’t restore the weight of the part for class classification, so only specify the weight of the part for box regression."""
tf.keras.backend.clear_session()

print('\nBuilding model and restoring weights for fine-tuning...', flush=True)

PATH_TRANSFER =  'ssd_mobilenet_v2_fpnlite_640x640'
PATH_MODEL_TRANSFER_PIPE = PATH_TRANSFER +'/pipeline.config'
PATH_MODEL_TRANSFER_CHECK = PATH_TRANSFER +'/checkpoint/ckpt-0'
print('\t'+PATH_MODEL_TRANSFER_PIPE)
PATH_MODELS_CHECKPOINT_STEP_STPE = 'ssd_eroski'
# pipeline_config = 'ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8/pipeline.config'
# checkpoint_path = 'ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8/checkpoint/ckpt-0'

# Load pipeline config and build a detection model.
#
# Since we are working off of a COCO architecture which predicts 90
# class slots by default, we override the `num_classes` field here to be just
# one (for our new rubber ducky class).
configs = config_util.get_configs_from_pipeline_file(PATH_MODEL_TRANSFER_PIPE)
configs['train_config'].fine_tune_checkpoint_type = "detection"
configs['train_config'].fine_tune_checkpoint = PATH_MODEL_TRANSFER_CHECK
# configs['DESCRIPTOR'] = "Luis config updated"
# configs['eval_input_configs'].DESCRIPTOR = configs['train_config'].DESCRIPTOR
# config_util.save_pipeline_config(configs, os.path.dirname(pipeline_config))
# print("UPDATE: " , pipeline_config)
# configs = config_util.get_configs_from_pipeline_file(pipeline_config)
configs['model'].ssd.num_classes = num_classes
configs['model'].ssd.freeze_batchnorm = True
detection_model = model_builder.build(model_config=configs['model'], is_training=True)
pipeline_proto = config_util.create_pipeline_proto_from_configs(configs)
config_util.save_pipeline_config(pipeline_proto,PATH_MODELS_CHECKPOINT_STEP_STPE )
print("Created: "+ PATH_MODELS_CHECKPOINT_STEP_STPE+"/pipeline_config")

# fine_tune_checkpoint_type: "detection" TODO ??
# use_bfloat16: false

# Set up object-based checkpoint restore --- RetinaNet has two prediction
# `heads` --- one for classification, the other for box regression.  We will
# restore the box regression head but initialize the classification head
# from scratch (we show the omission below by commenting out the line that
# we would add if we wanted to restore both heads)
fake_box_predictor = tf.compat.v2.train.Checkpoint(
    _base_tower_layers_for_heads=detection_model._box_predictor._base_tower_layers_for_heads,
    # _prediction_heads=detection_model._box_predictor._prediction_heads,
    #    (i.e., the classification head that we *will not* restore)
    _box_prediction_head=detection_model._box_predictor._box_prediction_head,
    )
fake_model = tf.compat.v2.train.Checkpoint(
          _feature_extractor=detection_model._feature_extractor,
          _box_predictor=fake_box_predictor)
ckpt = tf.compat.v2.train.Checkpoint(model=fake_model)
ckpt.restore(PATH_MODEL_TRANSFER_CHECK).expect_partial()

# Run model through a dummy image so that variables are created
image, shapes = detection_model.preprocess(tf.zeros([1, 640, 640, 3]))
prediction_dict = detection_model.predict(image, shapes)
_ = detection_model.postprocess(prediction_dict, shapes)
print('Weights restored!')

# Set up forward + backward pass for a single train step.
def get_model_train_step_function(model, optimizer, vars_to_fine_tune):
  """Get a tf.function for training step."""

  # Use tf.function for a bit of speed.
  # Comment out the tf.function decorator if you want the inside of the
  # function to run eagerly.
  @tf.function
  def train_step_fn(image_tensors,
                    groundtruth_boxes_list,
                    groundtruth_classes_list):
    """A single training iteration.

    Args:
      image_tensors: A list of [1, height, width, 3] Tensor of type tf.float32.
        Note that the height and width can vary across images, as they are
        reshaped within this function to be 640x640.
      groundtruth_boxes_list: A list of Tensors of shape [N_i, 4] with type
        tf.float32 representing groundtruth boxes for each image in the batch.
      groundtruth_classes_list: A list of Tensors of shape [N_i, num_classes]
        with type tf.float32 representing groundtruth boxes for each image in
        the batch.

    Returns:
      A scalar tensor representing the total loss for the input batch.
    """
    shapes = tf.constant(batch_size * [[640, 640, 3]], dtype=tf.int32)
    # print("groundtruth_boxes_list.shape: ",len(groundtruth_boxes_list) , " groundtruth_classes_list: ",  len(groundtruth_classes_list) )
    # print("groundtruth_boxes_list[0].shape: ",groundtruth_boxes_list[0].shape , " groundtruth_classes_list[0]: ",  groundtruth_classes_list[0].shape )
    model.provide_groundtruth(
        groundtruth_boxes_list=groundtruth_boxes_list,
        groundtruth_classes_list=groundtruth_classes_list)
    with tf.GradientTape() as tape:
      preprocessed_images = tf.concat( [detection_model.preprocess(image_tensor)[0] for image_tensor in image_tensors], axis=0)
      # print("preprocessed_images.shape: ",preprocessed_images.shape , " shapes: ", shapes)
      prediction_dict = model.predict(preprocessed_images, shapes)
      losses_dict = model.loss(prediction_dict, shapes)
      total_loss = losses_dict['Loss/localization_loss'] + losses_dict['Loss/classification_loss']
      gradients = tape.gradient(total_loss, vars_to_fine_tune)
      #vars_to_fine_tune por donde han ido el intento de poner las bbox
      optimizer.apply_gradients(zip(gradients, vars_to_fine_tune))
    return total_loss , losses_dict

  return train_step_fn

tf.keras.backend.set_learning_phase(True)

# These parameters can be tuned; since our training set has 5 images
# it doesn't make sense to have a much larger batch size, though we could
# fit more examples in memory if we wanted to.
batch_size = 8
learning_rate = 0.01
num_batches = 90000

# Select variables in top layers to fine-tune.
trainable_variables = detection_model.trainable_variables
to_fine_tune = []
prefixes_to_train = [
  'WeightSharedConvolutionalBoxPredictor/WeightSharedConvolutionalBoxHead',
  'WeightSharedConvolutionalBoxPredictor/WeightSharedConvolutionalClassHead']
for var in trainable_variables:
  if any([var.name.startswith(prefix) for prefix in prefixes_to_train]):
    to_fine_tune.append(var)

optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
train_step_fn = get_model_train_step_function(detection_model, optimizer, to_fine_tune)

#Checkpoint manager https://www.tensorflow.org/guide/checkpoint?hl=es-419
exported_ckpt_new = tf.compat.v2.train.Checkpoint(model=detection_model) #tf.train.Checkpoint(model=detection_model)
manager_ckpt_new = tf.train.CheckpointManager(exported_ckpt_new, directory=PATH_MODELS_CHECKPOINT_STEP_STPE + "/checkpoint/", max_to_keep=2 ) #manager_ckpt = tf.train.CheckpointManager(ckpt, './tf_ckpts', max_to_keep=3)
ckpt.restore(manager_ckpt_new.latest_checkpoint)
if manager_ckpt_new.latest_checkpoint:
  print("WARN (es nuevo ??) Restaurado de {}".format(manager_ckpt_new.latest_checkpoint))
else:
  print("Inicializando desde cero Path: ",PATH_MODELS_CHECKPOINT_STEP_STPE + "/checkpoint/")

print('Start fine-tuning!', flush=True)

time_start = time.time()
list_loss = []
for idx in range(num_batches):
  # Grab keys for a random subset of examples
  all_keys = list(range(len(train_images_np)))
  random.shuffle(all_keys)
  example_keys = all_keys[:batch_size]

  # Note that we do not do data augmentation in this demo.  If you want a
  # a fun exercise, we recommend experimenting with random horizontal flipping
  # and random cropping :)
  gt_boxes_list = [gt_box_tensors[key] for key in example_keys]
  gt_classes_list = [gt_classes_one_hot_tensors[key] for key in example_keys]
  image_tensors = [train_image_tensors[key] for key in example_keys]

  # Training step (forward pass + backwards pass)
  total_loss , losses_dict = train_step_fn(image_tensors, gt_boxes_list, gt_classes_list)
  list_loss.append(total_loss.numpy())
  print("=", end="")
  if len(list_loss) > 10 and all([not (l >= list_loss[-1]) for l in list_loss[-7:-1]]) and all( [not (l >= list_loss[-2]) for l in list_loss[-7:-1]]):
      print("EarlyStopping(monitor='loss')  END the lost does not impruve. Last loss: ", ",  ".join(['{:.5f}'.format(x) for x in list_loss[-16:-1]]))
      break
  if idx % 10 == 0:
    time_end = time.time(); loss_str = "\tbbox=" +str('{:.5f}'.format(losses_dict['Loss/localization_loss'].numpy()))+ "\tclass="+  str('{:.5f}'.format(losses_dict['Loss/classification_loss'].numpy()))
    print(' batch ' + str(idx) + ' of ' + str(num_batches) + '\tloss=' +  str('{:.5f}'.format(total_loss.numpy())), loss_str, "\tTime take: ", '{:.2f}'.format(time_end - time_start)+"s")
    time_start = time.time()

    if idx % 1000 == 0:
        print("DEBUG:  example_keys : ", example_keys)
        save_path = manager_ckpt_new.save()
        print("Checkpoint almacenado para el STEP: {}: {}".format(int(exported_ckpt_new.save_counter.numpy()), save_path),
            "  List_checkpoints:", manager_ckpt_new.checkpoints)


    # print("DEBUG:  example_keys : ", example_keys)
save_path = manager_ckpt_new.save()
print("Checkpoint almacenado para el STEP: {}: {}".format(int(exported_ckpt_new.save_counter.numpy()), save_path),"  List_checkpoints:" , manager_ckpt_new.checkpoints)

print('Done fine-tuning!')
#SAVE like checkpoint the traied model  https://github.com/tensorflow/models/issues/8862
save_detecion_pd_checkpoint(detection_model, PATH_MODELS_CHECKPOINT_STEP_STPE, configs)