# https://rockyshikoku.medium.com/how-to-use-tensorflow-object-detection-api-with-the-colab-sample-notebooks-477707fadf1b
import random
import time
import matplotlib
import matplotlib.pyplot as plt
import os
import scipy.misc
import numpy as np
from six import BytesIO
from PIL import Image, ImageDraw, ImageFont
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder

from Transfer_L_Mediun_utils import load_image_into_numpy_array, plot_detections, get_model_detection_function

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

"""6. Read the pipeline config (the configurations of the model), and build the model."""
category_index = {
    1: {'id': 1, 'name': 'zombie'},
    2: {'id': 2, 'name': 'cat'},
    3: {'id': 3, 'name': 'dog'}
}
num_classes = 3
gt_labels = [     # gt_classes_DELETE = [1,1,1,1,1,  2,2,2,2,2,2,2,2,  3,3,3,3,3,3,3,3]
    np.array([1 ]),
    np.array([1 ]),
    np.array([1 ]),
    np.array([1 ]),
    np.array([1 ]),
    np.array([2 ]),
    np.array([2 ]),
    np.array([2 ]),
    np.array([2 ]),
    np.array([2 ]),
    np.array([2 ]),
    np.array([2 ]),
    np.array([2 ]),
    np.array([3 ]),
    np.array([3 ]),
    np.array([3 ]),
    np.array([3 ]),
    np.array([3 ]),
    np.array([3 ]),
    np.array([3 ]),
    # np.array([1,2,2])
    np.array([3 ]),
]
gt_boxes = [
    np.array([[0.27333333, 0.41500586, 0.74333333, 0.57678781]], dtype=np.float32), #C:\Users\Luis\Desktop\Object_detec\training\training-zombie1.jpg
    np.array([[0.29833333, 0.45955451, 0.75666667, 0.61078546]], dtype=np.float32),
    np.array([[0.40833333, 0.18288394, 0.945, 0.34818288]], dtype=np.float32),
    np.array([[0.16166667, 0.61899179, 0.8, 0.91910903]], dtype=np.float32),
    np.array([[0.28833333, 0.12543962, 0.835, 0.35052755]], dtype=np.float32), #C:\Users\Luis\Desktop\Object_detec\training\training-zombie5.jpg

    np.array([[0.04064394, 0.226     , 0.65897727, 0.716     ]], dtype=np.float32),#cat.0.jpg
    np.array([[0.10897727, 0.15333333, 0.99064394, 0.90666667]], dtype=np.float32),
    np.array([[0.04231061, 0.14423077, 0.97064394, 0.86858974]], dtype=np.float32),
    np.array([[0.00397727, 0.008     , 0.91564394, 0.942     ]], dtype=np.float32),
    np.array([[0.08564394, 0.17835671, 0.95397727, 0.9759519 ]], dtype=np.float32),
    np.array([[0.37397727, 0.14285714, 0.81231061, 0.70857143]], dtype=np.float32),
    np.array([[0.39231061, 0.1425    , 0.76231061, 0.465     ]], dtype=np.float32),
    np.array([[0.03897727, 0.31919192, 0.98731061, 0.96363636]], dtype=np.float32),#cat.7.jpg

    np.array([[0.06397727, 0.15831663, 0.82397727, 0.75350701]], dtype=np.float32),#dog.0.jpg
    np.array([[0.10564394, 0.18042813, 0.94397727, 0.88073394]], dtype=np.float32),
    np.array([[0.23064394, 0.36898396, 0.92064394, 0.85561497]], dtype=np.float32),
    np.array([[0.        , 0.06813627, 0.97397727, 0.99398798]], dtype=np.float32),
    np.array([[0.01897727, 0.14      , 0.98564394, 0.84666667]], dtype=np.float32),
    np.array([[0.01064394, 0.250501  , 0.85897727, 0.59719439]], dtype=np.float32),
    np.array([[0.15064394, 0.09218437, 0.96564394, 0.91583166]], dtype=np.float32),
    #np.array([[0.436, 0.591, 0.629, 0.712],[0.539, 0.583, 0.73, 0.71]], dtype=np.float32),
    np.array([[0.13231061, 0., 0.96731061, 0.46488294]], dtype=np.float32)        #dog.7.jpg
           # , [0.24897727, 0.84280936, 0.64731061, 1.]])       		#cat.7.jpg
    ]

# The path to the pipeline config.
pipeline_config = "./ssd_mobilenet_v2_fpnlite_640x640/pipeline.config"
# The to the checkpoint.
model_dir = "./ssd_mobilenet_v2_fpnlite_640x640/checkpoint"
# Reading the model configurations.
configs = config_util.get_configs_from_pipeline_file(pipeline_config)
model_config = configs['model']
# Build the model with the configurations read.
detection_model = model_builder.build(model_config=model_config, is_training=False)
# Restore the weights from the checkpoint.
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(model_dir, 'ckpt-0')).expect_partial()

"""7. prepare the inferencing function"""
detect_fn = get_model_detection_function(detection_model)


"""8. Prepare the labels
For the inferencing of the object detection, you need the labels of the objects had been used in the training.
You can find the labels in “models/research/object_detection/data/” in the repository. We use mscoco_label_map.pbtxt because our model have been trained by COCO Datasets."""
label_map_path = './ssd_mobilenet_v2_fpnlite_640x640/mscoco_label_map_cat_dog.pbtxt'
label_map = label_map_util.load_labelmap(label_map_path)
categories = label_map_util.convert_label_map_to_categories(
     label_map, max_num_classes=label_map_util.get_max_label_map_index(label_map),use_display_name=True)
category_index = label_map_util.create_category_index(categories)
label_map_dict = label_map_util.get_label_map_dict(label_map, use_display_name=True)

import utils_transfer_learning
category_index = utils_transfer_learning.get_category_index()
NUM_CLASSES = len(category_index)
num_classes = NUM_CLASSES


"""6. Prepare the images and the label map, the annotations data Things required.
1. Array of paths to the images.
2. Label map ( Dictionary which label point to which ID)
3. Array of IDs
4. Array of bounding boxes

Indexes in arrays of images, labels, boxes have to be same with each other.
"""
import os
import glob
from PIL import Image
src = glob.glob('./patos/train/*.jpg') # Set paths to original images.
dst = './patos/train_r/' # Path to the destination directory for saving.
width = 640 # width you want
height = 640 # height
for f in src:
     img = Image.open(f)
     img = img.resize((width,height))
     print("\t", dst + os.path.basename(f))
     img.save(dst + os.path.basename(f))

"""7. Put images in numpy array"""
train_image_dir = './patos/train_r' # Path to the directory of images
train_images_np = []
train_image_filenames = []
list_names_order_img = ['./patos/train_r/training-zombie*.jpg','./patos/train_r/cat*.jpg','./patos/train_r/dog*.jpg'  ]
for name_order_img in list_names_order_img:
    for filename in glob.glob(name_order_img):
        train_image_filenames.append(filename)
        train_images_np.append(load_image_into_numpy_array(filename))
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
label_id_offset = 1
train_image_tensors = []
gt_classes_one_hot_tensors = []
gt_box_tensors = []
for (train_image_np, gt_box_np, gt_label_np) in zip(train_images_np, gt_boxes, gt_labels):
    train_image_tensors.append(tf.expand_dims(tf.convert_to_tensor(train_image_np, dtype=tf.float32), axis=0)) # put images in tensor
    gt_box_tensors.append(tf.convert_to_tensor(gt_box_np, dtype=tf.float32)) # put box in Tensor
    zero_indexed_groundtruth_classes = tf.convert_to_tensor(gt_label_np - label_id_offset) # put labels in Numpy array (min:0)
    gt_classes_one_hot_tensors.append(tf.one_hot(zero_indexed_groundtruth_classes, num_classes)) # label Tensor to one hot
print('Done prepping data. Num Data: ',len(train_images_np) )


"""9. Visualize ground truth boxes"""
dummy_scores = np.array([1.0], dtype=np.float32) # Temporarily put 100% scores
plt.figure(figsize=(30, 15))
for idx in range(20):
   plt.subplot(4, 6, idx+1)
   plot_detections(
       image_np = train_images_np[idx],
       boxes = gt_boxes[idx],
       classes = gt_labels[idx],
       scores = dummy_scores, category_index = category_index)
# plt.show()
print("\t  mosaico_ALL_labels.png")
plt.savefig(" mosaico_ALL_labels.png", bbox_inches='tight')


"""10. Build the model and restore the weights
Restore weights except the last layer. Only the last layer is initialized with random weights for training.

In this article, we use ResNet back bone RetinaNet.Rewrite the class number in the config file to the class number of your own dataset.

Head specifies the layer to restore from the checkpoint. 
This time, we don’t restore the weight of the part for class classification, so only specify the weight of the part for box regression."""
# tf.keras.backend.clear_session()
#
# print('Building model and restoring weights for fine-tuning...', flush=True)
#
# PATH_TRANSFER =  'ssd_mobilenet_v2_fpnlite_640x640'
# PATH_MODEL_TRANSFER_PIPE = PATH_TRANSFER +'/pipeline.config'
# PATH_MODEL_TRANSFER_CHECK = PATH_TRANSFER +'/checkpoint/ckpt-0'
# print('\t'+PATH_MODEL_TRANSFER_PIPE)
# PATH_MODELS_CHECKPOINT_STEP_STPE = 'ssd_cat_dog_zombie_Trans'
# # pipeline_config = 'ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8/pipeline.config'
# # checkpoint_path = 'ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8/checkpoint/ckpt Load pipeline config and build a detection model.
# #
# # Since we are working off of a COCO architecture which predicts 90
# # class slots by default, we override the `num_classes` field here to be just
# # one (for our new rubber ducky class).
# configs = config_util.get_configs_from_pipeline_file(PATH_MODEL_TRANSFER_PIPE)
# configs['train_config'].fine_tune_checkpoint_type = "detection"
# configs['train_config'].fine_tune_checkpoint = PATH_MODEL_TRANSFER_CHECK
# # configs['DESCRIPTOR'] = "Luis config updated"
# # configs['eval_input_configs'].DESCRIPTOR = configs['train_config'].DESCRIPTOR
# # config_util.save_pipeline_config(configs, os.path.dirname(pipeline_config))
# # print("UPDATE: " , pipeline_config)
# # configs = config_util.get_configs_from_pipeline_file(pipeline_config)
# configs['model'].ssd.num_classes = num_classes
# configs['model'].ssd.freeze_batchnorm = True
# detection_model = model_builder.build(model_config=configs['model'], is_training=True)
# pipeline_proto = config_util.create_pipeline_proto_from_configs(configs)
# config_util.save_pipeline_config(pipeline_proto,PATH_MODELS_CHECKPOINT_STEP_STPE )
# print("Created: "+ PATH_MODELS_CHECKPOINT_STEP_STPE+"/pipeline_config")

tf.keras.backend.clear_session()

print('Building model and restoring weights for fine-tuning...', flush=True)

PATH_TRANSFER =  'ssd_mobilenet_v2_fpnlite_640x640'
PATH_MODEL_TRANSFER_PIPE = PATH_TRANSFER +'/pipeline.config'
PATH_MODEL_TRANSFER_CHECK = PATH_TRANSFER +'/checkpoint/ckpt-0'
print('\t'+PATH_MODEL_TRANSFER_PIPE)
PATH_MODELS_CHECKPOINT_STEP_STPE = 'ssd_cat_dog_zombie_A1'
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
num_batches = 700

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
  print("=", end="")
  if idx % 10 == 0:
    time_end = time.time(); loss_str = "\tbbox=" +str('{:.5f}'.format(losses_dict['Loss/localization_loss'].numpy()))+ "\tclass="+  str('{:.5f}'.format(losses_dict['Loss/classification_loss'].numpy()))
    print(' batch ' + str(idx) + ' of ' + str(num_batches) + '\tloss=' +  str('{:.5f}'.format(total_loss.numpy())), loss_str, "\tTime take: ", '{:.2f}'.format(time_end - time_start)+"s")
    time_start = time.time()

    # print("DEBUG:  example_keys : ", example_keys)
save_path = manager_ckpt_new.save()
print("Checkpoint almacenado para el STEP: {}: {}".format(int(exported_ckpt_new.save_counter.numpy()), save_path),"  List_checkpoints:" , manager_ckpt_new.checkpoints)

print('Done fine-tuning!')
#SAVE like checkpoint the traied model  https://github.com/tensorflow/models/issues/8862
utils_transfer_learning.save_detecion_pd_checkpoint(detection_model, PATH_MODELS_CHECKPOINT_STEP_STPE, configs)