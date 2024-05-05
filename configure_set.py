import os
import tensorflow as tf
from object_detection.utils import config_util
from object_detection.protos import pipeline_pb2
from google.protobuf import text_format
import subprocess
import tarfile
import object_detection
import shutil
import requests
import zipfile




CUSTOM_MODEL_NAME = 'my_ssd_mobnet_tuned_v1.1'
PRETRAINED_MODEL_NAME = 'ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8'
PRETRAINED_MODEL_URL = 'http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz'
TF_RECORD_SCRIPT_NAME = 'generate_tfrecord.py'
LABEL_MAP_NAME = 'label_map.pbtxt'


paths = {
    'WORKSPACE_PATH': os.path.join('Dataset', 'workspace'),
    'SCRIPTS_PATH': os.path.join('Dataset', 'scripts'),
    'APIMODEL_PATH': os.path.join('Dataset', 'models'),
    'ANNOTATION_PATH': os.path.join('Dataset', 'workspace', 'annotations'),
    'IMAGE_PATH': os.path.join('Dataset', 'images'),
    'MODEL_PATH': os.path.join('Dataset', 'workspace', 'models'),
    'PRETRAINED_MODEL_PATH': os.path.join('Dataset', 'workspace','pre-trained-models'),
    'CHECKPOINT_PATH': os.path.join('Dataset', 'workspace','models',CUSTOM_MODEL_NAME),
    'OUTPUT_PATH': os.path.join('Dataset', 'workspace','models',CUSTOM_MODEL_NAME, 'export'),
    'TFJS_PATH':os.path.join('Dataset', 'workspace','models',CUSTOM_MODEL_NAME, 'tfjsexport'),
    'TFLITE_PATH':os.path.join('Dataset', 'workspace','models',CUSTOM_MODEL_NAME, 'tfliteexport'),
    'PROTOC_PATH':os.path.join('Dataset','protoc')
 }

files = {
    'PIPELINE_CONFIG':os.path.join('Dataset', 'workspace','models', CUSTOM_MODEL_NAME, 'pipeline.config'),
    'TF_RECORD_SCRIPT': os.path.join(paths['SCRIPTS_PATH'], TF_RECORD_SCRIPT_NAME),
    'LABELMAP': os.path.join(paths['ANNOTATION_PATH'], LABEL_MAP_NAME)
}

# config = config_util.get_configs_from_pipeline_file(files['PIPELINE_CONFIG'])

# for path in paths.values():
#     if not os.path.exists(path):
#         os.makedirs(path)
#
# if os.name == 'nt':
#     # Download the file
#     response = requests.get(PRETRAINED_MODEL_URL)
#     with open(PRETRAINED_MODEL_NAME + '.tar.gz', 'wb') as f:
#         f.write(response.content)
#
#     # Move the downloaded file to PRETRAINED_MODEL_PATH
#     os.rename(PRETRAINED_MODEL_NAME + '.tar.gz',
#               os.path.join(paths['PRETRAINED_MODEL_PATH'], PRETRAINED_MODEL_NAME + '.tar.gz'))
#
#     # Extract the tar.gz file
#     with tarfile.open(os.path.join(paths['PRETRAINED_MODEL_PATH'], PRETRAINED_MODEL_NAME + '.tar.gz'), 'r:gz') as tar:
#         tar.extractall(paths['PRETRAINED_MODEL_PATH'])


labels = [{'name':'no', 'id':1}, {'name':'number0', 'id':2}, {'name':'number1', 'id':3}, {'name':'okay', 'id':4}, {'name':'peace', 'id':5}, {'name':'thumbsdown', 'id':6}, {'name':'thumbsup', 'id':7}]
# #
# with open(files['LABELMAP'], 'w') as f:
#     for label in labels:
#         f.write('item { \n')
#         f.write('\tname:\'{}\'\n'.format(label['name']))
#         f.write('\tid:{}\n'.format(label['id']))
#         f.write('}\n')
# #
# if not os.path.exists(files['TF_RECORD_SCRIPT']):
#     # Clone the repository
#     subprocess.run(["git", "clone", "https://github.com/nicknochnack/GenerateTFRecord", paths['SCRIPTS_PATH']])

# # Generate TFRecord for training data
# train_record_command = [
#     "python", files['TF_RECORD_SCRIPT'],
#     "-x", os.path.join(paths['IMAGE_PATH'], 'train'),
#     "-l", files['LABELMAP'],
#     "-o", os.path.join(paths['ANNOTATION_PATH'], 'train.record')
# ]
# subprocess.run(train_record_command)
# #
# # # Generate TFRecord for testing data
# test_record_command = [
#     "python", files['TF_RECORD_SCRIPT'],
#     "-x", os.path.join(paths['IMAGE_PATH'], 'test'),
#     "-l", files['LABELMAP'],
#     "-o", os.path.join(paths['ANNOTATION_PATH'], 'test.record')
# ]
# subprocess.run(test_record_command)

# Check if the operating system is Windows
# if os.name == 'nt':
#     # Construct the source file path
#     src_file = os.path.join(paths['PRETRAINED_MODEL_PATH'], PRETRAINED_MODEL_NAME, 'pipeline.config')
#     # Construct the destination directory path
#     dest_dir = paths['CHECKPOINT_PATH']
#     # Construct the destination file path
#     dest_file = os.path.join(dest_dir, 'pipeline.config')
#
#     # Ensure the destination directory exists
#     os.makedirs(dest_dir, exist_ok=True)
#
#     # Copy the file
#     shutil.copy(src_file, dest_file)
#     print(f"File copied to {dest_file}")
# else:
#     print("This script is intended to run on Windows systems.")


# config
# #
# #
# pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
# with tf.io.gfile.GFile(files['PIPELINE_CONFIG'], "r") as f:
#     proto_str = f.read()
#     text_format.Merge(proto_str, pipeline_config)
#
#
#
#
# #
# pipeline_config.model.ssd.num_classes = len(labels)
# pipeline_config.train_config.batch_size = 4
# pipeline_config.train_config.fine_tune_checkpoint = os.path.join(paths['PRETRAINED_MODEL_PATH'], PRETRAINED_MODEL_NAME, 'checkpoint', 'ckpt-0')
# pipeline_config.train_config.fine_tune_checkpoint_type = "detection"
# pipeline_config.train_input_reader.label_map_path= files['LABELMAP']
# pipeline_config.train_input_reader.tf_record_input_reader.input_path[:] = [os.path.join(paths['ANNOTATION_PATH'], 'train.record')]
# pipeline_config.eval_input_reader[0].label_map_path = files['LABELMAP']
# pipeline_config.eval_input_reader[0].tf_record_input_reader.input_path[:] = [os.path.join(paths['ANNOTATION_PATH'], 'test.record')]
# #
# #
# config_text = text_format.MessageToString(pipeline_config)
#
#
# with tf.io.gfile.GFile(files['PIPELINE_CONFIG'], "wb") as f:
#     f.write(config_text)


TRAINING_SCRIPT = os.path.join(paths['APIMODEL_PATH'], 'research', 'object_detection', 'model_main_tf2.py')

# command = "python {} --model_dir={} --pipeline_config_path={} --num_train_steps=3000".format(TRAINING_SCRIPT, paths['CHECKPOINT_PATH'],files['PIPELINE_CONFIG'])

# print(command)
#

# command = "python {} --model_dir={} --pipeline_config_path={} --checkpoint_dir={}".format(TRAINING_SCRIPT, paths['CHECKPOINT_PATH'],files['PIPELINE_CONFIG'], paths['CHECKPOINT_PATH'])
#
# # # new = python Dataset\models\research\object_detection\model_main_tf2.py --model_dir=Dataset\workspace\models\my_ssd_mobnet --pipeline_config_path=Dataset\workspace\models\my_ssd_mobnet\pipeline.config --num_train_steps=2000
# #
# subprocess.run(command, shell=True)
#
# FREEZE_SCRIPT = os.path.join(paths['APIMODEL_PATH'], 'research', 'object_detection', 'exporter_main_v2.py ')
#
# command = "python {} --input_type=image_tensor --pipeline_config_path={} --trained_checkpoint_dir={} --output_directory={}".format(FREEZE_SCRIPT ,files['PIPELINE_CONFIG'], paths['CHECKPOINT_PATH'], paths['OUTPUT_PATH'])
#
# subprocess.run(command, shell=True)