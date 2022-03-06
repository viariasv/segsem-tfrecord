# import the necessary libraries
import os
import cv2
import json
import argparse
import numpy as np

# import the custom libraries
from preprocessing.cropprocessor import CropPreprocessor
from utils.buildoutputfilesSeg import BuildOutputFIlesSeg
from utils.tf_record_utils import (
    create_multispectral_image,
    _create_tf_record
)
# open the config file
with open("config.JSON") as json_data_file:
    config_data = json.load(json_data_file)

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument(
    "-pa",
    "--pattern",
    help="the channels configuration for the database",
    default="1",
)
ap.add_argument(
    "-pr",
    "--protocol",
    help="the ortomaps configuration for training and testing",
    default="ORIGINAL_PROTOCOL",
)
args = vars(ap.parse_args())

# create the config file constants
VAL_SIZE = config_data["VAL_SIZE"]
PATTERN = config_data["PATTERNS"][args["pattern"]]
PROTOCOL = config_data["PROTOCOLS"][args["protocol"]]
NUM_CHANNELS = config_data["CHANNELS"][args["pattern"]]
RAW_DATA_PATH = config_data["DATASET_PATH"]
WEEDMAP_DATASET_OUTPUT = os.path.join(
    "data/RedEdge/output", f"WeedMap_{NUM_CHANNELS}CH"
)

print("[INFO] Creating output folder for the tfrecord files")
# creating output directories
try:
    os.mkdir(WEEDMAP_DATASET_OUTPUT)
    os.mkdir(os.path.join(WEEDMAP_DATASET_OUTPUT, "train"))
    os.mkdir(os.path.join(WEEDMAP_DATASET_OUTPUT, "val"))
    os.mkdir(os.path.join(WEEDMAP_DATASET_OUTPUT, "test"))
except OSError as error:
    print(f"[ERROR] {error}")
    pass

print("[INFO] Loading the partition paths for the dataset")
pathsList = BuildOutputFIlesSeg.create_list_paths(
    path=RAW_DATA_PATH, validation_partition=VAL_SIZE,
    pattern=PATTERN, protocol=PROTOCOL
)
print("[INFO] Partition paths generated!")

# construct a list pairing the training, validation, and testing
# image paths along with their corresponding labels and output HDF5
# files
datasets = [
    (
        "train",
        pathsList[0],
        pathsList[1],
        os.path.join(WEEDMAP_DATASET_OUTPUT, "train"),
    ),
    (
        "val",
        pathsList[2],
        pathsList[3],
        os.path.join(WEEDMAP_DATASET_OUTPUT, "val")
    ),
    (
        "test",
        pathsList[4],
        pathsList[5],
        os.path.join(WEEDMAP_DATASET_OUTPUT, "test")
    ),
]

# initializr the crop preprocessor
cp = CropPreprocessor(224, 224)

crop_images: list = []
crop_masks: list = []

for (dType, pathss, labels, outputFolder) in datasets:
    print(f"[INFO] Construyendo tfrecords para la particion de {dType}")
    for (i, (path, label)) in enumerate(zip(pathss, labels)):
        # load the image and process it
        if (i % 5 != 0) or (i == 0):
            image = create_multispectral_image(path, PATTERN)
            label = cv2.cvtColor(cv2.imread(str(label), 1), cv2.COLOR_BGR2RGB)

            if crop_images == []:
                crop_images = cp.preprocess(image)
                crop_labels = cp.preprocess(label)
            else:
                crop_images = np.concatenate(
                    (crop_images, cp.preprocess(image)), axis=0
                )
                crop_labels = np.concatenate(
                    (crop_labels, cp.preprocess(label)), axis=0
                )
        else:
            output = os.path.join(
                outputFolder, f"weedmap_patch_{i // 5 + 1}.record"
            )
            # create tfrecord
            _create_tf_record(crop_images, crop_labels, output)
            crop_images = []
            crop_labels = []
    print(f"[INFO] tfrecords para la particion de {dType} almacenados")
