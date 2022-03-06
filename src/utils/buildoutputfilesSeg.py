import os
import cv2
import progressbar
import numpy as np
from glob import glob
from typing import Dict, List
from sklearn.model_selection import train_test_split
from preprocessing.cropprocessor import CropPreprocessor


class BuildOutputFIlesSeg:
    def __init__(self, path_list: list, hdf5_paths: list, pattern: list, channels: int):
        """build the hdf5 files"""
        self.path_list = path_list
        self.hdf5_paths = hdf5_paths
        self.pattern = pattern
        self.channels = channels

        # run iternally all the
        self.process()

    def process(self):
        # construct a list pairing the training, validation, and testing
        # image paths along with their corresponding labels and output HDF5
        # files
        datasets = [
            ("train", self.path_list[0], self.path_list[1], self.hdf5_paths[0]),
            ("val", self.path_list[2], self.path_list[3], self.hdf5_paths[1]),
            ("test", self.path_list[4], self.path_list[5], self.hdf5_paths[2]),
        ]

        input_dim = (224, 224, self.channels)
        output_dim = (224, 224, 3)

        # initialize the crop preprocessor
        cp = CropPreprocessor(224, 224)

        # loop over the dataset tuples
        for (dType, pathss, labels, outputPath) in datasets:
            # create HDF5 writer
            print("[INFO] building {}...".format(outputPath))
            writer = HDF5DatasetWriter([(len(pathss) * 5, 224, 224, self.channels), (len(pathss) * 5, 224, 224, 3)], outputPath)

            # initialize the progress bar
            widgets = ["Building Dataset: ", progressbar.Percentage(), " ", progressbar.Bar(), " ", progressbar.ETA()]
            pbar = progressbar.ProgressBar(maxval=len(pathss), widgets=widgets).start()

            # loop over the image paths
            for (i, (path, label)) in enumerate(zip(pathss, labels)):
                # load the image and process it

                image = self.create_multispectral_image(path)
                path_images = cp.preprocess(image)

                label = cv2.cvtColor(cv2.imread(str(label), 1), cv2.COLOR_BGR2RGB)
                path_labels = cp.preprocess(label)

                for i, (img, lbl) in enumerate(zip(path_images, path_labels)):
                    # add the image and label # to the HDF5 dataset
                    writer.add([img], [lbl])
                    pbar.update(i)

            # close the HDF5 writer
            pbar.finish()
            writer.close()

    def create_multispectral_image(self, path):
        """create the multiespectral input image"""
        # initialize the list of images
        list_arrays = []
        for type_img in self.pattern:
            path_img = str(path).replace(self.pattern[0], type_img)

            # take account the difference between 3 and 1 channel images
            if type_img == "RGB" or type_img == "CIR":
                image = cv2.cvtColor(cv2.imread(path_img, 1), cv2.COLOR_BGR2RGB)
            else:
                image = np.expand_dims(cv2.imread(path_img, 0), axis=-1)
            list_arrays.append(image)

        # join the channels
        image_final = np.concatenate(list_arrays, axis=-1)

        return image_final

    @staticmethod
    def create_list_paths(path: str, validation_partition: float, protocol: Dict[str, list], pattern: List[str]):
        """Extract the paths to the train/val/test data base of the data settings"""

        # initialize the paths lists
        trainPaths = []
        trainLabels = []
        testPaths = []
        testLabels = []

        for div, folders in protocol.items():
            if div == "train":
                for folder in folders:
                    # images
                    train_path = os.path.join(path, folder, "tile", pattern[0])
                    train_path_list = list(glob(os.path.join(train_path, "*.png")))
                    trainPaths += train_path_list
                    # masks
                    train_label = os.path.join(path, folder, "groundtruth")
                    train_label_list = list(glob(os.path.join(train_label, "*color.png")))
                    trainLabels += train_label_list
            else:
                for folder in folders:
                    # images
                    test_path = os.path.join(path, folder, "tile", pattern[0])
                    test_path_list = list(glob(os.path.join(test_path, "*.png")))
                    testPaths += test_path_list
                    # masks
                    test_label = os.path.join(path, folder, "groundtruth")
                    test_label_list = list(glob(os.path.join(test_label, "*color.png")))
                    testLabels += test_label_list

        # sort the paths to the images
        trainPaths = sorted(trainPaths, key=lambda X: (str(X).split("/")[3], str(X).split("/")[-1]))
        trainLabels = sorted(trainLabels, key=lambda X: (str(X).split("/")[3], str(X).split("/")[-1]))
        testPaths = sorted(testPaths, key=lambda X: (str(X).split("/")[3], str(X).split("/")[-1]))
        testLabels = sorted(testLabels, key=lambda X: (str(X).split("/")[3], str(X).split("/")[-1]))

        # perform another stratified sampling, this time to build the
        # validation data
        split = train_test_split(trainPaths, trainLabels, test_size=validation_partition, random_state=42)
        (trainPaths, valPaths, trainLabels, valLabels) = split

        return trainPaths, trainLabels, valPaths, valLabels, testPaths, testLabels
