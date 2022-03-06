import io
import os
import cv2
import PIL
import logging
import numpy as np
from typing import Any
import tensorflow as tf


def _bytes_feature(value: tf.Tensor) -> Any:
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() 
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value: float) -> Any:
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value: int) -> Any:
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _open_file(full_path: str) -> tuple:
    """Read an image path, load the image and encoded in bytes"""
    with tf.io.gfile.GFile(full_path, 'rb') as fid:
        encoded_file = fid.read()
    encoded_file_io = io.BytesIO(encoded_file)
    image = PIL.Image.open(encoded_file_io)
    return image, encoded_file


def _serialize_array(array: np.ndarray) -> Any:
    return tf.io.serialize_tensor(array)


def create_multispectral_image(path: str, pattern: list) -> np.ndarray:
    """create the multiespectral input image"""
    # initialize the list of images
    list_arrays = []
    for type_img in pattern:
        path_img = str(path).replace(pattern[0], type_img)

        # take account the difference between 3 and 1 channel images
        if type_img == "RGB" or type_img == "CIR":
            image = cv2.cvtColor(cv2.imread(path_img, 1), cv2.COLOR_BGR2RGB)
        else:
            image = np.expand_dims(cv2.imread(path_img, 0), axis=-1)
        list_arrays.append(image)

    # join the channels
    image_final = np.concatenate(list_arrays, axis=-1)

    return image_final


def create_tf_example(image_array: np.ndarray, mask_array: np.ndarray) -> Any:
    """method for create a tf example for a training example"""
    # convert the image
    encoded_image = _serialize_array(image_array)
    encoded_mask = _serialize_array(mask_array)
    height = image_array.shape[0]
    width = image_array.shape[1]

    feature_dict = {
        'image/input': _bytes_feature(encoded_image),
        'image/height': _int64_feature(height),
        'image/width': _int64_feature(width),
        'image/channels': _int64_feature(image_array.shape[2]),
        'mask/output': _bytes_feature(encoded_mask),
        'mask/channels': _int64_feature(mask_array.shape[2])
    }

    example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
    return example


def _create_tf_record(images: np.ndarray, masks: np.ndarray, output_path: str) -> None:
    logging.info(f'building {output_path}...')
    writer = tf.io.TFRecordWriter(output_path)
    for idx, image in enumerate(images):
        tf_example = create_tf_example(image, masks[idx])
        writer.write(tf_example.SerializeToString())
    writer.close()
    logging.info(f'building of {output_path} finished!')