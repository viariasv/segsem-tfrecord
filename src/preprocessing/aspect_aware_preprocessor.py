# import the necessary libraries
import cv2
import imutils
import numpy as np


class AspectAwarePreprocessor:
    def __init__(self, width: int, height: int, inter: int = cv2.INTER_AREA):
        # store the target image width, height, and interpolation
        # method used when resizing
        self.width = width
        self.height = height
        self.inter = inter

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        # grab the dimensions of the image and then initialize
        # the deltas to use when caspropping
        (h, w, depth) = image.shape
        dW = 0
        dH = 0

        # if the width is smaller than the height, then resize
        # along the width (i.e., the smaller dimension) and then
        # update the deltas to crop the height to the desired
        # dimension
        if w < h:
            if depth == 3:
                image = imutils.resize(image, width=self.width, inter=self.inter)
                dH = int((image.shape[0] - self.height) / 2.0)
            elif depth == 4:
                image_rgb = imutils.resize(image[:3], width=self.width, inter=self.inter)
                image_nir = imutils.resize(image[3], width=self.width, inter=self.inter)
                dH = int((image_rgb.shape[0] - self.height) / 2.0)
            elif depth == 5:
                image_rgb = imutils.resize(image[:3], width=self.width, inter=self.inter)
                image_nir = imutils.resize(image[3], width=self.width, inter=self.inter)
                image_re = imutils.resize(image[4], width=self.width, inter=self.inter)
                dH = int((image_rgb.shape[0] - self.height) / 2.0)
        # otherwise, the height is smaller than the width so
        # resize along the height and then update the deltas
        # crop along the width
        else:
            if depth == 3:
                image = imutils.resize(image, height=self.height, inter=self.inter)
                dW = int((image.shape[1] - self.width) / 2.0)
            elif depth == 4:
                image_rgb = imutils.resize(image[:, :, :3], height=self.height, inter=self.inter)
                image_nir = imutils.resize(image[:, :, 3], height=self.height, inter=self.inter)
                dW = int((image_rgb.shape[1] - self.width) / 2.0)
            elif depth == 5:
                image_rgb = imutils.resize(image[:, :, :3], height=self.height, inter=self.inter)
                image_nir = imutils.resize(image[:, :, 3], height=self.height, inter=self.inter)
                image_re = imutils.resize(image[:, :, :4], height=self.height, inter=self.inter)
                dW = int((image_rgb.shape[1] - self.width) / 2.0)

        # now that our images have been resized, we need to
        # re-grab the width and height, followed by performing
        # the crop
        if depth == 3:
            (h, w) = image.shape[:2]
            image = image[dH:h - dH, dW:w - dW]
        elif depth == 4:
            (h, w) = image_rgb.shape[:2]
            image_rgb = image_rgb[dH:h - dH, dW:w - dW]
            image_nir = image_nir[dH:h - dH, dW:w - dW]
        elif depth == 5:
            (h, w) = image_rgb.shape[:2]
            image_rgb = image_rgb[dH:h - dH, dW:w - dW]
            image_nir = image_nir[dH:h - dH, dW:w - dW]
            image_re = image_re[dH:h - dH, dW:w - dW]
        # finally, resize the image to the provided spatial
        # dimensions to ensure our output image is always a fixed
        # size
        if depth == 3:
            image_final = cv2.resize(image, (self.width, self.height), interpolation=self.inter)
        elif depth == 4:
            image_rgb = cv2.resize(image_rgb, (self.width, self.height), interpolation=self.inter)
            image_nir = np.expand_dims(
                cv2.resize(image_nir, (self.width, self.height), interpolation=self.inter),
                axis=-1,
            )
            image_final = np.concatenate([image_rgb, image_nir], axis=-1)
        elif depth == 5:
            image_rgb = cv2.resize(image_rgb, (self.width, self.height), interpolation=self.inter)
            image_nir = np.expand_dims(
                cv2.resize(image_nir, (self.width, self.height), interpolation=self.inter),
                axis=-1,
            )
            image_re = np.expand_dims(
                cv2.resize(image_re, (self.width, self.height), interpolation=self.inter),
                axis=-1,
            )
            image_final = np.concatenate([image_rgb, image_nir, image_re], axis=-1)

        return image_final
