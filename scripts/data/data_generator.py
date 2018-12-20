import os
from typing import Callable, Tuple
from types import FunctionType
from numbers import Integral

import cv2
import numpy as np


_available_resize_methods = ("central_crop_and_resize", "resize")


def _central_crop(image):
    shape = image.shape
    height, width = shape[0], shape[1]

    crop_size = int(np.round(np.minimum(height, width) * 0.85))

    crop_top = (height - crop_size) // 2
    crop_left = (width - crop_size) // 2

    return image[crop_top:crop_top + crop_size, crop_left:crop_left + crop_size]


def _check_path(path):
    path = os.path.realpath(path)
    if not os.path.exists(path):
        raise FileNotFoundError("Cannot find '{}'".format(path))


def _check_init_parameters(
        path_to_images: str,
        path_to_labels: str,
        impixel_preprocess: Callable,
        imsize_preprocess: str,
        imsize: Integral):

    _check_path(path_to_images)
    _check_path(path_to_labels)

    if impixel_preprocess is not None:
        if not isinstance(impixel_preprocess, FunctionType):
            raise TypeError("`impixel_preprocess` must be either a callable or None")

    if not isinstance(imsize_preprocess, str):
        raise TypeError("`imsize_preprocess` must be a string, indicating what method should be used for resize.")

    if imsize_preprocess not in _available_resize_methods:
        raise ValueError("Preprocess method '{}' is not supported".format(imsize_preprocess))

    if not isinstance(imsize, Integral):
        raise TypeError("Specified image size must be an integer number")

    if imsize < 2:
        raise ValueError("Image size must be at least two pixels.")


class DataGenerator:
    """Creates a generator which yields batches of preprocessed images."""

    def __init__(
            self,
            path_to_images,
            path_to_labels,
            impixel_preprocess=None,
            imsize_preprocess="central_crop_and_resize",
            imsize=224):
        """Creates a generator which yields batches of preprocessed images.

        Parameters
        ----------
        path_to_images: str
            A path to the folder containing images
        path_to_labels: str
            A path to the list of labels formed from pairs *(path/to/image.JPEG, image_label*

            Path to the image is considered as relative to the specified `path_to_images`
        impixel_preprocess: Callable
            A function which modifies the pixel data of the input images
        imsize_preprocess: str
            A method used to resize the input image:

             * "resize" - simple resize
             * "central_crop_and_resize" - cut the central square part of the image
               and then resize it
        imsize: Integral
            The size of the output square images.
        """

        _check_init_parameters(path_to_images, path_to_labels, impixel_preprocess, imsize_preprocess, imsize)

        self._path_to_images = path_to_images
        self._path_to_labels = path_to_labels
        self._imsize = imsize
        self._impixel_preprocess = impixel_preprocess if impixel_preprocess is not None else lambda x: x

        if imsize_preprocess == "central_crop_and_resize":
            self._imsize_preprocess = lambda x: cv2.resize(_central_crop(x), (self._imsize, self._imsize))
        elif imsize_preprocess == "resize":
            self._imsize_preprocess = lambda x: cv2.resize(x, (self._imsize, self._imsize))

        else:
            raise NotImplementedError("Image resize method '{}' is not implemented")

    def generate_batches(self, batch_size, max_batch_number=None) -> Tuple[np.ndarray, np.ndarray]:
        """Creates a generator which yields batched data.

        Parameters
        ----------
        batch_size: int
            The number of images per one batch
        max_batch_number: int, optional
            Limits the number of batches that will be yield
        Yields
        ------
        Tuple[np.ndarray, np.ndarray]:
            A pair (batched_images, batched_labels)
        """
        with open(self._path_to_labels, "r") as f:
            lines = list(map(lambda x: x[:-1], f.readlines()))

        im_path, im_lbl = zip(*list(map(lambda x: x.split(" ", 1), lines)))
        im_lbl = np.array(im_lbl, dtype=np.int32)
        images_index = np.arange(im_lbl.shape[0])
        np.random.shuffle(images_index)
        total_batch_count = images_index.shape[0] // batch_size

        for btch_idx in range(total_batch_count-1):

            if max_batch_number is not None:
                if btch_idx >= max_batch_number:
                    return

            curr_image_idx = images_index[btch_idx * batch_size: (btch_idx + 1) * batch_size]
            im_batch = -1*np.ones([batch_size, self._imsize, self._imsize, 3])
            label_batch = -1*np.ones([batch_size])

            for i in range(batch_size):
                path_to_im = os.path.join(self._path_to_images, im_path[curr_image_idx[i]])
                raw_im = cv2.imread(path_to_im)[:, :, ::-1]
                curr_im = self._impixel_preprocess(self._imsize_preprocess(raw_im))
                im_batch[i] = curr_im
                label_batch[i] = im_lbl[curr_image_idx[i]]

            yield im_batch, label_batch
