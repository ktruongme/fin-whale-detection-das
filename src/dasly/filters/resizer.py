"""This module contains functions for downsampling and resizing data.

The module provides utilities for:
    - Block aggregation downsampling.
    - Decimation-based downsampling.
    - Resizing images with OpenCV.
    - Rescaling images to training scales.
    - Letterboxing images.
    - Converting grayscale images to RGB.
"""

import logging
import copy

import numpy as np
import cv2

# Set up logger
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def resize_cv2(
    data: np.ndarray[float],
    new_size: tuple[int],
    interpolation=cv2.INTER_LINEAR
) -> np.ndarray[float]:
    """Resize data to a new size using OpenCV.

    Args:
        data (np.ndarray[float]): Data to be resized.
        new_size (tuple[int]): New size as (width, height).
        interpolation (int, optional): Interpolation method. Options:
            cv2.INTER_NEAREST: Nearest neighbor interpolation.
                Fast but may produce blocky images.
            cv2.INTER_LINEAR: Bilinear interpolation.
                Good for general purposes.
            cv2.INTER_CUBIC: Bicubic interpolation.
                Better quality but slower.
            cv2.INTER_AREA: Pixel area relation.
                Best for shrinking.
            Defaults to cv2.INTER_LINEAR.

    Returns:
        np.ndarray[float]: Resized data.
    """
    resized = cv2.resize(src=data,
                         dsize=new_size,
                         interpolation=interpolation)
    return resized


def match_train_scale(
    data: np.ndarray,
    train_dt: float,
    train_ds: float,
    infer_dt: float,
    infer_ds: float
) -> np.ndarray:
    """Rescale inference data to match training scale.

    Args:
        data (np.ndarray): Inference data to be rescaled.
        train_dt (float): Temporal sampling period of the training data.
        train_ds (float): Spatial sampling interval of the training data.
        infer_dt (float): Temporal sampling period of inference data.
        infer_ds (float): Spatial sampling interval of inference data.

    Returns:
        np.ndarray: Rescaled data.
    """
    # Scaling factors along time (rows) and space (columns)
    target_w = int(round(data.shape[1] * infer_ds / train_ds))
    target_h = int(round(data.shape[0] * infer_dt / train_dt))

    rescaled = cv2.resize(data, (target_w, target_h))
    return rescaled


def rgb_transform(image: np.ndarray) -> np.ndarray:
    """Convert grayscale image to RGB.

    Args:
        image (np.ndarray): Grayscale image of shape (H, W).

    Returns:
        np.ndarray: RGB image of shape (H, W, 3).
    """
    image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image_rgb


class DASResizer:

    def resize_cv2(
        self,
        new_size: tuple[int],
        interpolation=cv2.INTER_LINEAR
    ):
        data_resized = resize_cv2(data=self,
                                  new_size=new_size,
                                  interpolation=interpolation)
        data_resized = self.__class__(data_resized)
        data_resized.meta = copy.deepcopy(self.meta)

        # Update metadata
        # new_size is (width, height); self.shape is (height, width)
        new_dt = self.meta.dt * self.shape[0] / new_size[1]
        new_ds = self.meta.ds * self.shape[1] / new_size[0]
        data_resized.meta.update(
            dt=new_dt,
            ds=new_ds,
            timestamps=(self.meta.timestamps[0] +
                        np.arange(data_resized.shape[0]) * new_dt),
            channels=(self.meta.channels[0] +
                      np.arange(data_resized.shape[1]) * new_ds)
        )
        return data_resized

    def match_train_scale(
        self,
        train_dt: float,
        train_ds: float,
    ):
        """Rescale data so that its dt/ds match the training sampling
        intervals.
        """
        data_rescaled = match_train_scale(
            data=self,
            train_dt=train_dt,
            train_ds=train_ds,
            infer_dt=self.meta.dt,
            infer_ds=self.meta.ds
        )
        data_rescaled = self.__class__(data_rescaled)
        data_rescaled.meta = copy.deepcopy(self.meta)

        # After rescaling each pixel corresponds to the training sampling
        # intervals.
        new_dt = train_dt
        new_ds = train_ds
        data_rescaled.meta.update(
            dt=new_dt,
            ds=new_ds,
            timestamps=(self.meta.timestamps[0] +
                        np.arange(data_rescaled.shape[0]) * new_dt),
            channels=(self.meta.channels[0] +
                      np.arange(data_rescaled.shape[1]) * new_ds)
        )
        return data_rescaled

    def rgb_transform(self):
        data_rgb = rgb_transform(image=self)
        data_rgb = self.__class__(data_rgb)
        data_rgb.meta = copy.deepcopy(self.meta)
        return data_rgb
