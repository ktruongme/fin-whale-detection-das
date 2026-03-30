"""Helpers for rescaling arrays used by the CLI detection pipeline."""

import copy

import numpy as np
import cv2


def match_train_scale(
    data: np.ndarray,
    train_dt: float,
    train_dn: float,
    infer_dt: float,
    infer_dn: float
) -> np.ndarray:
    """Rescale inference data to match training scale.

    Args:
        data (np.ndarray): Inference data to be rescaled.
        train_dt (float): Temporal sampling period of the training data.
        train_dn (float): Spatial sampling interval of the training data.
        infer_dt (float): Temporal sampling period of inference data.
        infer_dn (float): Spatial sampling interval of inference data.

    Returns:
        np.ndarray: Rescaled data.
    """
    target_w = int(round(data.shape[1] * infer_dn / train_dn))
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

    def match_train_scale(
        self,
        train_dt: float,
        train_dn: float,
    ):
        """Rescale data so that its dt/dn match the training sampling
        intervals.
        """
        data_rescaled = match_train_scale(
            data=self,
            train_dt=train_dt,
            train_dn=train_dn,
            infer_dt=self.meta.dt,
            infer_dn=self.meta.dn
        )
        data_rescaled = self.__class__(data_rescaled)
        data_rescaled.meta = copy.deepcopy(self.meta)

        # After rescaling each pixel corresponds to the training sampling
        # intervals.
        new_dt = train_dt
        new_dn = train_dn
        data_rescaled.meta.update(
            dt=new_dt,
            dn=new_dn,
            timestamps=(self.meta.timestamps[0] +
                        np.arange(data_rescaled.shape[0]) * new_dt),
            channels=(self.meta.channels[0] +
                      np.arange(data_rescaled.shape[1]) * new_dn)
        )
        return data_rescaled

    def rgb_transform(self):
        data_rgb = rgb_transform(image=self)
        data_rgb = self.__class__(data_rgb)
        data_rgb.meta = copy.deepcopy(self.meta)
        return data_rgb
