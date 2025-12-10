from typing import TYPE_CHECKING
import copy

import numpy as np
import cv2

from ..execution import box_saver

if TYPE_CHECKING:
    from ..core.dasarray import DASArray


def create_v_template(
    v: float,
    dt: float,
    dx: float,
    t: float = None,
    x: float = None,
    t_width: float = None,
    x_width: float = None
) -> np.ndarray[float]:
    """Create a V-shaped template for a given velocity.

    The template is created in the time-space domain, where the V-shape is
    defined by the velocity 'v'. The template is centered at the origin and
    has a width in both time and space dimensions.

    Args:
        v (float): Velocity in m/s.
        dt (float): Temporal sampling period in seconds.
        dx (float): Spatial sampling period in meters.
        t (float, optional): Time duration of the template. Either 't' or 'x'
            must be provided, but not both. Defaults to None.
        x (float, optional): Spatial extent of the template. Either 't' or 'x'
            must be provided, but not both. Defaults to None.
        t_width (float, optional): Width of the template in time. Either
            't_width' or 'x_width' must be provided, but not both. Defaults to
            None.
        x_width (float, optional): Width of the template in space. Either
            't_width' or 'x_width' must be provided, but not both. Defaults to
            None.

    Returns:
        np.ndarray[float]: V-shaped template.
    """
    # Validate input parameters
    if (t is None and x is None) or (t is not None and x is not None):
        raise ValueError("Either 't' or 'x' must be provided, but not both.")
    if (
        (t_width is None and x_width is None) or
        (t_width is not None and x_width is not None)
    ):
        raise ValueError("Either 't_width' or 'x_width' must be provided, "
                         + "but not both.")

    # Determine template size
    if t is not None:
        # Number of time samples
        T = int(np.ceil(t / dt)) + 1
        # Spatial extent from -v*t to +v*t
        X = int(np.ceil((2 * v * t) / dx)) + 1
        x = 2 * v * t
    else:
        X = int(np.ceil(x / dx)) + 1
        T = int(np.ceil(x / (2 * v * dt))) + 1
        t = (T - 1) * dt

    # Determine width in both dimensions
    if t_width is not None:
        TW = t_width
        XW = 2 * v * TW
    else:
        XW = x_width
        TW = x_width / (2 * v)

    # Create grid of time and space
    t_vals = np.linspace(0, t, T).reshape(T, 1)  # Shape (T, 1)
    x_centered = np.linspace(-x/2, x/2, X).reshape(1, X)  # Shape (1, X)

    # Calculate the two edges of the "V"
    # Edge 1: x = v * t
    # Edge 2: x = -v * t

    if x_width is not None:
        # Using spatial width: check |x - v*t| <= x_width/2 or
        # |x + v*t| <= x_width/2
        edge1 = np.abs(x_centered - v * t_vals) <= (XW / 2)
        edge2 = np.abs(x_centered + v * t_vals) <= (XW / 2)
    else:
        # Using temporal width: check |t - (x / v)| <= t_width/2 or
        # |t - (-x / v)| <= t_width/2
        # Since t >=0 and x can be negative, the second condition simplifies to
        # t >= x / v
        # To prevent negative time, we only consider the absolute x
        edge1 = np.abs(t_vals - (x_centered / v)) <= (TW / 2)
        edge2 = np.abs(t_vals + (x_centered / v)) <= (TW / 2)

    # Combine both edges
    template = np.logical_or(edge1, edge2).astype(np.uint8)

    return template


def template_matching(
    data: np.ndarray[np.uint8],
    template: np.ndarray[np.uint8],
    method: int = cv2.TM_CCOEFF_NORMED,
    pad_value: float = None
) -> np.ndarray[float]:
    """Apply template matching to the data.

    The output has the same shape as the input `data`. The matching score at
    position `(i, j)` corresponds to the template centered at that position in
    the input.

    Args:
        data (np.ndarray[np.uint8]): Input data.
        template (np.ndarray[np.uint8]): Template to match.
        method (int, optional): Matching method. More options
            at https://docs.opencv.org/5.x/df/dfb/group__imgproc__object.html.
            Defaults to cv2.TM_CCOEFF_NORMED.
        pad_value (float, optional): Value to pad the result with. If None,
            use BORDER_DEFAULT (BORDER_REFLECT_101). Defaults to None.

    Returns:
        np.ndarray[float]: Matching result.
    """
    # Compute padding amounts so that the output size matches the input size
    pad_top = template.shape[0] // 2
    pad_bottom = template.shape[0] - pad_top - 1
    pad_left = template.shape[1] // 2
    pad_right = template.shape[1] - pad_left - 1

    # Determine border type and value based on pad_value
    if pad_value is None:
        border_type = cv2.BORDER_DEFAULT
        border_value = 0  # ignored for non-constant border types
    else:
        border_type = cv2.BORDER_CONSTANT
        border_value = pad_value

    # Pad the input data so that template matching covers edges
    padded_data = cv2.copyMakeBorder(
        data,
        pad_top, pad_bottom,
        pad_left, pad_right,
        borderType=border_type,
        value=border_value
    )

    # Run template matching on the padded data
    res_full = cv2.matchTemplate(padded_data, template, method)
    return res_full


def clip_negative_values(data: np.ndarray[float]) -> np.ndarray[float]:
    """Sets all negative values in the input array to zero.

    Args:
        data (np.ndarray[float]): Input data.

    Returns:
        np.ndarray[float]: Array with negative values replaced by zero.
    """
    result = np.maximum(data, 0)
    return result


def adjust_template_boxes(
    boxesn: np.ndarray[float],
    data_shape: tuple,
    template_shape: tuple,
    expand_pct: float = 0.25
) -> np.ndarray[float]:
    """Adjust normalized bounding boxes by a percentage of the template size.

    Each box is expanded by half the template height and width (expressed in
    normalized units) so that the adjusted box fully covers the template
    centred on the original box. The result is clipped to stay inside the image
    bounds and returned in the same normalized coordinate system.

    Args:
        boxesn (np.ndarray[float]): Normalized bounding boxes, shape (N, 4).
        data_shape (tuple): Shape of the data (height, width).
        template_shape (tuple): Shape of the template (height, width).
        expand_pct (float, optional): Percentage (0–1) of the template
            width/height by which each side of the box is expanded. ``0.5``
            means the box is expanded by half the template size. Defaults to
            0.25.

    Returns:
        np.ndarray[float]: Adjusted bounding boxes in normalized coordinates.
    """
    # Work on a copy to avoid in‑place modification of the caller’s array
    boxesn = boxesn.copy()

    # Half the template size, expressed in *normalized* units
    half_width_norm = (template_shape[1] * expand_pct) / data_shape[1]
    half_height_norm = (template_shape[0] * expand_pct) / data_shape[0]

    # Expand each box by the template half‑size
    boxesn[:, 0] -= half_width_norm   # x1
    boxesn[:, 1] -= half_height_norm  # y1
    boxesn[:, 2] += half_width_norm   # x2
    boxesn[:, 3] += half_height_norm  # y2

    # Clip to image bounds [0, 1]
    boxesn[:, [0, 2]] = np.clip(boxesn[:, [0, 2]], 0.0, 1.0)
    boxesn[:, [1, 3]] = np.clip(boxesn[:, [1, 3]], 0.0, 1.0)

    return boxesn


class DASTemplate:

    def template_matching(
        self,
        template: np.ndarray[np.uint8],
        method: int = cv2.TM_CCOEFF_NORMED,
        pad_value: float = None
    ) -> 'DASArray':
        """Apply template matching to the data.

        Args:
            template (np.ndarray[np.uint8]): Template to match.
            method (int, optional): Matching method. More options at
                https://docs.opencv.org/5.x/df/dfb/group__imgproc__object.html.
                Defaults to cv2.TM_CCOEFF_NORMED.
            pad_value (float, optional): Value to pad the result with. If None,
                the minimum value of the result is used. Defaults to None.

        Returns:
            DASArray: Matching result.
        """
        matching_result = template_matching(
            data=self,
            template=template,
            method=method,
            pad_value=pad_value
        )
        result = self.__class__(matching_result)
        result.meta = copy.deepcopy(self.meta)
        return result

    def clip_negative_values(self) -> 'DASArray':
        """Sets all negative values in the input array to zero.

        Returns:
            DASArray: Array with negative values replaced by zero.
        """
        return clip_negative_values(self)

    def adjust_template_boxes(self,
                              template_shape: tuple,
                              expand_pct: float = 0.25) -> 'DASArray':
        """Adjust normalized bounding boxes based on the template shape and a
        user‑specified expansion percentage.

        Args:
            template_shape (tuple): The shape of the template (height, width).
            expand_pct (float, optional): Percentage of the template size by
                which to expand each side of the box. Defaults to 0.25.

        Returns:
            DASArray: Updated instance of the DASArray with modified boxes.
        """
        adjusted_boxesn = adjust_template_boxes(
            boxesn=self.meta.boxesn,
            data_shape=self.shape,
            template_shape=template_shape,
            expand_pct=expand_pct
        )

        boxesd = box_saver.denormalize_boxesn(
            boxesn=adjusted_boxesn,
            t_start=self.meta.timestamps[0],
            t_end=self.meta.timestamps[-1],
            s_start=self.meta.channels[0],
            s_end=self.meta.channels[-1],
        )

        boxesp = box_saver.cast_box_times_to_datetime64(boxes=boxesd)

        # Update metadata with the adjusted normalized boxes only
        self.meta.update(boxesn=adjusted_boxesn,
                         boxesp=boxesp)

        return self
