import copy
from typing import Tuple

import numpy as np


def slice_by_normalized_coords(
    array: np.ndarray,
    coords: Tuple[float, float, float, float]
) -> np.ndarray:
    """Slice a 2D numpy array using normalized coordinates.

    Args:
        array (np.ndarray): Input 2D array.
        coords (Tuple[float, float, float, float]): Normalized coordinates
            (x1, y1, x2, y2), where 0 <= x1, y1, x2, y2 <= 1.

    Returns:
        np.ndarray: The sliced array. If the input array is a DASArray, then
            the resulting array will also be a DASArray with updated metadata.
    """
    if len(coords) != 4:
        raise ValueError("Coordinates tuple must have exactly 4 elements " +
                         "(x1, y1, x2, y2).")
    if not all(0 <= val <= 1 for val in coords):
        raise ValueError("All coordinates must be normalized values between " +
                         "0 and 1.")

    x1, y1, x2, y2 = coords
    rows, cols = array.shape

    # Convert normalized coordinates to absolute indices
    x1_idx = int(round(x1 * cols))
    y1_idx = int(round(y1 * rows))
    x2_idx = int(round(x2 * cols))
    y2_idx = int(round(y2 * rows))

    # Ensure indices are within bounds
    x1_idx, x2_idx = sorted((max(0, x1_idx), min(cols, x2_idx)))
    y1_idx, y2_idx = sorted((max(0, y1_idx), min(rows, y2_idx)))

    sliced_array = array[y1_idx:y2_idx, x1_idx:x2_idx]

    # Handle special DASArray type by updating metadata if necessary.
    from ..core.dasarray import DASArray

    def _is_dasarray(obj) -> bool:
        return isinstance(obj, DASArray)

    if _is_dasarray(array):
        sliced_array = copy.deepcopy(sliced_array)
        sliced_channels = array.meta.channels[x1_idx:x2_idx].copy()
        sliced_timestamps = array.meta.timestamps[y1_idx:y2_idx].copy()
        sliced_array.meta = copy.deepcopy(array.meta)
        sliced_array.meta.update(
            channels=sliced_channels,
            timestamps=sliced_timestamps
        )

    return sliced_array


def extract_binary_coords(
    binary_array: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract the x and y coordinates from the binary 2D array where the value
    is 1.

    Args:
        binary_array (np.ndarray): Binary 2D array.

    Returns:
        Tuple[np.ndarray, np.ndarray]: x and y coordinates where the value is
            1.
    """
    coords = np.argwhere(binary_array == 1)
    # Note: np.argwhere returns (row, col); we treat row as y and col as x.
    y_coords = coords[:, 0]
    x_coords = coords[:, 1]
    return x_coords, y_coords
