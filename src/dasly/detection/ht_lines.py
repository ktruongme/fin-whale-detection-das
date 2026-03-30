"""Provide functions to detect line segments in the data using Hough transform.
"""

import math
from typing import Callable, TYPE_CHECKING
import logging
from datetime import datetime

import numpy as np
import pandas as pd
import cv2
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.stats import trim_mean
from scipy.stats.mstats import trimmed_std

from ..execution import box_saver

if TYPE_CHECKING:
    from ..core.dasarray import DASArray

# Set up logger
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def compute_hough_theta(
    dt: float,
    dn: float,
    dxn: float,
    target_velocity: float,
    velocity_res: float
) -> float:
    """Calculate theta for Hough transform.

    Theta is the angle resolution of the accumulator in radians. Increasing
    theta might velocity up the computation but can make it less accurate, and
    vice versa. The suitable theta can be infered back based on the desired
    velocity resolution. Important: note that this is a heuristic approach,
    because when combining theta and rho, the actual velocity resolution will
    much finner than the desired velocity resolution.

    Args:
        dt (float): Temporal sampling period.
        dn (float): Spatial channnel sampling period.
        dxn (float): Channel spacing, in meters.
        target_velocity (float): Desired velocity to be tracked, in m/s.
        velocity_res (float): Desired velocity resolution, in m/s.

    Returns:
        float: Theta for Hough transform
    """
    # Convert target velocity to channel per second
    target_velocity_c = target_velocity / dxn
    target_velocity_c = np.abs(target_velocity_c)
    # Angle of target velocity (radians)
    angle1 = math.atan((1 / target_velocity_c) * (dn / dt))
    # Angle of target velocity + resolution (radians)
    angle2 = math.atan(
        (1 / (target_velocity_c + velocity_res)) * (dn / dt))
    # Calculate theta (radians)
    theta = np.abs(angle1 - angle2)
    return theta


def compute_hough_line_length(
    dt: float,
    dn: float,
    dxn: float,
    target_velocity: float,
    len_m: float = None,
    len_s: float = None,
) -> float:
    """Calculate length in pixel (for Hough transform) of the line segment to
    be detected, from the length in either meters or seconds of the signal.


    Note that the length is calculated based on the Pythagorean theorem, in
    which assume that spatial dimension and temporal dimension are equally
    spaced. This is a heuristic approach and may not be accurate in all cases.
    For example, if we want to detect a signal having length of 100 meters,
    this function will output length in pixel. But the Hough transform might
    detect much shoter signal (in meters) if the signal is more vertial and the
    temporal sampling rate is high.It is recommended to try different
    combinations of parameters and take the smallest length in pixel returned.


    Args:
        dt (float): Temporal sampling period.
        dn (float): Spatial channnel sampling period.
        dxn (float): Channel spacing, in meters.
        target_velocity (float): Desired velocity to be tracked, in m/s.
        len_m (float, optional): Length of the signal to be tracked in meters.
            Defaults to None.
        len_s (float, optional): Length of the signal to be tracked in seconds.
            Defaults to None.

    Returns:
        float: Length in pixel of the signal to be detected
    """
    if (
        (len_m is None and len_s is None) or
        (len_m is not None and len_s is not None)
    ):
        raise ValueError("Either meters or seconds must be provided.")
    # Convert target velocity to channel per second
    target_velocity_c = target_velocity / dxn
    # Calculate the length in channel and seconds
    if len_m:
        len_c = len_m / dxn  # Convert length in meters to channel
        len_s = len_c / target_velocity_c
    else:
        len_c = target_velocity_c * len_s
    len_pixel = np.sqrt(
        (len_c / dn) ** 2 +
        (len_s / dt) ** 2
    )
    return len_pixel


def hough_lines(
    data: np.ndarray[np.uint8],
    rho: float,
    theta: float | Callable,
    threshold: int,
    minLineLength: float,
    maxLineGap: float
) -> np.ndarray[np.int32]:
    """Apply Hough transform to detect line segments in the binaray image data.

    Args:
        data (np.ndarray[np.uint8]): Data to detect line segments.
        rho (float): Distance resolution of the accumulator in pixels.
        theta (float): Angle resolution of the accumulator in radians.
        threshold (int): Accumulator threshold parameter. Only those lines
            are returned that get enough votes ( > threshold).
        minLineLength (float): Minimum length of line. Line segments shorter
            than this are rejected.
        maxLineGap (float): Maximum allowed gap between line segments to treat
            them as single line.

    Returns:
        np.ndarray[np.int32]: Detected line segments in the format of
            (x1, y1, x2, y2).
    """
    lines = cv2.HoughLinesP(
        data,
        rho=rho,
        theta=theta,
        threshold=threshold,
        minLineLength=minLineLength,
        maxLineGap=maxLineGap
    )
    if lines is None:
        lines = np.empty((0, 4))  # empty array
    else:  # lines are detected
        # note that the default shape of output lines is (N, 1, 4), where n
        # is the number of lines. The additional dimension in the middle is
        # designed to maintain a consistent multi-dimensional structure,
        # providing compatibility with other OpenCV functions. In this
        # project's context, we only need the 4 coordinates of each line,
        # so we can remove the middle dimension by np.squeeze() function.
        lines = np.squeeze(lines, axis=1)
        lines = _standardize_line_endpoints(lines)  # Reorder coordinates
    return lines


def _standardize_line_endpoints(
    lines: np.ndarray[float | pd.Timestamp | datetime]
) -> np.ndarray[float | pd.Timestamp | datetime]:
    """Reorder the 2 endpoints of lines segments so that y1 <= y2. If y1 == y2,
    ensure x1 <= x2.

    Args:
        lines (np.ndarray[float | pd.Timestamp | datetime]): Array of line
            segments. Shape (N, 4) where N is the number of line segments, or
            (4,) for a single line. Each line segment is defined by:
            - s1 (float): Spatial start index of the line segment.
            - t1 (float | pd.Timestamp, datetime): Start timestamp.
            - s2 (float): Spatial end index of the line segment.
            - t2 (float | pd.Timestamp, datetime): End timestamp.

    Returns:
        np.ndarray[float | pd.Timestamp | datetime]: Reordered array of
            line segments. Shape (N, 4) where N is the number of line segments,
            if input is 2D array. If input is 1D array, return (4,).
    """
    # Create a copy of the original array to avoid modifying it
    lines_copy = lines.copy()

    # If input is 1D array of shape (4,), reshape it to (1, 4)
    single_line = False
    if lines.ndim == 1:
        lines = np.atleast_2d(lines)
        single_line = True

    # Extract the coordinates
    x1 = lines_copy[:, 0]
    y1 = lines_copy[:, 1]
    x2 = lines_copy[:, 2]
    y2 = lines_copy[:, 3]

    # Create a mask where y1 > y2
    mask_y = y1 > y2

    # Swap coordinates where the mask is True (y1 > y2)
    x1[mask_y], x2[mask_y] = x2[mask_y], x1[mask_y]
    y1[mask_y], y2[mask_y] = y2[mask_y], y1[mask_y]

    # Create a mask where y1 == y2 and x1 > x2
    mask_x = (y1 == y2) & (x1 > x2)

    # Swap coordinates where the mask is True (y1 == y2 and x1 > x2)
    x1[mask_x], x2[mask_x] = x2[mask_x], x1[mask_x]
    y1[mask_x], y2[mask_x] = y2[mask_x], y1[mask_x]

    # Combine the coordinates back into the lines array
    lines_copy = np.stack((x1, y1, x2, y2), axis=1)

    # If it was a single line segment, reshape the output back to (4,)
    if single_line:
        return lines_copy.reshape(4,)

    return lines_copy


def _compute_space_overlap(
    lines1: np.ndarray,  # Shape (N, 4) or (4,)
    lines2: np.ndarray   # Shape (M, 4) or (4,)
) -> np.ndarray:
    """Calculate the space overlaping indices limits of line segments.

    Args:
        lines1 (np.ndarray): Line segments 1. Shape (N, 4) or (4,), where each
            row is defined by:
            - s1 (Union[int, float]): Start index of the line segment.
            - t1 (Union[int, float, pd.Timestamp, datetime]): Start timestamp.
            - s2 (Union[int, float]): End index of the line segment.
            - t2 (Union[int, float, pd.Timestamp, datetime]): End timestamp.
        lines2 (np.ndarray): Line segments 2. Shape (M, 4) or (4,), where each
            row is defined by:
            - s1 (Union[int, float]): Start index of the line segment.
            - t1 (Union[int, float, pd.Timestamp, datetime]): Start timestamp.
            - s2 (Union[int, float]): End index of the line segment.
            - t2 (Union[int, float, pd.Timestamp, datetime]): End timestamp.

    Returns:
        np.ndarray: Space overlaping indices limits. Shape (N, M, 2) or (2,) if
            input shapes were (4,).
    """
    # If inputs are 1D arrays of shape (4,), reshape them to (1, 4)
    single_lines = False
    if lines1.ndim == 1 and lines2.ndim == 1:
        lines1 = np.atleast_2d(lines1)
        lines2 = np.atleast_2d(lines2)
        single_lines = True

    # Extract the relevant parts of the line segments
    s11, s12 = lines1[:, 0], lines1[:, 2]
    s21, s22 = lines2[:, 0], lines2[:, 2]

    # Reshape arrays for broadcasting if needed
    s11 = s11[:, np.newaxis]  # Shape (N, 1)
    s12 = s12[:, np.newaxis]  # Shape (N, 1)
    s21 = s21[np.newaxis, :]  # Shape (1, M)
    s22 = s22[np.newaxis, :]  # Shape (1, M)

    # Calculate the min and max for the ranges
    range1_min = np.minimum(s11, s12)  # Shape (N, 1)
    range1_max = np.maximum(s11, s12)  # Shape (N, 1)
    range2_min = np.minimum(s21, s22)  # Shape (1, M)
    range2_max = np.maximum(s21, s22)  # Shape (1, M)

    # Calculate the intersection of the two ranges
    lim1 = np.maximum(range1_min, range2_min)  # Shape (N, M)
    lim2 = np.minimum(range1_max, range2_max)  # Shape (N, M)

    # Stack the results along the last dimension to get shape (N, M, 2)
    result = np.stack([lim1, lim2], axis=-1)  # Shape (N, M, 2)

    # If original inputs were 1D, return a 1D result
    if single_lines:
        result = result.squeeze(axis=(0, 1))  # Shape (2,)

    return result.astype(float)


def _have_same_slope_sign(
    lines1: np.ndarray[float],  # Shape (N, 4)
    lines2: np.ndarray[float]   # Shape (M, 4)
) -> np.ndarray[bool] | bool:  # Shape (N, M)
    """Check if the slopes of two line segments are on the same sign."""
    # Calculate slopes for each line segment
    with np.errstate(divide='ignore', invalid='ignore'):
        # slope1: Shape (N, )
        slope1 = (lines1[:, 3] - lines1[:, 1]) / (lines1[:, 2] - lines1[:, 0])
        # slope2: Shape (M, )
        slope2 = (lines2[:, 3] - lines2[:, 1]) / (lines2[:, 2] - lines2[:, 0])
        slopex = np.outer(slope1, slope2)  # Shape (N, M)
    slopex = np.nan_to_num(
        slopex,
        posinf=np.nan,
        neginf=np.nan,
        nan=np.nan
    )
    slopex = (slopex >= 0)  # Note: 0/-1 = -0.0 is considered as positive
    return slopex


def compute_time_distance(
    lines1: np.ndarray,  # Shape (N, 4) or (4,)
    lines2: np.ndarray   # Shape (M, 4) or (4,)
) -> float | np.ndarray[float]:  # float or array shape (N, M)
    """Calculate the average time gap between line segments.

    Args:
        lines1 (np.ndarray): Line segments 1. Shape (N, 4) or (4,), where each
            row is defined by:
            - s1 (Union[int, float]): Start index of the line segment.
            - t1 (Union[int, float, pd.Timestamp, datetime]): Start timestamp.
            - s2 (Union[int, float]): End index of the line segment.
            - t2 (Union[int, float, pd.Timestamp, datetime]): End timestamp.
        lines2 (np.ndarray): Line segments 2. Shape (M, 4) or (4,), where each
            row is defined by:
            - s1 (Union[int, float]): Start index of the line segment.
            - t1 (Union[int, float, pd.Timestamp, datetime]): Start timestamp.
            - s2 (Union[int, float]): End index of the line segment.
            - t2 (Union[int, float, pd.Timestamp, datetime]): End timestamp.

    Returns:
        Union[float, np.ndarray[float]]: If the input is a single segment,
            returns a float. If the input is an array of segments, returns an
            array of average time gaps.
    """
    # If inputs are 1D arrays of shape (4,), reshape them to (1, 4)
    single_lines = False
    if lines1.ndim == 1 and lines2.ndim == 1:
        lines1 = np.atleast_2d(lines1)  # Shape (1, 4)
        lines2 = np.atleast_2d(lines2)  # Shape (1, 4)
        single_lines = True

    # Calculate slopes for each line segment
    with np.errstate(divide='ignore', invalid='ignore'):
        # slope1: Shape (N, )
        slope1 = (lines1[:, 3] - lines1[:, 1]) / (lines1[:, 2] - lines1[:, 0])
        # slope2: Shape (M, )
        slope2 = (lines2[:, 3] - lines2[:, 1]) / (lines2[:, 2] - lines2[:, 0])

    # Calculate the overlapping space indices limits
    overlap_lim = _compute_space_overlap(lines1, lines2)  # Shape (N, M, 2)

    # Extract coordinates for lines1
    x1_1 = lines1[:, 0].reshape(-1, 1, 1)  # Shape (N, 1, 1)
    y1_1 = lines1[:, 1].reshape(-1, 1, 1)  # Shape (N, 1, 1)

    # Extract coordinates for lines2
    x1_2 = lines2[:, 0].reshape(1, -1, 1)  # Shape (1, M, 1)
    y1_2 = lines2[:, 1].reshape(1, -1, 1)  # Shape (1, M, 1)

    # Calculate the y values for each line segment at the overlapping space idx
    with np.errstate(invalid='ignore'):  # Ignore division by zero
        # Calculate y values for lines1 at each x position
        y_values_line1 = (  # Shape (N, M, L)
            slope1[:, np.newaxis, np.newaxis] * (overlap_lim - x1_1) + y1_1)
        y_values_line2 = (  # Shape (N, M, L)
            slope2[np.newaxis, :, np.newaxis] * (overlap_lim - x1_2) + y1_2)

    # Compute the absolute difference
    abs_diff = np.abs(y_values_line1 - y_values_line2)  # Shape (N, M, L)

    # Compute the average along the L dimension
    avg_abs_diff = np.mean(abs_diff, axis=-1)  # Shape (N, M)

    # Check the invalid_lim for overlap_lim and update the avg_abs_diff
    # This means 2 line segments do not overlap in space
    invalid_lim = overlap_lim[:, :, 0] > overlap_lim[:, :, -1]
    avg_abs_diff[invalid_lim] = np.inf

    # Fix the case where the slopes are not the same sign
    slopex = _have_same_slope_sign(lines1, lines2)
    avg_abs_diff[np.logical_not(slopex)] = np.finfo(np.float64).max
    avg_abs_diff = np.nan_to_num(
        avg_abs_diff,
        nan=np.finfo(np.float64).max,
        posinf=np.finfo(np.float64).max,
        neginf=np.finfo(np.float64).max
    )

    # If original inputs were 1D, return a float
    if single_lines:
        return avg_abs_diff[0, 0]

    return avg_abs_diff


def _square_matrix_to_condensed(distance_matrix: np.ndarray) -> np.ndarray:
    """Convert a square distance matrix to a condensed format (upper
    triangular).

    Args:
        distance_matrix (np.ndarray): Square distance matrix.

    Returns:
        np.ndarray: Condensed distance matrix.
    """
    return distance_matrix[np.triu_indices(len(distance_matrix), k=1)]


def perform_single_linkage_clustering(
    distance_matrix: np.ndarray,
    epsilon: float
) -> np.ndarray:
    """Perform Single-Linkage Clustering with an epsilon threshold.

    Args:
        distance_matrix (np.ndarray): Precomputed NxN pairwise distance matrix.
        epsilon (float): Distance threshold for forming clusters.

    Returns:
        np.ndarray: Cluster labels for each data point.
    """
    # Step 1: Convert the distance matrix into a condensed format
    # Required by `linkage` (since it uses a triangular array representation)
    condensed_dist_matrix = _square_matrix_to_condensed(distance_matrix)

    # Step 2: Perform single-linkage clustering
    linkage_matrix = linkage(condensed_dist_matrix, method='single')

    # Step 3: Extract flat clusters using the epsilon threshold
    cluster_labels = fcluster(linkage_matrix, t=epsilon, criterion='distance')

    return cluster_labels


def compute_first_endpoints_manhattan_distances(
    lines: np.ndarray
) -> np.ndarray:
    """Calculate pairwise Manhattan x, y distances of the first endpoints of N
    line segments.

    Args:
        data (np.ndarray): Array of line segments. Shape (N, 4) where N is the
            number of line segments. Each line segment is defined by:
            - x1 (float): x-coordinate of the first endpoint.
            - y1 (float): y-coordinate of the first endpoint.
            - x2 (float): x-coordinate of the second endpoint.
            - y2 (float): y-coordinate of the second endpoint.

    Returns:
        np.ndarray: Array of pairwise Manhattan x, y distances, shape (N, N, 2)
        where N is the number of line segments.
        - result[..., 0] contains the distances in the  x-dimension.
        - result[..., 1] contains the distances in the  y-dimension.

    """
    # Assuming the input array is called `lines` with shape (N, 4)
    # Extract x1 and y1
    x1 = lines[:, 0]  # Shape (N,)
    y1 = lines[:, 1]  # Shape (N,)

    # Compute pairwise differences for x and y using broadcasting
    x_distance = np.abs(x1[:, np.newaxis] - x1[np.newaxis, :])  # Shape (N, N)
    y_distance = np.abs(y1[:, np.newaxis] - y1[np.newaxis, :])  # Shape (N, N)

    # Stack the results into a single array with shape (N, N, 2)
    distance = np.stack((x_distance, y_distance), axis=-1)

    # Make the distance infinite if they have the same slope
    # (not 2 sides of hyperbola)
    same_slope_matrix = _have_same_slope_sign(lines, lines)
    distance[same_slope_matrix] = np.inf

    return distance


def compute_harmonic_distance(
    distance: np.ndarray,
    dt: float,
    dn: float,
    dxn: float,
    target_velocity: float,
) -> np.ndarray:
    """Calculate the harmonic distance between each pair of points in the
    space dimension. The harmonic distance is a scaled distance that accounts
    for the desired velocity to be tracked. The distance is scaled such that
    the desired velocity is represented as a 45-degree angle in the space-time
    domain.

    Args:
        distance (np.ndarray): Array of pairwise Manhattan distances. Shape
            (N, N, 2) where N is the number of line segments.
        dt (float): Temporal sampling period.
        dn (float): Spatial channel sampling period.
        dxn (float): Channel spacing, in meters.
        target_velocity (float): Desired velocity to be tracked, in m/s.

    Returns:
        np.ndarray: Array of harmonic distances. Shape (N, N).
    """
    # Adjust the scale of the points in the space dimension
    # For example, if the target_velocity is 1500 m/s. We want to scale so
    # 1500 m is equivalent to 1 second (to make the velocity is 45-degree).
    # The corresponding number of pixels in the space dimension is 1500/ds,
    # and the corresponding number of pixels in the time dimension is 1/dt.
    # The scaling factor is (1/dt) / (1500/ds). We also need to adjust the
    # velocity from m/s to channnels/s. So the scaling factor is:
    # (1/dt) / (1500/dx/ds).

    x = distance[..., 0]  # x distances
    y = distance[..., 1]  # y distances

    x_scaled = x * (1 / dt) / (target_velocity / dn / dxn)

    # Calculate the distances
    distance = np.sqrt(x_scaled ** 2 + y ** 2)

    return distance


def group_pairs(
    distances: np.ndarray,
    threshold: float
) -> np.ndarray:
    """Assign cluster numbers to line segments based on the pairwise distance
    of the N first endpoints. This is an approximate solution. This method may
    not always produce the globally optimal grouping in terms of total minimal
    distances, as it focuses on mutual nearest neighbors. Nevertheless, it is
    more than sufficient for most practical applications and is significantly
    more efficient than exhaustive search methods by using vectorized
    operations.

    Args:
        distances (np.ndarray): A (N, N) array where distances[i, j]
            represents the distance between (the first endpoints of) the line
            segments i and j.
        threshold (float): The distance threshold below which pairs are
            eligible for grouping.

    Returns:
        np.ndarray: A 1D array of length N where each element represents the
            cluster number assigned to the corresponding line segment. Each
            line is assigned to exactly one cluster. If a line is not grouped
            with any other line, it forms its own cluster. Shape: (N,).
    """
    N = distances.shape[0]

    # Step 1: Mask distances to exclude ineligible pairs
    # Replace ineligible entries with np.inf
    mask = (distances < threshold) & (~np.eye(N, dtype=bool))
    distances_masked = np.where(mask, distances, np.inf)

    # Step 2: For each line i, find the index j with minimal sum_distance
    min_indices = np.argmin(distances_masked, axis=1)  # Shape: (N,)

    # Step 3: Identify mutual pairs where min_indices[min_indices[i]] == i
    # This means i and min_indices[i] are mutual nearest neighbors
    mutual_pairs_mask = (min_indices[min_indices] == np.arange(N))  # (N,)
    mutual_indices = np.where(mutual_pairs_mask)[0]

    # To avoid duplicates, consider pairs where i < min_indices[i]ˆ
    i_mask = mutual_indices < min_indices[mutual_indices]
    i_indices = mutual_indices[i_mask]
    j_indices = min_indices[i_indices]

    # Step 4: Assign cluster numbers to mutual pairs
    cluster_assignments = -1 * np.ones(N, dtype=int)
    cluster_nums = np.arange(len(i_indices))
    cluster_assignments[i_indices] = cluster_nums
    cluster_assignments[j_indices] = cluster_nums

    # Step 5: Assign unique cluster numbers to ungrouped lines
    ungrouped_indices = np.where(cluster_assignments == -1)[0]
    # Assign each ungrouped line a unique cluster number starting from
    # len(i_indices)
    cluster_assignments[ungrouped_indices] = (
        np.arange(len(i_indices), len(i_indices) + len(ungrouped_indices)))

    return cluster_assignments


def infer_lines_info(
    lines_coords: np.ndarray[int],
    dn: float,
    dt: float,
    dxn: float,
    timestamps: np.ndarray[float],
    channels: np.ndarray[float]
) -> pd.DataFrame:
    """Infer information (channels, times, velocity, ...) from the coordinates.

    Args:
        lines_coords (np.ndarray[int]): Coordinates of the detected lines,
            resulted from Hough transform, shape (N, 4).
        dn (float): Spatial channnel sampling period.
        dt (float): Temporal sampling period.
        dxn (float): Channel spacing, in meters.
        timestamps (np.ndarray[float]): Timestamps of the data.
        channels (np.ndarray[float]): Spatial channels of the data.

    Returns:
        pd.DataFrame: DataFrame containing the inferred information.
    """
    # Convert the coordinates to spatial and temporal distances
    lines = lines_coords.copy()
    lines = lines.astype(float)  # Convert to float for calculations
    lines[:, [0, 2]] *= dn  # Convert spatial index to channel distance
    lines[:, [0, 2]] += channels[0]  # Convert spatial idx to channels
    lines[:, [1, 3]] *= dt  # Convert temporal index to temporal distance
    lines[:, [1, 3]] += timestamps[0]  # Convert temporal index to timestamps

    # Calculate the velocity and temporal at the first and last channel
    with np.errstate(divide='ignore', invalid='ignore'):
        # 0/0 = nan, 0/1 = 0, 0/-1 = -0, 1/0 = inf, -1/0 = -inf
        slope = (lines[:, 3] - lines[:, 1]) / (lines[:, 2] - lines[:, 0])
        # Formular of a straight line: y = slope * x + y_0
        y_0 = lines[:, 1] - slope * lines[:, 0]  # temporal at channel 0
        y_first = slope * channels[0] + y_0  # temporal at first channel
        y_last = slope * channels[-1] + y_0  # temporal at last channel
        velocity = (1 / slope) * dxn

    # Swap y_first and y_last to ensure t1_edge <= t2_edge
    t1_edge = np.minimum(y_first, y_last)
    t2_edge = np.maximum(y_first, y_last)

    s = np.abs(lines[:, 2] - lines[:, 0])
    t = lines[:, 3] - lines[:, 1]

    # Create a dictionary of the inferred information
    lines = {
        'x1': lines_coords[:, 0],
        'y1': lines_coords[:, 1],
        'x2': lines_coords[:, 2],
        'y2': lines_coords[:, 3],
        'x1n': lines_coords[:, 0] / len(channels),
        'y1n': lines_coords[:, 1] / len(timestamps),
        'x2n': lines_coords[:, 2] / len(channels),
        'y2n': lines_coords[:, 3] / len(timestamps),
        'n1': lines[:, 0],
        't1': lines[:, 1],
        'n2': lines[:, 2],
        't2': lines[:, 3],
        't1_edge': t1_edge,
        't2_edge': t2_edge,
        's': s,
        't': t,
        'velocity': velocity
    }

    # Create the DataFrame
    lines = pd.DataFrame(lines)
    return lines


def filter_lines_by_velocity(
    lines: pd.DataFrame,
    velocity_low: float,
    velocity_high: float,
    velocity_col: str = 'velocity',
) -> pd.DataFrame:
    """Keep the lines that have velocity within the specified range.

    Args:
        lines (pd.DataFrame): DataFrame containing the lines information.
        velocity_low (float): Lower bound of the velocity range.
        velocity_high (float): Upper bound of the velocity range.
        velocity_col (str, optional): Name of the column containing the
            velocity information. Defaults to 'velocity'.

    Returns:
        pd.DataFrame: DataFrame containing the lines that have velocity within
            the specified range.
    """
    lines_keep = (
        lines
        .loc[lambda df:
             df[velocity_col].abs().between(velocity_low, velocity_high)]
    )
    return lines_keep


def aggregate_line_clusters(
    lines: pd.DataFrame,
    line_clusters: np.ndarray[int],
    trim_prop: float,
) -> pd.DataFrame:
    """Aggregate the line segments by the cluster.

    Args:
        lines (pd.DataFrame): Line segments data. Must contain columns:
            x1, y1, x2, y2, velocity.
        line_clusters (np.ndarray[int]): Cluster labels for each line segment.
        trim_prop (float): Proportion to trim to calculate the mean and std.

    Returns:
        pd.DataFrame: Aggregated line segments data.
            The data frame will contain the following columns:
            - x1: The mean x coordinate of the start point.
            - y1: The mean y coordinate of the start point.
            - x2: The mean x coordinate of the end point.
            - y2: The mean y coordinate of the end point.
            - velocity: The mean velocity of the line segments.
            - velocity_std: The standard deviation of the velocity.
            - velocity_min: The minimum velocity.
            - velocity_max: The maximum velocity.
            - num_lines: The number of line segments in the cluster.
    """
    # Aggregate the line segments by the cluster
    def agg_func_mean(x):
        """Calculate the trimmed mean of the velocity.
        """
        return trim_mean(x, proportiontocut=trim_prop)

    def agg_func_std(x):
        """Calculate the trimmed standard deviation of the velocity.
        """
        y = trimmed_std(x, limits=(trim_prop, trim_prop))
        return np.nan if isinstance(y, np.ma.core.MaskedConstant) else y

    lines_agg = (
        lines
        .assign(cluster=line_clusters)
        .groupby('cluster')
        .agg(
            x1=('x1', agg_func_mean),
            y1=('y1', agg_func_mean),
            x2=('x2', agg_func_mean),
            y2=('y2', agg_func_mean),
            velocity=('velocity', agg_func_mean),
            velocity_std=('velocity', agg_func_std),
            velocity_min=('velocity', 'min'),
            velocity_max=('velocity', 'max'),
            num_lines=('x1', 'size'),
        )
        .round({
            'x1': 0, 'y1': 0, 'x2': 0, 'y2': 0,
            'velocity': 1,
            'velocity_std': 1,
            'velocity_min': 1,
            'velocity_max': 1
        })
        .reset_index(drop=True)
    )
    return lines_agg


def build_boxes_from_lines(
    lines: pd.DataFrame,
    timestamps: np.ndarray,
    channels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Build boxes from the lines.

    Args:
        lines (pd.DataFrame): DataFrame containing the lines information.
        timestamps (np.ndarray): Timestamps of the data.
        channels (np.ndarray): Spatial channels of the data.

    Returns:
        tuple[np.ndarray, np.ndarray]: Tuple containing two arrays:
            - boxesn: Normalized boxes, shape (N, 4).
            - boxesp: Boxes in physical units, shape (N, 4).
        If there are no lines, return empty arrays.
    """
    if len(lines) == 0:
        # If there are no lines, return empty arrays
        return np.empty((0, 4)), np.empty((0, 4), dtype=object)

    if 'pair_group' not in lines.columns:
        # If the lines are not grouped, create a new column for grouping
        lines['pair_group'] = np.arange(len(lines))

    group_cols = ['x1n', 'y1n', 'x2n', 'y2n']
    boxesn = lines.groupby('pair_group')[group_cols].apply(
        lambda group: [
            min(group['x1n'].min(), group['x2n'].min()),
            min(group['y1n'].min(), group['y2n'].min()),
            max(group['x1n'].max(), group['x2n'].max()),
            max(group['y1n'].max(), group['y2n'].max())
        ]
    )
    boxesn = np.stack(boxesn.to_numpy())

    boxesd = box_saver.denormalize_boxesn(
        boxesn=boxesn,
        t_start=timestamps[0],
        t_end=timestamps[-1],
        n_start=channels[0],
        n_end=channels[-1],
    )

    boxesp = box_saver.cast_box_times_to_datetime64(boxes=boxesd)

    # Convert to NumPy array
    boxesn = np.array(boxesn)
    boxesp = np.array(boxesp)

    return boxesn, boxesp


class DASHoughLines:
    """Hough transform DAS data."""

    def hough_lines(
        self,
        rho: float,
        theta: float | Callable,
        threshold: float | Callable,
        minLineLength: float | Callable,
        maxLineGap: float | Callable,
    ) -> 'DASArray':
        """Apply Hough transform to detect line segments in the DAS data.

        Args:
            rho (float): Distance resolution of the accumulator in pixels.
            theta (float | Callable): Angle resolution of the accumulator in
                radians. If callable, it should return the float.
            threshold (float | Callable): Accumulator threshold parameter. Only
                those lines are returned that get enough votes ( > threshold).
                If callable, it should return the float.
            minLineLength (float | Callable): Minimum line length. Line
                segments shorter than this are rejected. If callable, it should
                return the float.
            maxLineGap (float | Callable): Maximum allowed gap between line
                segments to treat them as single line. If callable, it should
                return the float.
        """
        # Get the Hough transform parameters
        #######################################################################
        if callable(theta):
            theta = theta(self)
        if callable(threshold):
            threshold = threshold(self)
        if callable(minLineLength):
            minLineLength = minLineLength(self)
        if callable(maxLineGap):
            maxLineGap = maxLineGap(self)

        # Apply Hough transform to detect line segments
        #######################################################################
        lines_coords = hough_lines(
            data=self,
            rho=rho,
            theta=theta,
            threshold=threshold,
            minLineLength=minLineLength,
            maxLineGap=maxLineGap
        )

        # Infer information (channels, times, velocity, ...) from the coords.
        #######################################################################
        lines = infer_lines_info(
            lines_coords=lines_coords,
            ds=self.meta.dn,
            dt=self.meta.dt,
            dx=self.meta.dxn,
            timestamps=self.meta.timestamps,
            channels=self.meta.channels
        )

        # Update the meta information
        self.meta.update(lines=lines)
        return self

    def filter_lines_by_velocity(
        self,
        velocity_low: float,
        velocity_high: float
    ) -> 'DASArray':
        """Keep the lines that have velocity within the specified range.

        Args:
            velocity_low (float): Lower bound of the velocity range.
            velocity_high (float): Upper bound of the velocity range.
        """
        lines_keep = filter_lines_by_velocity(
            lines=self.meta.lines,
            velocity_low=velocity_low,
            velocity_high=velocity_high
        )
        lines_keep = lines_keep.reset_index(drop=True)
        self.meta.update(lines=lines_keep)
        return self

    def perform_single_linkage_clustering(
        self,
        epsilon: float,
        trim_prop: float = 0.2,
    ) -> 'DASArray':
        """Cluster the line segments using Single-Linkage Clustering.

        Args:
            epsilon (float): Distance threshold for forming clusters.
            trim_prop (float, optional): Proportion to trim to calculate the
                mean and std. Defaults to 0.2.
        """
        if len(self.meta.lines) == 0:
            logger.info("No line segments to cluster.")
            # Add 3 new empty columns (to keep the same number of columns)
            self.meta.lines[[
                'velocity_std',
                'velocity_min',
                'velocity_max',
                'num_lines'
            ]] = None
            return self

        if len(self.meta.lines) == 1:
            # If there is only one line segment, assign it to cluster 0.
            # Because performing single_linkage_clustering on a single point
            # will raise an error when there is only one point in the dataset.
            line_clusters = np.array([0])
        else:
            # Calculate the distance matrix between line segments
            ###################################################################
            distance_matrix = compute_time_distance(
                lines1=self.meta.lines[['x1', 'y1', 'x2', 'y2']].to_numpy(),
                lines2=self.meta.lines[['x1', 'y1', 'x2', 'y2']].to_numpy()
            )

            # Perform Single-Linkage Clustering with an epsilon threshold
            ###################################################################
            line_clusters = perform_single_linkage_clustering(
                distance_matrix=distance_matrix,
                epsilon=epsilon
            )

        # Aggregate the line segments by the cluster
        #######################################################################
        lines_agg = aggregate_line_clusters(
            lines=self.meta.lines[['x1', 'y1', 'x2', 'y2', 'velocity']],
            line_clusters=line_clusters,
            trim_prop=trim_prop
        )

        lines_info = infer_lines_info(
            lines_coords=lines_agg[['x1', 'y1', 'x2', 'y2']].to_numpy(),
            ds=self.meta.dn,
            dt=self.meta.dt,
            dx=self.meta.dxn,
            timestamps=self.meta.timestamps,
            channels=self.meta.channels
        )

        lines = pd.concat([
            lines_info,
            lines_agg[['velocity_std',
                       'velocity_min',
                       'velocity_max',
                       'num_lines']],
        ], axis=1)

        self.meta.update(lines=lines)
        return self

    def group_lines_by_pairs(
        self,
        threshold: float,
        target_velocity: float
    ) -> 'DASArray':
        """Group the line segments into pairs.

        Args:
            threshold (float): The distance threshold below which pairs are
                eligible for grouping.
            target_velocity (float): Desired velocity to be tracked, in m/s.
                This is used to harmonize the distance between the line
                segments, such that the desired velocity is represented as a
                45-degree angle in the space-time domain.
        """
        if len(self.meta.lines) == 0:
            logger.info("No line segments to groups.")
            # Add new empty columns (to keep the same number of columns)
            self.meta.lines[['pair_group']] = None
            return self

        # Calculate the distance matrix between line segments
        #######################################################################
        xy_first_endpoints_distance_matrix = (
            compute_first_endpoints_manhattan_distances(
                lines=self.meta.lines[['x1', 'y1', 'x2', 'y2']].to_numpy()
            )
        )

        first_endpoints_distance_matrix = compute_harmonic_distance(
            distance=xy_first_endpoints_distance_matrix,
            dt=self.meta.dt,
            ds=self.meta.dn,
            dx=self.meta.dxn,
            target_velocity=target_velocity
        )

        # Group the line segments into pairs
        #######################################################################
        pair_groups = group_pairs(
            distances=first_endpoints_distance_matrix,
            threshold=threshold
        )
        self.meta.lines['pair_group'] = pair_groups
        return self

    def build_boxes_from_lines(self) -> 'DASArray':
        """Build boxes from the lines.

        Returns:
            DASArray: The DASArray object with updated meta information.
        """
        # Build boxes from the lines
        boxesn, boxesp = build_boxes_from_lines(
            lines=self.meta.lines,
            timestamps=self.meta.timestamps,
            channels=self.meta.channels
        )

        # Update the meta information
        self.meta.update(boxesn=boxesn, boxesp=boxesp)
        return self
