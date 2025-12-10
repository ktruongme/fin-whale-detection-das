"""This module provides functions for fitting hyperbolas to binary 2D arrays
derived from DAS (Distributed Acoustic Sensing) data. It includes methods
for fitting hyperbolas using least squares and optimization techniques,
as well as utilities for error calculation and hyperbola metrics derivation.
"""
from typing import Callable, Tuple, Optional, List, TYPE_CHECKING
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy.optimize import least_squares

from ..filters.box_extract import (
    slice_by_normalized_coords,
    extract_binary_coords
)

if TYPE_CHECKING:
    from ..core.dasarray import DASArray

# Constants for fallback values
FALLBACK_A_GUESS = 10
FALLBACK_B_GUESS = 10
FALLBACK_H_GUESS = 0
FALLBACK_K_GUESS = 0


def _hyperbola_model(
    x: np.ndarray,
    a: float,
    b: float,
    h: float,
    k: float
) -> np.ndarray:
    """Hyperbola model function used for curve fitting (Upper Branch / Smile).

    The analytic form follows:
        (y - k)^2 / b^2 - (x - h)^2 / a^2 = 1

    Resolved for y (Upper Branch, positive root):
        y = k + b * sqrt(...)

    Args:
        x (np.ndarray): Input x-values.
        a (float): Parameter a of the hyperbola.
        b (float): Parameter b of the hyperbola.
        h (float): Parameter h of the hyperbola (horizontal shift).
        k (float): Parameter k of the hyperbola (vertical shift).

    Returns:
        np.ndarray: Predicted y-values for input x based on parameters a, b, h,
            k.
    """
    return k + b * np.sqrt(1 + ((x - h) ** 2) / a**2)


def _default_initial_guess(
    binary_array: np.ndarray
) -> Tuple[float, float, float, float]:
    """Generate default initial guess parameters for hyperbola fitting.

    Args:
        binary_array (np.ndarray): Binary 2D array.

    Returns:
        Tuple[float, float, float, float]: Initial guess parameters (a, b, h,
            k).
    """
    x_coords, y_coords = extract_binary_coords(binary_array)

    if x_coords.size == 0:
        return (FALLBACK_A_GUESS, FALLBACK_B_GUESS, 
                FALLBACK_H_GUESS, FALLBACK_K_GUESS)

    # 1. h (Horizontal Center): Median of X is usually safe
    h_guess = np.median(x_coords)

    # 2. b (Vertical Scale):
    # We estimate b by looking at how "steep" the curve is.
    # A safe heuristic is the distance from the vertex to the bottom of the box
    # (Or simpler: just set a sensible default like 10% of the window height)
    b_guess = (np.max(y_coords) - np.min(y_coords)) / 2.0
    if b_guess <= 0:
        b_guess = FALLBACK_B_GUESS

    # 3. k (Vertical Center):
    # precise_vertex = k + b  ->  k = precise_vertex - b
    # precise_vertex is roughly min(y_coords)
    k_guess = np.min(y_coords) - b_guess

    # 4. a (Horizontal Scale):
    a_guess = (np.max(x_coords) - np.min(x_coords)) / 2.0
    if a_guess <= 0:
        a_guess = FALLBACK_A_GUESS

    return a_guess, b_guess, h_guess, k_guess


def _calculate_errors(
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    params: Tuple[float, float, float, float],
    array_shape: Tuple[int, int]
) -> Tuple[float, float]:
    """Calculate the RMSE and MAE errors for the hyperbola fitting.

    Args:
        x_coords (np.ndarray): x-coordinates of the binary array.
        y_coords (np.ndarray): y-coordinates of the binary array.
        params (Tuple): Fitted hyperbola parameters (a, b, h, k).
        array_shape (Tuple): Shape of the binary array.

    Returns:
        Tuple[float, float]: Normalized RMSE and MAE errors.
    """
    a, b, h, k = params
    y_pred = _hyperbola_model(x_coords, a, b, h, k)

    rmse = np.sqrt(np.mean((y_coords - y_pred) ** 2))
    mae = np.mean(np.abs(y_coords - y_pred))

    # Normalize errors by the height of bounding box
    rmse_norm = rmse / array_shape[0]
    mae_norm = mae / array_shape[0]

    return rmse_norm, mae_norm


@dataclass
class HyperbolaFitResult:
    """Holds the result of the hyperbola fitting process.

    Attributes:
        params (Tuple[float, float, float, float]): Parameters: a, b, h, k.
        errors (Tuple[float, float]): Normalized RMSE and MAE errors.
        physical_params (Optional[HyperbolaPhysicalParams]): Derived physical
            parameters if metadata is available.
    """
    params: Tuple[float, float, float, float]
    errors: Tuple[float, float]
    physical_params: Optional[object] = None


def fit_hyperbola_least_squares(
    binary_array: np.ndarray,
    maxiter: int = 10_000,
    initial_guess: Optional[Tuple[float, float, float, float]] = None,
) -> HyperbolaFitResult:
    """Fit a hyperbola to a binary 2D array using `least_squares`. This method
    uses the least squares optimization algorithm to fit the hyperbola to the
    binary array. If wish to use a different optimization method, consider
    using `fit_hyperbola_minimize`.

    Args:
        binary_array (np.ndarray): Binary 2D array.
        maxiter (int): Maximum iterations for optimization. Defaults to 10_000.
        initial_guess (Optional[Tuple[float, float, float, float]]): Initial
            guess for the hyperbola parameters. Defaults to None.

    Returns:
        HyperbolaFitResult: Object containing the fitted parameters and the
            errors.
    """
    x_coords, y_coords = extract_binary_coords(binary_array)
    initial_guess = initial_guess or _default_initial_guess(binary_array)

    def residuals(params):
        return _hyperbola_model(x_coords, *params) - y_coords

    result = least_squares(
        fun=residuals,
        x0=initial_guess,
        max_nfev=maxiter
    )

    params = tuple(result.x)
    errors = _calculate_errors(x_coords, y_coords, params, binary_array.shape)

    return HyperbolaFitResult(
        params=params,
        errors=errors,
    )


def _slice_and_transform(
    array: np.ndarray,
    box: Tuple[float, float, float, float],
    num_points: int,
    by_channel: bool
) -> np.ndarray:
    """Slice array by normalized coordinates and binary transform.

    Args:
        array (DASArray): Input 2D array.
        box (Tuple[float, float, float, float]): Normalized coordinates of the
            box.
        num_points (int): Number of points to retain in binary transformation.
        by_channel (bool): If True, the binary transformation is done by
            channel.

    Returns:
        DASArray: Binary transformed array.
    """
    sliced_array = slice_by_normalized_coords(array=array, coords=box)
    return sliced_array.binary_transform(
        num_points=num_points,
        by_channel=by_channel
    )


def fit_multiple_hyperbolas_least_squares(
    array: np.ndarray,
    boxesn: np.ndarray,
    num_points: int = 10,
    by_channel: bool = True,
    maxiter: int = 10_000,
    initial_guess: Optional[Tuple[float, float, float, float]] = None,
    max_workers: int = 4
) -> List[HyperbolaFitResult]:
    """Fit hyperbolas to the binary arrays obtained from the normalized
    coordinates.

    Args:
        array (np.ndarray): Input 2D array.
        boxesn (np.ndarray): Normalized coordinates of the boxes.
        num_points (int): Number of points to retain in binary transformation.
            Defaults to 10.
        by_channel (bool): If True, the binary transformation is done by
            channel. Defaults to True.
        maxiter (int): Maximum iterations for curve fitting. Defaults to
            10_000.
        initial_guess (Optional[Tuple]): Initial guess for the hyperbola
            parameters. Defaults to None.
        max_workers (int): Number of workers for parallel processing. Though
            experiment in workstation, 4 workers are recommended for
            performance. Defaults to 4.

    Returns:
        List[HyperbolaFitResult]: List of HyperbolaFitResult objects.
    """
    def _fit_single_box(box):
        binary_array = _slice_and_transform(array, box, num_points, by_channel)
        return fit_hyperbola_least_squares(
            binary_array=binary_array,
            maxiter=maxiter,
            initial_guess=initial_guess
        )

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        hyperpolas = list(executor.map(_fit_single_box, boxesn))

    return hyperpolas


def derive_hyperbola_metrics(
    hyperbolas: List[HyperbolaFitResult],
    ds: float,
    dx: float
) -> dict:
    """Calculate source distances and residuals from a list of hyperbolas.

    Args:
        hyperbolas (List[HyperbolaFitResult]): List of HyperbolaFitResult
            objects.
        ds (float): Spatial sampling period in channels.
        dx (float): Channel spacing in meters.

    Returns:
        dict: Dictionary containing source distances, RMSE norms, and MAE
            norms.
    """
    source_distances = []
    rmse_norms = []
    mae_norms = []

    for hyperbola in hyperbolas:
        a, _, _, _ = hyperbola.params
        rmse_norm, mae_norm = hyperbola.errors
        source_distance = np.abs(a) * ds * dx

        source_distances.append(source_distance)
        rmse_norms.append(rmse_norm)
        mae_norms.append(mae_norm)

    return {
        'source_distance': np.array(source_distances),
        'hyper_rmse_norm': np.array(rmse_norms),
        'hyper_mae_norm': np.array(mae_norms),
    }
