"""This module provides functions to filter DAS data using band-pass, low-pass,
and high-pass filters. It also includes methods to transform the data to
binary and grayscale representations, apply the Sobel operator to detect edges,
and compute the Fast Fourier Transform (FFT) of the signal.

Example:
    filtered_data = bandpass_filter(data, sampling_rate, low_freq, high_freq)
    filtered_data = lowpass_filter(data, sampling_rate, cutoff_freq)
    filtered_data = highpass_filter(data, sampling_rate, cutoff_freq)
    binary_data = binary_transform(data, quantile=0.95)
    grayscale_data = grayscale_transform(data)
    grayscale_data = grayscale_transform_cv2(data)
    sobel_data = sobel_filter(data)
    frequencies, agg_spectrum = fft(data, t_rate)
"""

# Set environment for Numba threading and OpenMP nesting before any other
# imports
import os
from typing import TYPE_CHECKING
import copy  # noqa: E402

import numpy as np  # noqa: E402
from scipy.fft import (fft, ifft, fftshift, ifftshift, fftfreq,  # noqa: E402
                       rfft, irfft, rfftfreq)
import cv2  # noqa: E402
from numba import njit, prange  # noqa: E402

os.environ.setdefault('NUMBA_THREADING_LAYER', 'workqueue')
os.environ.setdefault('OMP_MAX_ACTIVE_LEVELS', '1')

if TYPE_CHECKING:
    from ..core.dasarray import DASArray


def binary_transform(
    data: np.ndarray,
    quantile: float = None,
    threshold: float = None,
    num_points: int = None,
    by_channel: bool = False,
    by_time: bool = False
) -> np.ndarray:
    """
    Transform the signal attribute to a binary representation.

    Args:
        data (np.ndarray): Input data array.
        quantile (float, optional): Quantile to compute threshold if no
            threshold is provided. Defaults to None.
        threshold (float, optional): Fixed threshold value. If None,
            the quantile is used to compute the threshold. Defaults to None.
        num_points (int, optional): Number of points to retain. Defaults to
            None.
        by_channel (bool, optional): Compute the threshold individually
            for each column (channel). Defaults to False.
        by_time (bool, optional): Compute the threshold individually
            for each row (time). Defaults to False.

    Returns:
        np.ndarray: Binary representation of the signal.
    """
    # Ensure only one of `quantile`, `threshold`, or `num_points` is specified
    provided_params = [quantile, threshold, num_points]
    if sum(param is not None for param in provided_params) > 1:
        raise ValueError("Specify only one of `quantile`, `threshold, or " +
                         "`num_points`.")

    # Ensure `by_channel` and `by_time` are not both True
    if by_channel and by_time:
        raise ValueError("Both `by_channel` and `by_time` cannot be True " +
                         "simultaneously.")

    data_abs = np.abs(data)
    # Treat NaNs as lowest values so they are never selected
    data_abs = np.nan_to_num(data_abs, nan=-np.inf)

    if num_points is not None:
        result = _handle_num_points(data_abs, num_points, by_channel, by_time)
    else:
        result = _apply_threshold(
            data_abs, quantile, threshold, by_channel, by_time)

    return result


def _handle_num_points(
    data: np.ndarray,
    num_points: int,
    by_channel: bool,
    by_time: bool
) -> np.ndarray:
    """Handle the binary transformation based on the number of points to
    retain.

    Args:
        data (np.ndarray): Absolute data array.
        num_points (int): Number of points to retain.
        by_channel (bool): Whether to retain points by channel.
        by_time (bool): Whether to retain points by time.

    Returns:
        np.ndarray: Binary representation of the data.
    """
    if by_channel:
        return _retain_top_n_by_axis(data, num_points, axis=0)
    elif by_time:
        return _retain_top_n_by_axis(data, num_points, axis=1)
    else:
        flat_indices = np.argpartition(
            data.flatten(), -num_points)[-num_points:]
        binary_array = np.zeros(data.size, dtype=np.uint8)
        binary_array[flat_indices] = 1
        return binary_array.reshape(data.shape)


def _retain_top_n_by_axis(
    data: np.ndarray,
    num_points: int,
    axis: int
) -> np.ndarray:
    """
    Retain the top N points along the specified axis.

    Args:
        data (np.ndarray): Absolute data array.
        num_points (int): Number of points to retain.
        axis (int): Axis along which to retain points.

    Returns:
        np.ndarray: Binary representation of the data.
    """
    partitioned_indices = (
        np.argpartition(-data, num_points, axis=axis)
        .take(indices=range(num_points), axis=axis))
    mask = np.zeros_like(data, dtype=bool)

    if axis == 0:  # Retaining by columns
        mask[partitioned_indices, np.arange(data.shape[1])] = True
    elif axis == 1:  # Retaining by rows
        mask[np.arange(data.shape[0])[:, None], partitioned_indices] = True

    return mask.astype(np.uint8)


def _apply_threshold(
    data: np.ndarray,
    quantile: float,
    threshold: float,
    by_channel: bool,
    by_time: bool
) -> np.ndarray:
    """
    Apply a threshold to compute the binary transformation.

    Args:
        data (np.ndarray): Absolute data array.
        quantile (float): Quantile to compute threshold.
        threshold (float): Fixed threshold value.
        by_channel (bool): Compute the threshold by channel.
        by_time (bool): Compute the threshold by time.

    Returns:
        np.ndarray: Binary representation of the data.
    """
    if threshold is None:
        if by_channel:
            threshold = np.nanquantile(data, quantile, axis=0)
        elif by_time:
            threshold = np.nanquantile(data, quantile, axis=1).reshape(-1, 1)
        else:
            threshold = np.nanquantile(data, quantile)

    return (data >= threshold).astype(np.uint8)


def grayscale_transform(
    data: np.ndarray[float],
    by_column: bool = False
) -> np.ndarray[int]:
    """Transform data to grayscale (0 to 255) using min-max scaling.

    Args:
        data (np.ndarray[float]): Input 2D data.
        by_column (bool): If True, perform column-wise scaling; otherwise scale
            the entire array.

    Returns:
        np.ndarray[int]: Grayscale signal as an array of uint8.
    """
    abs_data = np.abs(data)

    if by_column:
        # Compute min and max per column (keepdims for broadcasting)
        col_min = abs_data.min(axis=0, keepdims=True)
        col_max = abs_data.max(axis=0, keepdims=True)
        # Avoid division by zero in columns with constant values
        scale = np.where(col_max - col_min == 0, 1, col_max - col_min)
        # Vectorized column-wise normalization to 0-255
        normalized = (abs_data - col_min) / scale * 255
        signal_gray = np.round(normalized).astype(np.uint8)
    else:
        # Normalize the entire array at once using cv2.normalize
        signal_gray = cv2.normalize(
            abs_data,
            None,
            alpha=0,
            beta=255,
            norm_type=cv2.NORM_MINMAX
        )
        signal_gray = np.round(signal_gray).astype(np.uint8)

    return signal_gray


def fk_filter_real(
    data: np.ndarray,
    f_min: float,
    f_max: float,
    v_min: float,
    v_max: float,
    dt: float,
    dn: float,
    dxn: float,
    num_workers: int = -1
) -> np.ndarray:
    """Apply a FK filter to the data using real FFT.

    This implementation uses scipy that leverages multi-threading for the FFT.
    This is proved via experiments to be faster than using numpy and pyfftw.

    Args:
        data (np.ndarray[float]): Input data.
        f_min (float): Minimum frequency (Hz).
        f_max (float): Maximum frequency (Hz).
        v_min (float): Minimum velocity (m/s).
        v_max (float): Maximum velocity (m/s).
        dt (float): Temporal sampling period (s).
        dn (float): Spatial sampling period (channels).
        dxn (float): Channel spacing in meters.
        num_workers (int, optional): Number of workers for parallel processing.
            If negative, the value wraps around from os.cpu_count(). Based on
            experience, the optimal number of workers is usually fewer than
            half of the available CPU cores. For example, the optimal number of
            workers for a 24-core CPU is 10. Defaults to -1.
    Returns:
        np.ndarray[float]: Filtered data.
    """
    Nt, Nx = data.shape
    if data.dtype != np.float32:
        data = data.astype(np.float32)

    # rFFT over time (real FFT → only positive freqs)
    data_fft_t = rfft(data, axis=0, workers=num_workers)
    f_axis = rfftfreq(Nt, d=dt).astype(np.float32)  # Shape (Nt//2+1,)

    # FFT in space (full spectrum)
    data_fft_tf = fft(data_fft_t, axis=1, workers=num_workers)
    k_axis = fftfreq(Nx, d=dn * dxn).astype(np.float32)  # Shape (Nx,)

    # Shift spatial axis for easier masking
    data_fft_shift = fftshift(data_fft_tf, axes=(1,))
    k_shift = fftshift(k_axis)

    absF = f_axis[:, None]        # (Nt//2+1, 1)
    absK = np.abs(k_shift)[None, :]  # (1, Nx)

    cond_f = (absF >= f_min) & (absF <= f_max)
    cond_k = (absK > 0)
    cond_v = (absF >= v_min * absK) & (absF <= v_max * absK) & cond_k

    mask = cond_f & cond_v
    data_fft_shift[~mask] = 0

    # Inverse: undo shift, then iFFT over space, iRFFT over time
    data_ifft_space = ifft(ifftshift(data_fft_shift, axes=(1,)), axis=1,
                           workers=num_workers)
    data_filtered = irfft(data_ifft_space, n=Nt, axis=0, workers=num_workers)

    return np.ascontiguousarray(data_filtered.real, dtype=np.float32)


@njit(parallel=True, fastmath=True)
def rms(data: np.ndarray, window_size: int) -> np.ndarray:
    """Compute causal RMS over time (rows) for each column (space), handling
    early rows with partial windows. This version uses Numba for acceleration,
    which is faster than the numpy version for large arrays.

    Args:

        data (np.ndarray): 2D array (time x space).
        window_size (int): Number of time steps for RMS window.

    Returns:
        np.ndarray: 2D array of RMS values (same shape as input).
    """
    t, n = data.shape
    out = np.empty((t, n), dtype=np.float32)

    for j in prange(n):  # Parallel across columns (space)
        sum_sq = 0.0
        for i in range(t):
            val = data[i, j]
            sum_sq += val * val
            if i >= window_size:
                old_val = data[i - window_size, j]
                sum_sq -= old_val * old_val
                count = window_size
            else:
                count = i + 1
            out[i, j] = (sum_sq / count) ** 0.5

    return out


class DASFilter:
    """Mixin class that provides filtering methods for DAS data."""

    def binary_transform(
        self,
        quantile: float = None,
        threshold: float = None,
        num_points: int = None,
        by_channel: bool = False,
        by_time: bool = False
    ) -> 'DASArray':
        binary_data = binary_transform(
            data=self,
            quantile=quantile,
            threshold=threshold,
            num_points=num_points,
            by_channel=by_channel,
            by_time=by_time
        )
        result = self.__class__(binary_data)
        result.meta = copy.deepcopy(self.meta)
        return result

    def grayscale_transform(self, by_column: bool = False) -> 'DASArray':
        grayscale_data = grayscale_transform(data=self, by_column=by_column)
        result = self.__class__(grayscale_data)
        result.meta = copy.deepcopy(self.meta)
        return result

    def fk_filter_real(
        self,
        f_min: float,
        f_max: float,
        v_min: float,
        v_max: float,
        num_workers: int = 10
    ) -> 'DASArray':
        filtered_data = fk_filter_real(
            data=self,
            f_min=f_min,
            f_max=f_max,
            v_min=v_min,
            v_max=v_max,
            dt=self.meta.dt,
            dn=self.meta.dn,
            dxn=self.meta.dxn,
            num_workers=num_workers
        )
        result = self.__class__(filtered_data)
        result.meta = copy.deepcopy(self.meta)
        return result

    def rms(
        self,
        window_size_second: int
    ) -> 'DASArray':
        window_size = int(window_size_second / self.meta.dt)
        rms_data = rms(data=self, window_size=window_size)
        result = self.__class__(rms_data)
        result.meta = copy.deepcopy(self.meta)
        return result
