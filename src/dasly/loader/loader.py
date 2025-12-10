"""This module contains functions to load DAS data from HDF5 files.

Functions:
    load_hdf5: Load DAS data from HDF5 files.
"""

from datetime import datetime, timezone
from typing import TYPE_CHECKING
import math

import numpy as np
import pandas as pd
from scipy.stats import mode
import h5py

from .fsearcher import (
    _parse_datetime_str,
    get_hdf5_file_paths_range,
    get_hdf5_header,
    get_all_hdf5_file_paths
)

if TYPE_CHECKING:
    from ..core.dasarray import DASArray


def _scale(data: np.ndarray, scale_factor: float) -> np.ndarray:
    """Scale data by a given factor.

    Args:
        data (np.ndarray): Data to scale.
        scale_factor (float): Factor to scale data by.

    Returns:
        np.ndarray: Scaled data.
    """
    return data * scale_factor


def _integrate(data: np.ndarray, dt: float) -> np.ndarray:
    """Integrate data to get strain.

    Args:
        data (np.ndarray): Data to integrate.
        dt (float): Temporal period (time step).

    Returns:
        np.ndarray: Strain data.
    """
    return np.cumsum(data, axis=0) * dt


def _format_time_string(timestamp: float) -> str:
    """Format a timestamp into 'YYYYMMDD HHMMSS[.ffffff]'."""
    dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)
    base = dt.strftime('%Y%m%d %H%M%S')
    if dt.microsecond:
        fraction = f"{dt.microsecond:06d}".rstrip('0')
        return f"{base}.{fraction}"
    return base


def _infer_time(
    start: str = None,
    duration: float = None,
    end: str = None
) -> tuple[str, float, str]:
    """Infer start, end or duration from the other two of them.

    Infer start if duration and end are provided. Infer duration if start
    and end are provided. Infer end if start and duration are provided.

    Args:
        start (str): The start time in the format 'YYYYMMDD HHMMSS' or other
            ISO 8601 formats, inclusive. If None, load all. Default is None.
        duration (float): Duration of the time in seconds. Defaults to None.
        end (str): The end time in the format 'YYYYMMDD HHMMSS' or other ISO
            8601 formats, exclusive. If None, load all. Default is None.

    Raises:
        ValueError: The function accepts two and only two out of three
            (start, duration, end)

    Returns:
        tuple[str, float, str]: start, duration, end. The start and end are
            always in the format 'YYYYMMDD HHMMSS' with optional fractional
            seconds.
    """
    # Check if two and only two out of three are inputted
    if (start is None) + (duration is None) + (end is None) != 1:
        raise ValueError('The function accepts two and only two out of '
                         + 'three (start, end, duration)')

    if start is not None:
        start = _parse_datetime_str(start).timestamp()

    if end is not None:
        end = _parse_datetime_str(end).timestamp()

    if duration is None:  # If start and end are provided
        duration = end - start
    elif start is None:  # If duration and end are provided
        start = end - duration
    else:  # If start and duration are provided
        end = start + duration

    # Convert back to strings
    start = _format_time_string(start)
    end = _format_time_string(end)

    return start, duration, end


def load(
    exp_path: str = None,
    t_start: str = None,
    t_end: str = None,
    dt: float = None,
    duration: float = None,
    s_start: int = None,
    s_end: int = None,
    ds: int = None,
    channels: list[int] = None,
    reset_channels: bool = False,
    scale: bool = True,
    integrate: bool = False,
    file_paths: list[str] = None
) -> tuple[np.ndarray[float], dict]:
    """Load DAS data from HDF5 files.

    Args:
        exp_path (str): Path to the experiment directory. Either `exp_path` or
            `file_paths` must be provided, but not both. Default is None.
        t_start (str): The start time in the format 'YYYYMMDD HHMMSS' or other
            ISO 8601 formats, inclusive. If None, load all. Default is None.
        t_end (str): The end time in the format 'YYYYMMDD HHMMSS' or other ISO
            8601 formats, exclusive. If None, load all. Default is None.
        dt (float): Temporal sampling period in seconds. If None, get the raw
            temporal sampling period, i.e., the smallest temporal sampling
            period in the data files. Default is None.
        duration (float): Duration of the time in seconds. Only two out of
            three (t_start, t_end, duration) should be provided. Defaults to
            None.
        s_start (int): The start channel number, inclusive. If None, load all.
            Default is None.
        s_end (int): The end channel number, exclusive. If None, load all.
            Default is None.
        ds (int): Spatial sampling period in number of channels. If None, get
            the raw spatial sampling period, i.e., the smallest spatial
            sampling period in the data files. Default is None.
        channels (list[int]): List of channel numbers. If provided, ignore
            `s_start`, `s_end` and `ds`. When the `channels` argument is
            provided with incontinuous channel numbers, e.g. 4, 6, 8, 100, 102,
            the function will take the provided channels in the data and
            rename them to continuous numbers, starting from the smallest, i.e.
            4, 6, 8, 10, 12. Default is None.
        reset_channels (bool): Whether to reset the channels to start from 0.
            Default is False.
        scale (bool): Whether to scale the data to get strain rate from time
            differentiated phase. Default is True.
        integrate (bool): Whether to integrate the data to get strain. Default
            is False.
        file_paths (list[str]): List of file paths. Either `exp_path` or
            `file_paths` must be provided, but not both. Default is None.

    Returns:
        tuple[np.ndarray[float], dict]: A tuple containing the data as a numpy
            array and metadata as a dictionary.
    """
    # Get the file paths if not provided
    ###########################################################################
    if exp_path is None and file_paths is None:
        raise ValueError("Either 'exp_path' or 'file_paths' must be provided.")
    if exp_path is not None and file_paths is not None:
        raise ValueError(
            "Only one of 'exp_path' or 'file_paths' must be provided.")
    if file_paths is None:
        if (t_start is None) and (duration is None) and (t_end is None):
            file_paths = get_all_hdf5_file_paths(exp_path)
        else:
            # Infer the time range
            t_start, duration, t_end = _infer_time(
                start=t_start,
                duration=duration,
                end=t_end
            )
            # Get the file paths from the experiment directory
            file_paths = get_hdf5_file_paths_range(
                exp_path=exp_path,
                start=t_start,
                end=t_end
            )

    # Load data
    ###########################################################################
    data = []
    cumulative_rows = 0  # track total rows processed across files
    # Iterate over all hdf5 files
    for i, file_path in enumerate(file_paths):

        # Get the header information from the first file
        #######################################################################
        if i == 0:
            # Get the header information from the first file
            header = get_hdf5_header(file_path)

            # Get channels and channels indices
            if channels is not None:
                if not set(channels).issubset(set(header.channels)):
                    raise ValueError(
                        "The desired channels are not present in the raw data."
                    )
                # Get indices
                channels_idx = np.searchsorted(header.channels, channels)
                # Infer ds by getting the mode of the differences
                ds_array = np.diff(channels)
                mode_result = mode(ds_array)
                ds = mode_result.mode
                # Reset the channels to continuous numbers starting from the
                # smallest
                channels = np.arange(
                    start=channels[0],
                    stop=channels[0] + len(channels) * ds,
                    step=ds
                )

            else:
                # Spatial channel sampling period ds
                if ds is None:
                    ds = header.ds
                elif ds % header.ds != 0:
                    raise ValueError(
                        f"The desired channel sampling period ({ds}) is "
                        + "not a multiple of the raw's channel sampling period"
                        + f" ({header.ds})."
                    )

                # Adjust the spatial channel sampling rate
                if s_start is None:
                    s_start = header.s_start
                if s_end is None:
                    s_end = header.s_end
                # Convert requested channel bounds to array indices relative
                # to the first channel to avoid relying on absolute channel
                # numbers (which may not start at zero).
                raw_channels = header.channels
                s_start_idx = np.searchsorted(
                    raw_channels, s_start, side='left'
                )
                s_end_idx = np.searchsorted(
                    raw_channels, s_end, side='left'
                )
                s_step_idx = int(ds / header.ds)
                channels_idx = np.arange(s_start_idx, s_end_idx, s_step_idx)
                if channels_idx.size == 0:
                    raise ValueError(
                        "No channels found for the requested spatial range."
                    )
                channels = header.channels[channels_idx]

            if reset_channels:
                channels = np.arange(start=0, stop=len(channels) * ds, step=ds)

            # Map the input temporal range to the raw temporal range index
            if dt is None:
                dt = header.dt
            else:
                temporal_ratio = dt / header.dt
                rounded_ratio = round(temporal_ratio)
                if rounded_ratio <= 0 or not math.isclose(
                    temporal_ratio,
                    rounded_ratio,
                    rel_tol=1e-9,
                    abs_tol=1e-12
                ):
                    raise ValueError(
                        f"The desired temporal sampling period ({dt}) is "
                        + "not a multiple of the raw's temporal sampling "
                        + f"period ({header.dt})."
                    )
            if t_start is None:  # Means file_paths was provided (not exp_path)
                t_start = header.t_start  # Unix timestamp
            else:
                t_start = _parse_datetime_str(t_start).timestamp()
                # The input start time may not match to any raw data point.
                # So we need to adjust the start time to the nearest next
                # raw data point.
                difference = t_start - header.t_start
                # Round to avoid floating point errors
                i = math.ceil(round(difference / header.dt, 2))
                t_start = header.t_start + (i * header.dt)
                # Ensure that the start time is not before the raw start
                t_start = np.max([t_start, header.t_start])
            temporal_ratio = dt / header.dt
            rounded_ratio = round(temporal_ratio)
            if rounded_ratio <= 0:
                raise ValueError(
                    "The desired temporal sampling period must be at least "
                    "the raw temporal sampling period."
                )
            t_step_idx = int(rounded_ratio)

        # From the second file, only get the data (and slice the data)
        ###################################################################
        with h5py.File(file_path, 'r') as hdf_file:
            raw_data = hdf_file['data']
            num_rows = raw_data.shape[0]
            # maintain consistent downsampling across file boundaries
            t_start_idx_infile = (-cumulative_rows) % t_step_idx
            t_idx_infile = np.arange(
                t_start_idx_infile, num_rows, t_step_idx
            )
            data_slice = raw_data[t_idx_infile, :][:, channels_idx]
            data.append(data_slice)
            cumulative_rows += num_rows

    # Combine all data into a single numpy array
    data = np.concatenate(data, axis=0)

    t_end_raw = header.t_start + data.shape[0] * dt
    if t_end is None:
        t_end = t_end_raw
    else:
        t_end = _parse_datetime_str(t_end).timestamp()
        # Fix the end time to match the raw data
        difference = t_end - t_start
        i = math.ceil(round(difference / dt, 2))
        t_end = t_start + i * dt
        # Ensure that the end time is not after the raw end
        t_end = np.min([t_end, t_end_raw])

    # Slice the data to only keep the provided time range
    #######################################################################
    # Round to avoid floating point errors
    t_start_idx = int(round((t_start - header.t_start) / dt))
    t_end_idx = int(round((t_end - header.t_start) / dt))
    data = data[t_start_idx:t_end_idx]

    # Scale and integrate the data
    #######################################################################
    if scale:  # Get strain rate from time differentiated phase
        data = _scale(
            data=data,
            scale_factor=header.data_scale / header.sensitivity
        )
    if integrate:  # Integrate data to get strain
        data = _integrate(data=data, dt=dt)

    # Generate the timestamps
    #######################################################################
    # Don't use np.arange(t_start, t_start + data.shape[0] * t_step, t_step)
    # because it may not be accurate due to floating point arithmetic
    timestamps = np.linspace(
        start=t_start,
        stop=t_end,
        num=data.shape[0],
        endpoint=False
    )
    # Generate metadata
    #######################################################################
    meta = {
        'dt': dt,  # Temporal sampling period in seconds
        'ds': ds,  # Spatial sampling period in number of channels
        'dx': header.dx,  # Channel spacing in meters
        'timestamps': timestamps,
        'channels': channels,
        'gauge_length': header.gauge_length,
        'data_scale': header.data_scale,
        'sensitivity': header.sensitivity,
        'file_paths': file_paths
    }
    return data, meta


def to_df(
    data: np.ndarray[float],
    channels: np.ndarray[int],
    timestamps: np.ndarray[float],
) -> pd.DataFrame:
    """Convert the DASArray to a pandas DataFrame.

    Args:
        data (np.ndarray[float]): DAS data.
        channels (np.ndarray[int]): Channel numbers.
        timestamps (np.ndarray[float]): Unix timestamps.

    Returns:
        pd.DataFrame: DAS data as a DataFrame.
    """
    timestamp_datetime = [
        datetime.fromtimestamp(ts, tz=timezone.utc) for ts in timestamps
    ]
    data_to_use = data[:, :, 0] if data.ndim == 3 else data
    df = pd.DataFrame(
        data=data_to_use,
        index=timestamp_datetime,
        columns=channels
    )
    return df


class DASLoader:

    def load(
        self,
        exp_path: str = None,
        t_start: str = None,
        t_end: str = None,
        dt: float = None,
        duration: float = None,
        s_start: int = None,
        s_end: int = None,
        ds: int = None,
        channels: list[int] = None,
        reset_channels: bool = False,
        scale: bool = True,
        integrate: bool = False,
        file_paths: list[str] = None
    ) -> 'DASArray':
        """Load DAS data from HDF5 files.

        Args:
            exp_path (str): Path to the experiment directory. Either `exp_path`
                or `file_paths` must be provided, but not both. Default is
                None.
            t_start (str): The start time in the format 'YYYYMMDD HHMMSS' or
                other ISO 8601 formats, inclusive. If None, load all. Default
                is None.
            t_end (str): The end time in the format 'YYYYMMDD HHMMSS' or other
                ISO 8601 formats, exclusive. If None, load all. Default is
                None.
            dt (float): Temporal sampling period in seconds. If None, get the
                raw temporal sampling period, i.e., the smallest temporal
                sampling period in the data files. Default is None.
            duration (float): Duration of the time in seconds. Only two out of
                three (t_start, t_end, duration) should be provided. Defaults
                to None.
            s_start (int): The start channel number, inclusive. If None, load
                all. Default is None.
            s_end (int): The end channel number, exclusive. If None, load all.
                Default is None.
            ds (int): Spatial sampling period in number of channels. If None,
                get the raw spatial sampling period, i.e., the smallest spatial
                sampling period in the data files. Default is None.
            channels (list[int]): List of channel numbers. If provided, ignore
                `s_start`, `s_end` and `ds`. When the `channels` argument is
                provided with incontinuous channel numbers, e.g. 4, 6, 8, 100,
                102, the function will take the provided channels in the data
                and rename them to continuous numbers, starting from the
                smallest, i.e. 4, 6, 8, 10, 12. Default is None.
            reset_channels (bool): Whether to reset the channels to start from
                0. Default is False.
            scale (bool): Whether to scale the data to get strain rate from
                time differentiated phase. Default is True.
            integrate (bool): Whether to integrate the data to get strain.
                Default is False.
            file_paths (list[str]): List of file paths. Either `exp_path` or
                `file_paths` must be provided, but not both. Default is None.

        Returns:
            DASArray: DAS data.
        """
        data, meta = load(
            exp_path=exp_path,
            t_start=t_start,
            t_end=t_end,
            dt=dt,
            duration=duration,
            s_start=s_start,
            s_end=s_end,
            ds=ds,
            channels=channels,
            reset_channels=reset_channels,
            scale=scale,
            integrate=integrate,
            file_paths=file_paths
        )
        result = self.__class__(data, **meta)
        return result

    def to_df(self) -> pd.DataFrame:
        """Convert the DASArray to a pandas DataFrame.

        Returns:
            pd.DataFrame: DAS data as a DataFrame.
        """
        return to_df(
            data=self,
            channels=self.meta.channels,
            timestamps=self.meta.timestamps
        )
