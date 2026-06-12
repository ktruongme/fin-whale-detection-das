"""This module contains functions to load DAS data from HDF5 files."""

from datetime import datetime, timezone
from typing import TYPE_CHECKING
import math
import time
import warnings

import numpy as np
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


_FILE_CONTINUITY_TOLERANCE_S = 1.0


class IncompatibleFileBatchError(ValueError):
    """Raised when selected files cannot be loaded as one continuous batch."""


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


def _read_raw_sampling_info(
    file_path: str
) -> tuple[float, np.ndarray, float, int]:
    """Read only the raw metadata needed for loader preflight checks."""
    with h5py.File(file_path, 'r') as hdf_file:
        dt = float(hdf_file['header/dt'][()])
        channels = hdf_file['header/channels'][()]
        t_start = float(hdf_file['header/time'][()])
        num_rows = int(hdf_file['header/dimensionRanges/dimension0/size'][()])
    return dt, channels, t_start, num_rows


def _ensure_homogeneous_sampling(file_paths: list[str]) -> None:
    """Reject file collections with incompatible raw sampling metadata."""
    if len(file_paths) <= 1:
        return

    first_path = file_paths[0]
    first_dt, first_channels, prev_t_start, prev_num_rows = (
        _read_raw_sampling_info(first_path)
    )
    temporal_mismatches = []
    spatial_mismatches = []
    continuity_mismatches = []
    prev_path = first_path
    prev_dt = first_dt

    for file_path in file_paths[1:]:
        current_dt, current_channels, current_t_start, current_num_rows = (
            _read_raw_sampling_info(file_path)
        )
        if not math.isclose(
            current_dt,
            first_dt,
            rel_tol=1e-9,
            abs_tol=1e-12
        ):
            temporal_mismatches.append((file_path, current_dt))
        if not np.array_equal(current_channels, first_channels):
            spatial_mismatches.append(file_path)
        expected_t_start = prev_t_start + (prev_num_rows * prev_dt)
        gap = abs(current_t_start - expected_t_start)
        if gap > _FILE_CONTINUITY_TOLERANCE_S:
            continuity_mismatches.append(
                (prev_path, file_path, expected_t_start, current_t_start)
            )
        prev_path = file_path
        prev_dt = current_dt
        prev_t_start = current_t_start
        prev_num_rows = current_num_rows

    if temporal_mismatches:
        mismatch_summary = ", ".join(
            f"{path.rsplit('/', 1)[-1]}: dt={dt:g}s"
            for path, dt in temporal_mismatches[:5]
        )
        if len(temporal_mismatches) > 5:
            mismatch_summary += ", ..."
        raise IncompatibleFileBatchError(
            "Mixed raw temporal sampling periods are not supported by "
            "dasly.loader.load. Split the request into homogeneous groups "
            f"before loading. Expected dt={first_dt:g}s from "
            f"{first_path.rsplit('/', 1)[-1]}; mismatches: "
            f"{mismatch_summary}."
        )

    if spatial_mismatches:
        first_dn = (
            float(first_channels[1] - first_channels[0])
            if len(first_channels) > 1 else 0.0
        )
        mismatch_summary = ", ".join(
            path.rsplit("/", 1)[-1] for path in spatial_mismatches[:5]
        )
        if len(spatial_mismatches) > 5:
            mismatch_summary += ", ..."
        raise IncompatibleFileBatchError(
            "Mixed raw spatial sampling is not supported by "
            "dasly.loader.load. All selected files must share the same "
            "raw channel grid. "
            f"Expected dn={first_dn:g} and the same channels as "
            f"{first_path.rsplit('/', 1)[-1]}; mismatches: "
            f"{mismatch_summary}."
        )

    if continuity_mismatches:
        mismatch_summary = ", ".join(
            f"{prev_path.rsplit('/', 1)[-1]} -> "
            f"{file_path.rsplit('/', 1)[-1]}"
            for prev_path, file_path, _, _ in continuity_mismatches[:5]
        )
        if len(continuity_mismatches) > 5:
            mismatch_summary += ", ..."
        raise IncompatibleFileBatchError(
            "Selected files are not temporally contiguous enough for "
            "dasly.loader.load. Files must be ordered and contiguous within "
            f"{_FILE_CONTINUITY_TOLERANCE_S:g}s. Discontinuities: "
            f"{mismatch_summary}."
        )


def load(
    exp_path: str = None,
    t_start: str = None,
    t_end: str = None,
    dt: float = None,
    duration: float = None,
    n_start: int = None,
    n_end: int = None,
    dn: int = None,
    channels: list[int] = None,
    exclude_channels: list[int] = None,
    reset_channels: bool = False,
    scale: bool = False,
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
        dt (float): Temporal sampling period in seconds. If None, use the raw
            temporal sampling period from the selected files. All selected
            files must share the same raw temporal sampling period. Default is
            None.
        duration (float): Duration of the time in seconds. Only two out of
            three (t_start, t_end, duration) should be provided. Defaults to
            None.
        n_start (int): The start channel number, inclusive. If None, load all.
            Default is None.
        n_end (int): The end channel number, exclusive. If None, load all.
            Default is None.
        dn (int): Spatial sampling period in number of channels. If None, get
            the raw spatial sampling period, i.e., the smallest spatial
            sampling period in the data files. Default is None.
        channels (list[int]): List of channel numbers. If provided, ignore
            `n_start`, `n_end` and `dn`. Whenever `channels` (or
            `exclude_channels`) is used, the resulting `meta.channels` is
            renumbered onto a uniform-gap grid (step = mode of the raw
            channel diffs). The mapping from each renumbered label to the
            original HDF5 cable index is stored in `meta.channel_map`
            (load-time artifact only -- not kept in sync by downstream
            spatial transformations). Default is None.
        exclude_channels (list[int]): List of channel numbers to exclude from
            loading. Channels in this list that do not exist in the data are
            silently ignored. Applied after `channels` or `n_start`/`n_end`/
            `dn` selection. The post-exclusion channels are renumbered to a
            uniform-gap grid (see `channels`). Default is None.
        reset_channels (bool): If True, the renumbered `meta.channels` starts
            at 0; if False, it starts at the first surviving raw channel.
            `meta.channel_map` always carries the renumbered -> raw cable
            index mapping. Default is False.
        scale (bool): Whether to scale the data to get strain rate from time
            differentiated phase. Default is False.
        integrate (bool): Whether to integrate the data to get strain. Default
            is False.
        file_paths (list[str]): List of file paths. Either `exp_path` or
            `file_paths` must be provided, but not both. All selected files
            must share the same raw temporal sampling period and raw channel
            grid, and must be ordered and temporally contiguous within 1
            second. Default is None.

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
    _ensure_homogeneous_sampling(file_paths)

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
                if exclude_channels is not None:
                    exclude_set = set(exclude_channels)
                    channels = [c for c in channels
                                if c not in exclude_set]
                available_set = set(header.channels)
                missing = [c for c in channels if c not in available_set]
                if missing:
                    warnings.warn(
                        f"{len(missing)} of {len(channels)} requested "
                        f"channels are not present in the raw data and "
                        f"will be ignored (e.g. {missing[:5]})."
                    )
                    channels = [c for c in channels if c in available_set]
                if not channels:
                    raise ValueError(
                        "None of the desired channels are present in the "
                        "raw data."
                    )
                # Get indices — convert to slice for fast HDF5 hyperslab reads
                channels_idx = np.searchsorted(header.channels, channels)
                if len(channels_idx) == 1:
                    channels_idx = slice(
                        int(channels_idx[0]), int(channels_idx[0]) + 1
                    )
                else:
                    idx_step = int(channels_idx[1] - channels_idx[0])
                    if idx_step > 0 and np.all(
                        np.diff(channels_idx) == idx_step
                    ):
                        channels_idx = slice(
                            int(channels_idx[0]),
                            int(channels_idx[-1]) + idx_step,
                            idx_step
                        )

            else:
                # Spatial channel sampling period dn
                if dn is None:
                    dn = header.dn
                elif dn % header.dn != 0:
                    raise ValueError(
                        f"The desired channel sampling period ({dn}) is "
                        + "not a multiple of the raw's channel sampling period"
                        + f" ({header.dn})."
                    )

                # Adjust the spatial channel sampling rate
                if n_start is None:
                    n_start = header.n_start
                if n_end is None:
                    n_end = header.n_end
                # Convert requested channel bounds to array indices relative
                # to the first channel to avoid relying on absolute channel
                # numbers (which may not start at zero).
                raw_channels = header.channels
                n_start_idx = np.searchsorted(
                    raw_channels, n_start, side='left'
                )
                n_end_idx = np.searchsorted(
                    raw_channels, n_end, side='left'
                )
                n_step_idx = int(dn / header.dn)
                channels_idx = slice(n_start_idx, n_end_idx, n_step_idx)
                channels = header.channels[channels_idx]
                if channels.size == 0:
                    raise ValueError(
                        "No channels found for the requested spatial range."
                    )
                if exclude_channels is not None:
                    mask = ~np.isin(channels, exclude_channels)
                    channels = channels[mask]
                    if channels.size == 0:
                        raise ValueError(
                            "No channels remaining after exclusion."
                        )
                    # Recompute indices for HDF5 hyperslab reads
                    channels_idx = np.searchsorted(
                        header.channels, channels
                    )
                    if len(channels_idx) == 1:
                        channels_idx = slice(
                            int(channels_idx[0]),
                            int(channels_idx[0]) + 1
                        )
                    else:
                        idx_diffs = np.diff(channels_idx)
                        idx_step = int(idx_diffs[0])
                        if idx_step > 0 and np.all(
                            idx_diffs == idx_step
                        ):
                            channels_idx = slice(
                                int(channels_idx[0]),
                                int(channels_idx[-1]) + idx_step,
                                idx_step
                            )

            # Snapshot the true HDF5 cable indices before renumbering so
            # we can build the new->raw channel map below.
            raw_indices = np.asarray(channels).copy()

            # Renumber to a uniform-gap grid. Step is the mode of the raw
            # diffs (no-op when the raw channels are already uniform; picks
            # the typical spacing when exclusions or arbitrary `channels`
            # carve gaps). Start is 0 when reset_channels, else the first
            # surviving raw channel.
            if len(raw_indices) > 1:
                dn = int(mode(np.diff(raw_indices)).mode)
            elif dn is None:
                dn = int(header.dn)
            ch_start = 0 if reset_channels else int(raw_indices[0])
            channels = np.arange(
                start=ch_start,
                stop=ch_start + len(raw_indices) * dn,
                step=dn,
            )

            # Map each renumbered channel label to its raw HDF5 cable
            # index. This is a load-time artifact only -- it is NOT kept
            # consistent by downstream operations (spatial subsetters /
            # interpolators), so consumers should read it immediately
            # after `load()` (e.g. when writing a Zarr store) and not
            # rely on it after spatial transformations.
            channel_map = {
                int(new): int(raw)
                for new, raw in zip(channels, raw_indices)
            }

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
            # Use slice (not np.arange) for HDF5 hyperslab selection
            t_slice = slice(int(t_start_idx_infile), num_rows, t_step_idx)
            data_slice = raw_data[t_slice, channels_idx]
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
        'dn': dn,  # Spatial sampling period in number of channels
        'dxn': header.dxn,  # Channel spacing in meters
        'timestamps': timestamps,
        'channels': channels,
        'channel_map': channel_map,
        'gauge_length': header.gauge_length,
        'data_scale': header.data_scale,
        'sensitivity': header.sensitivity,
        'file_paths': file_paths
    }
    return data, meta


class DASLoader:

    def load(
        self,
        exp_path: str = None,
        t_start: str = None,
        t_end: str = None,
        dt: float = None,
        duration: float = None,
        n_start: int = None,
        n_end: int = None,
        dn: int = None,
        channels: list[int] = None,
        exclude_channels: list[int] = None,
        reset_channels: bool = False,
        scale: bool = False,
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
            dt (float): Temporal sampling period in seconds. If None, use the
                raw temporal sampling period from the selected files. All
                selected files must share the same raw temporal sampling
                period. Default is None.
            duration (float): Duration of the time in seconds. Only two out of
                three (t_start, t_end, duration) should be provided. Defaults
                to None.
            n_start (int): The start channel number, inclusive. If None, load
                all. Default is None.
            n_end (int): The end channel number, exclusive. If None, load all.
                Default is None.
            dn (int): Spatial sampling period in number of channels. If None,
                get the raw spatial sampling period, i.e., the smallest spatial
                sampling period in the data files. Default is None.
            channels (list[int]): List of channel numbers. If provided, ignore
                `n_start`, `n_end` and `dn`. Whenever `channels` (or
                `exclude_channels`) is used, the resulting `meta.channels` is
                renumbered onto a uniform-gap grid (step = mode of the raw
                channel diffs). The mapping from each renumbered label to
                the original HDF5 cable index is stored in
                `meta.channel_map` (load-time artifact only -- not kept in
                sync by downstream spatial transformations). Default is
                None.
            exclude_channels (list[int]): List of channel numbers to exclude.
                Channels not present in the data are silently ignored. The
                post-exclusion channels are renumbered to a uniform-gap grid
                (see `channels`). Default is None.
            reset_channels (bool): If True, the renumbered `meta.channels`
                starts at 0; if False, it starts at the first surviving raw
                channel. `meta.channel_map` always carries the renumbered ->
                raw cable index mapping. Default is False.
            scale (bool): Whether to scale the data to get strain rate from
                time differentiated phase. Default is False.
            integrate (bool): Whether to integrate the data to get strain.
                Default is False.
            file_paths (list[str]): List of file paths. Either `exp_path` or
                `file_paths` must be provided, but not both. All selected
                files must share the same raw temporal sampling period and raw
                channel grid, and must be ordered and temporally contiguous
                within 1 second. Default is None.

        Returns:
            DASArray: DAS data.
        """
        timer = time.time()  # Start the timer
        data, meta = load(
            exp_path=exp_path,
            t_start=t_start,
            t_end=t_end,
            dt=dt,
            duration=duration,
            n_start=n_start,
            n_end=n_end,
            dn=dn,
            channels=channels,
            exclude_channels=exclude_channels,
            reset_channels=reset_channels,
            scale=scale,
            integrate=integrate,
            file_paths=file_paths
        )
        result = self.__class__(data, **meta)
        result.meta.update(_timer=timer)  # Update the timer
        return result
