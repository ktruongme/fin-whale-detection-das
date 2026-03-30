"""Load the HDF5 data needed by the CLI detection pipeline."""

from typing import TYPE_CHECKING
import math

import numpy as np
from scipy.stats import mode
import h5py

from .fsearcher import get_hdf5_header

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
    reset_channels: bool = False,
    scale: bool = True,
    integrate: bool = False,
    file_paths: list[str] = None
) -> tuple[np.ndarray[float], dict]:
    """Load DAS data from explicit HDF5 file paths.

    Args:
        exp_path (str): Unsupported in the CLI-only loader. Default is None.
        t_start (str): Unsupported in the CLI-only loader. Default is None.
        t_end (str): Unsupported in the CLI-only loader. Default is None.
        dt (float): Temporal sampling period in seconds. If None, get the raw
            temporal sampling period. Default is None.
        duration (float): Unsupported in the CLI-only loader. Default is None.
        n_start (int): The start channel number, inclusive. If None, load all.
            Default is None.
        n_end (int): The end channel number, exclusive. If None, load all.
            Default is None.
        dn (int): Spatial sampling period in number of channels. If None, get
            the raw spatial sampling period. Default is None.
        channels (list[int]): List of channel numbers. If provided, ignore
            `n_start`, `n_end` and `dn`. Default is None.
        reset_channels (bool): Whether to reset the channels to start from 0.
            Default is False.
        scale (bool): Whether to scale the data to get strain rate from time
            differentiated phase. Default is True.
        integrate (bool): Whether to integrate the data to get strain. Default
            is False.
        file_paths (list[str]): List of HDF5 file paths to load. Required.

    Returns:
        tuple[np.ndarray[float], dict]: A tuple containing the data as a numpy
            array and metadata as a dictionary.
    """
    if file_paths is None:
        raise ValueError("'file_paths' must be provided.")
    if not file_paths:
        raise ValueError("'file_paths' must contain at least one HDF5 file.")
    if exp_path is not None:
        raise ValueError(
            "Loading from 'exp_path' was removed from the CLI path."
        )
    if any(value is not None for value in (t_start, t_end, duration)):
        raise ValueError(
            "Time-range loading was removed from the CLI path. "
            "Provide explicit 'file_paths' instead."
        )

    # Load data
    ###########################################################################
    data = []
    cumulative_rows = 0
    for i, file_path in enumerate(file_paths):

        if i == 0:
            header = get_hdf5_header(file_path)

            # Get channels and channels indices
            if channels is not None:
                if not set(channels).issubset(set(header.channels)):
                    raise ValueError(
                        "The desired channels are not present in the raw data."
                    )
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
                dn_array = np.diff(channels)
                mode_result = mode(dn_array)
                dn = mode_result.mode
                channels = np.arange(
                    start=channels[0],
                    stop=channels[0] + len(channels) * dn,
                    step=dn
                )

            else:
                if dn is None:
                    dn = header.dn
                elif dn % header.dn != 0:
                    raise ValueError(
                        f"The desired channel sampling period ({dn}) is "
                        + "not a multiple of the raw's channel sampling period"
                        + f" ({header.dn})."
                    )

                if n_start is None:
                    n_start = header.n_start
                if n_end is None:
                    n_end = header.n_end
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

            if reset_channels:
                channels = np.arange(start=0, stop=len(channels) * dn, step=dn)

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
            t_start = header.t_start
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
            t_start_idx_infile = (-cumulative_rows) % t_step_idx
            # Use slice (not np.arange) for HDF5 hyperslab selection
            t_slice = slice(int(t_start_idx_infile), num_rows, t_step_idx)
            data_slice = raw_data[t_slice, channels_idx]
            data.append(data_slice)
            cumulative_rows += num_rows

    # Combine all data into a single numpy array
    data = np.concatenate(data, axis=0)

    t_end = header.t_start + data.shape[0] * dt

    # Scale and integrate the data
    #######################################################################
    if scale:
        data = _scale(
            data=data,
            scale_factor=header.data_scale / header.sensitivity
        )
    if integrate:
        data = _integrate(data=data, dt=dt)

    # Generate the timestamps
    #######################################################################
    timestamps = np.linspace(
        start=t_start,
        stop=t_end,
        num=data.shape[0],
        endpoint=False
    )
    # Generate metadata
    #######################################################################
    meta = {
        'dt': dt,
        'dn': dn,
        'dxn': header.dxn,
        'timestamps': timestamps,
        'channels': channels,
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
        reset_channels: bool = False,
        scale: bool = True,
        integrate: bool = False,
        file_paths: list[str] = None
    ) -> 'DASArray':
        """Load DAS data from HDF5 files.

        Args:
            exp_path (str): Path to the experiment directory. Default is None.
            t_start (str): Start time. Default is None.
            t_end (str): End time. Default is None.
            dt (float): Temporal sampling period. Default is None.
            duration (float): Duration in seconds. Default is None.
            n_start (int): Start channel number. Default is None.
            n_end (int): End channel number. Default is None.
            dn (int): Spatial sampling period. Default is None.
            channels (list[int]): Channel numbers. Default is None.
            reset_channels (bool): Reset channels to start from 0.
            scale (bool): Scale data. Default is True.
            integrate (bool): Integrate data. Default is False.
            file_paths (list[str]): File paths. Default is None.

        Returns:
            DASArray: DAS data.
        """
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
            reset_channels=reset_channels,
            scale=scale,
            integrate=integrate,
            file_paths=file_paths
        )
        result = self.__class__(data, **meta)
        return result
