"""Module for searching and collecting HDF5 file paths."""

import os
from datetime import datetime, timedelta
from dataclasses import dataclass

import h5py
import numpy as np


def parse_file_path(file_path: str) -> tuple[str, str, str, str]:
    """Parses the file path to extract exp_path, yyyymmdd, hhmmss and file name
    (hhmmss.hdf5).

    Args:
        file_path (str): The file path to parse.

    Returns:
        tuple[str, str, str, str]: A tuple containing exp_path, yyyymmdd,
            hhmmss and file name (hhmmss.hdf5).
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} does not exist.")
    try:
        exp_path, yyyymmdd, _, hhmmss_hdf5 = file_path.rsplit("/", 3)
        hhmmss = hhmmss_hdf5.removesuffix(".hdf5")
    except ValueError as e:
        raise ValueError(
            "Invalid file path format. Expected format: "
            "<exp_path>/<YYYYMMDD>/dphi/<HHMMSS>.hdf5"
        ) from e
    return exp_path, yyyymmdd, hhmmss, hhmmss_hdf5


def _get_datetime_from_strings(yyyymmdd: str, hhmmss: str) -> datetime:
    """Converts date and time strings into a datetime object.

    Args:
        yyyymmdd (str): The date string in the format 'YYYYMMDD'.
        hhmmss (str): The time string in the format 'HHMMSS'.

    Returns:
        datetime: The datetime object.
    """
    try:
        return datetime.strptime(f"{yyyymmdd}{hhmmss}", "%Y%m%d%H%M%S")
    except ValueError as e:
        raise ValueError("Invalid date or time format in file path.") from e


def _get_available_dates(exp_path: str) -> list[str]:
    """Gets available date directories in reverse chronological order.

    Args:
        exp_path (str): The experiment path.

    Returns:
        list[str]: A list of available date directories.
    """
    exp_path_dates = os.path.join(exp_path)
    available_dates = [
        d for d in os.listdir(exp_path_dates)
        if os.path.isdir(os.path.join(exp_path_dates, d))
        and d.isdigit() and len(d) == 8
    ]
    return sorted(available_dates, reverse=True)


def _get_file_times(date_dir: str) -> list[str]:
    """Gets all HHMMSS time strings available in a date directory.

    Args:
        date_dir (str): The date directory containing hdf5 files.

    Returns:
        list[str]: A list of HHMMSS strings.
    """
    try:
        file_list = os.listdir(date_dir)
    except OSError:
        return []
    file_times = [
        f[:-5] for f in file_list
        if f.endswith('.hdf5') and f[:-5].isdigit() and len(f[:-5]) == 6
    ]
    return sorted(file_times, reverse=True)


@dataclass
class HDF5HeaderInfo:
    """Dataclass for HDF5 header information."""
    dn: float  # Spatial resolution (channel sampling period)
    n_start: int  # Start channel
    n_end: int  # End channel
    channels: np.ndarray  # Channel numbers
    dt: float  # Time resolution
    t_start: float  # Start timestamp
    t_end: float  # End timestamp
    timestamps: np.ndarray  # Timestamps
    dxn: float  # Distance in meters between two consecutive channel indices
    gauge_length: float  # Gauge length in meters
    data_scale: float  # Data scale factor
    sensitivity: float  # Sensitivity factor


def get_hdf5_header(file_path: str) -> HDF5HeaderInfo:
    """Reads the header information from an HDF5 file.

    Args:
        file_path (str): Path to the HDF5 file.

    Returns:
        HDF5HeaderInfo: A dataclass containing the header information.
    """
    with h5py.File(file_path, 'r') as hdf_file:
        channels = hdf_file['header/channels'][()]
        dn = channels[1] - channels[0]
        n_start = channels[0]
        n_end = channels[-1] + int(dn)

        dt = hdf_file['header/dt'][()]
        N = hdf_file['header/dimensionRanges/dimension0/size'][()]
        t_start = hdf_file['header/time'][()]
        t_end = t_start + dt * (N - 1)
        timestamps = np.linspace(t_start, t_end, N)

        dxn = hdf_file['header/dx'][()]
        gauge_length = hdf_file['header/gaugeLength'][()]
        data_scale = hdf_file['header/dataScale'][()]
        sensitivity = hdf_file['header/sensitivities'][()][0][0]

        return HDF5HeaderInfo(
            dn=dn,
            n_start=n_start,
            n_end=n_end,
            channels=channels,
            dt=dt,
            t_start=t_start,
            t_end=t_end,
            timestamps=timestamps,
            dxn=dxn,
            gauge_length=gauge_length,
            data_scale=data_scale,
            sensitivity=sensitivity
        )


def _is_time_within_gap(
    last_dt: datetime,
    current_dt: datetime,
    max_gap_seconds: int = 15
) -> bool:
    """Checks if the time gap between two datetimes is within the allowed
    range.

    Args:
        last_dt (datetime): The last datetime.
        current_dt (datetime): The current datetime to check.
        max_gap_seconds (int): Maximum allowed time gap in seconds.

    Returns:
        bool: True if the gap is within the allowed range, False otherwise.
    """
    time_diff = last_dt - current_dt
    return timedelta(0) < time_diff <= timedelta(seconds=max_gap_seconds)


def get_recent_hdf5_file_paths(file_path: str, num_file: int) -> list[str]:
    """Gets a list of HDF5 file paths including the input file_path and
    previous (num_file - 1) continuous files without time gaps.

    Args:
        file_path (str): The input file path.
        num_file (int): The number of files to collect.

    Returns:
        List[str]: A list of HDF5 file paths.
    """
    if num_file <= 0:
        return []

    if num_file == 1:
        return [file_path]

    exp_path, yyyymmdd, hhmmss, _ = parse_file_path(file_path)
    available_dates = _get_available_dates(exp_path)
    file_path_dt = _get_datetime_from_strings(yyyymmdd, hhmmss)

    collected_files = [file_path]
    last_dt = file_path_dt
    num_collected = 1

    start_date_index = available_dates.index(yyyymmdd)

    for date_str in available_dates[start_date_index:]:
        date_dir = os.path.join(exp_path, date_str, 'dphi')

        if not os.path.isdir(date_dir):
            continue

        file_times = _get_file_times(date_dir)

        if date_str == yyyymmdd:
            file_times = [t for t in file_times if t < hhmmss]

        for t in file_times:
            current_dt = datetime.strptime(date_str + t, '%Y%m%d%H%M%S')
            if _is_time_within_gap(last_dt, current_dt):
                file_candidate = os.path.join(date_dir, f"{t}.hdf5")
                if os.path.exists(file_candidate):
                    collected_files.append(file_candidate)
                    last_dt = current_dt
                    num_collected += 1
                    if num_collected >= num_file:
                        return collected_files[::-1]
            else:
                break

    return collected_files[::-1]
