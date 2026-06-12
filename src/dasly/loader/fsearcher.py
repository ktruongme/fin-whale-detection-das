"""Module for searching and collecting HDF5 file paths."""

import os
import re
from datetime import datetime, timedelta, timezone
from typing import Optional
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


def get_all_hdf5_file_paths(directory: str) -> list[str]:
    """Get all HDF5 files in a directory and its subdirectories.

    Args:
        directory (str): The directory to search for HDF5 files.

    Returns:
        list[str]: A list of HDF5 file paths.
    """
    hdf5_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.h5') or file.endswith('.hdf5'):
                hdf5_files.append(os.path.join(root, file))

    # Sort the list of HDF5 files in reverse chronological order
    hdf5_files.sort()
    return hdf5_files


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


def _normalize_iso_datetime_string(dt_str: str) -> str:
    """Normalize ISO strings for datetime.fromisoformat."""
    if dt_str.endswith('Z'):
        dt_str = dt_str[:-1] + '+00:00'
    match = re.match(
        r'^(?P<prefix>.*\d)(?P<fraction>\.\d+)'
        r'(?P<suffix>(?:Z|[+-]\d{2}:?\d{2})?)$',
        dt_str
    )
    if not match:
        return dt_str
    fraction = match.group('fraction')[1:]
    padded_fraction = (fraction + '000000')[:6]
    suffix = match.group('suffix') or ''
    return f"{match.group('prefix')}.{padded_fraction}{suffix}"


def _parse_datetime_str(dt_str: str) -> datetime:
    """Parse date string supporting ISO and 'YYYYMMDD HHMMSS[.ffffff]'
    formats."""
    if dt_str is None:
        raise ValueError("Datetime string cannot be None.")
    normalized = _normalize_iso_datetime_string(dt_str.strip())
    try:
        dt = datetime.fromisoformat(normalized)
    except ValueError:
        match = re.fullmatch(
            r'(?P<date>\d{8}) (?P<time>\d{6})(?P<fraction>\.\d+)?',
            normalized
        )
        if not match:
            raise ValueError(
                "Invalid datetime format. Expected ISO 8601 or "
                "'YYYYMMDD HHMMSS[.ffffff]'."
            ) from None
        dt = datetime.strptime(
            f"{match.group('date')}{match.group('time')}",
            "%Y%m%d%H%M%S"
        )
        fraction = match.group('fraction')
        if fraction:
            microsecond = int((fraction[1:] + '000000')[:6])
            dt = dt.replace(microsecond=microsecond)
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        else:
            dt = dt.astimezone(timezone.utc)
    return dt


def _calculate_time_range(
    start: Optional[str] = None,
    duration: Optional[float] = None,
    end: Optional[str] = None
) -> tuple[datetime, datetime]:
    """Validates and calculates the start and end datetime objects based on
    provided arguments.

    Args:
        start (Optional[str]): The start time in the format 'YYYYMMDD HHMMSS'
            or other ISO 8601 formats. Default is None.
        duration (Optional[float]): The duration in seconds. Default is None.
        end (Optional[str]): The end time in the format 'YYYYMMDD HHMMSS' or
            other ISO 8601 formats. Default is None.

    Returns:
        tuple[datetime, datetime]: A tuple containing the start and end
            datetime objects.
    """
    provided_args = [start, end, duration]
    if sum(arg is not None for arg in provided_args) != 2:
        raise ValueError(
            "Exactly two of 'start', 'end', or 'duration' must be provided."
        )

    if start and end:
        start_dt = _parse_datetime_str(start)
        end_dt = _parse_datetime_str(end)
    elif start and duration:
        start_dt = _parse_datetime_str(start)
        end_dt = start_dt + timedelta(seconds=duration)
    else:  # end and duration
        end_dt = _parse_datetime_str(end)
        start_dt = end_dt - timedelta(seconds=duration)

    if start_dt >= end_dt:
        raise ValueError("Start time must be before end time.")

    return start_dt, end_dt


def _collect_files_in_range(
    exp_path: str,
    start_dt: datetime,
    end_dt: datetime
) -> list[str]:
    """Collects HDF5 file paths within a specified time range.

    This function searches for and retrieves all HDF5 files from the given
    experiment path (`exp_path`) that fall within the specified start
    (`start_dt`) and end (`end_dt`) datetime range. The first file corresponds
    to `start_dt` or later (inclusive), and the last file is the closest one
    before `end_dt` (exclusive).

    Args:
        exp_path (str): The experiment path.
        start_dt (datetime): The start datetime.
        end_dt (datetime): The end datetime.

    Returns:
        list[str]: A list of hdf5 file paths.
    """
    collected_files = []
    # include the preceding file to cover data before start_dt
    coverage = timedelta(seconds=10)
    adjusted_start_dt = start_dt - coverage

    # Collect dates to cover
    date_list = []
    current_date = adjusted_start_dt.date()
    end_date = end_dt.date()
    while current_date <= end_date:
        date_list.append(current_date.strftime('%Y%m%d'))
        current_date += timedelta(days=1)

    # Loop through the dates
    for date_str in date_list:

        # Skip if the date directory does not exist
        date_dir = os.path.join(exp_path, date_str, 'dphi')
        if not os.path.isdir(date_dir):
            continue

        # Get the list of file times in the date directory
        file_times = _get_file_times(date_dir)

        for hhmmss in file_times:

            # Get the datetime object of the file
            dt_format = '%Y%m%d %H%M%S'
            file_dt_str = f"{date_str} {hhmmss}"
            file_dt = datetime.strptime(file_dt_str, dt_format).replace(
                tzinfo=timezone.utc
            )

            # Include files whose nominal time is between adjusted_start_dt
            # (inclusive) and end_dt (exclusive)
            if adjusted_start_dt <= file_dt < end_dt:
                file_path = os.path.join(date_dir, f"{hhmmss}.hdf5")
                collected_files.append(file_path)

    # Sort collected files in chronological order
    collected_files.sort()

    return collected_files


def parse_file_datetime(file_path: str) -> datetime:
    """Extracts and returns the datetime from a file path.

    The file path is expected to have a date folder in the format 'YYYYMMDD'
    and a file name in the format 'HHMMSS.hdf5'. For example:
    '/Volumes/LaCie/Monaco/20250129/dphi/000001.hdf5'. In this case, the
    function extracts '20250129' from the directory and '000001' from the file
    name and returns the corresponding datetime.

    Args:
        file_path (str): The file path string.

    Returns:
        datetime: The datetime extracted from the file path.
    """
    # Split the file path into parts
    parts = file_path.split(os.sep)
    # Assuming the structure is fixed and the date folder is at index 4:
    # Example: ['', 'Volumes', 'LaCie', 'Monaco', '20250129', 'dphi',
    # '000001.hdf5']

    date_str = parts[-3]  # The date part (e.g., '20250129')
    # Extract the time part from the file name (e.g., '000001' from
    # '000001.hdf5')
    base_name = os.path.basename(file_path)
    time_str, ext = os.path.splitext(base_name)
    dt_str = f"{date_str} {time_str}"

    # Parse the combined datetime string.
    return datetime.strptime(dt_str, "%Y%m%d %H%M%S")


def get_files_from_datetime(
    file_paths: list[str],
    start_dt_str: str,
    k: int
) -> list[str]:
    """Retrieves k file paths with datetime values greater than or equal to the
    given start datetime.

    The function first converts each file path into a datetime using
    `parse_file_datetime`, sorts the file paths by this datetime, and then
    selects the first k file paths where the file's datetime is on or after the
    start datetime.

    Args:
        file_paths (list[str]): A list of file path strings.
        start_dt_str (str): The start datetime as a string in the format
            'YYYYMMDD HHMMSS'.
        k (int): The number of file paths to retrieve.

    Returns:
        list[str]: A list of file paths meeting the datetime criteria.

    Raises:
        ValueError: If the start datetime string is not in the correct format.
    """
    try:
        start_dt = datetime.strptime(start_dt_str, "%Y%m%d %H%M%S")
    except ValueError as ve:
        raise ValueError(
            "start_dt_str must be in 'YYYYMMDD HHMMSS' format.") from ve

    # Create a list of tuples (file_datetime, file_path)
    file_dt_pairs = []
    for path in file_paths:
        try:
            file_dt = parse_file_datetime(path)
            file_dt_pairs.append((file_dt, path))
        except ValueError:
            # Optionally, you can log or handle files that do not match the
            # expected format.
            continue

    # Sort the list by the datetime value.
    file_dt_pairs.sort(key=lambda x: x[0])

    # Retrieve the first k file paths where the file datetime is >= start_dt.
    selected_files = [path for dt, path in file_dt_pairs if dt >= start_dt]
    return selected_files[:k]


def get_hdf5_file_paths_range(
    exp_path: str,
    start: Optional[str] = None,
    duration: Optional[float] = None,
    end: Optional[str] = None
) -> list[str]:
    """Collects HDF5 file paths within a specified time range.

    This function searches for and retrieves all HDF5 files from the given
    experiment path (`exp_path`) that fall within the specified start
    (`start_dt`) and end (`end_dt`) datetime range. The first file corresponds
    to `start_dt` or later, and the last file is the closest one before
    `end_dt`. This function also includes the previous file if the start time
    is not an integer, i.e. the file does not start at exactly second.

    Args:
        exp_path (str): The experiment path.
        start (Optional[str]): The start time in the format 'YYYYMMDD HHMMSS'
            or other ISO 8601 formats. Default is None.
        duration (Optional[float]): The duration in seconds. Default is None.
        end (Optional[str]): The end time in the format 'YYYYMMDD HHMMSS' or
            other ISO 8601 formats. This is exclusive. Default is None.

    Returns:
        list[str]: A list of hdf5 file.
    """
    # Validate and calculate start_dt and end_dt
    start_dt, end_dt = _calculate_time_range(start, duration, end)

    # Collect files in the specified time range
    file_paths = _collect_files_in_range(
        exp_path, start_dt, end_dt
    )

    if not file_paths:
        return []

    # Check if the first file's t_start is not integer and if the second file
    # matches the start time
    if len(file_paths) > 1:
        start_time = start_dt.strftime("%H%M%S")
        second_file_time = os.path.splitext(os.path.basename(file_paths[1]))[0]
        header_info = get_hdf5_header(file_paths[0])
        if (second_file_time == start_time and
                float(header_info.t_start).is_integer()):
            file_paths = file_paths[1:]
    return file_paths
