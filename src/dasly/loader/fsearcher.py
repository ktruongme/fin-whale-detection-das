"""Helpers to locate and parse DAS HDF5 files for the paper notebooks."""

import os
import re
from datetime import datetime, timedelta, timezone
from typing import Optional
from dataclasses import dataclass

import h5py
import numpy as np


def parse_file_path(file_path: str) -> tuple[str, str, str, str]:
    """Parse file path into components (exp_path, yyyymmdd, hhmmss, filename)."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} does not exist.")
    try:
        exp_path, yyyymmdd, _, hhmmss_hdf5 = file_path.rsplit("/", 3)
        hhmmss = hhmmss_hdf5.removesuffix(".hdf5")
    except ValueError as e:
        raise ValueError(
            "Invalid file path format. Expected <exp_path>/<YYYYMMDD>/dphi/<HHMMSS>.hdf5"
        ) from e
    return exp_path, yyyymmdd, hhmmss, hhmmss_hdf5


@dataclass
class HDF5HeaderInfo:
    ds: float
    s_start: int
    s_end: int
    channels: np.ndarray
    dt: float
    t_start: float
    t_end: float
    timestamps: np.ndarray
    dx: float
    gauge_length: float
    data_scale: float
    sensitivity: float


def get_hdf5_header(file_path: str) -> HDF5HeaderInfo:
    """Read header information from an HDF5 file."""
    with h5py.File(file_path, "r") as hdf_file:
        ds = (hdf_file["header/channels"][()][1] - hdf_file["header/channels"][()][0])
        s_start = hdf_file["header/channels"][()][0]
        s_end = hdf_file["header/channels"][()][-1] + int(ds)
        channels = hdf_file["header/channels"][()]

        dt = hdf_file["header/dt"][()]
        N = hdf_file["header/dimensionRanges/dimension0/size"][()]
        t_start = hdf_file["header/time"][()]
        t_end = t_start + dt * (N - 1)
        timestamps = np.linspace(t_start, t_end, N)

        dx = hdf_file["header/dx"][()]
        gauge_length = hdf_file["header/gaugeLength"][()]
        data_scale = hdf_file["header/dataScale"][()]
        sensitivity = hdf_file["header/sensitivities"][()][0][0]

        return HDF5HeaderInfo(
            ds=ds,
            s_start=s_start,
            s_end=s_end,
            channels=channels,
            dt=dt,
            t_start=t_start,
            t_end=t_end,
            timestamps=timestamps,
            dx=dx,
            gauge_length=gauge_length,
            data_scale=data_scale,
            sensitivity=sensitivity,
        )


def get_all_hdf5_file_paths(directory: str) -> list[str]:
    """Return all HDF5 file paths under a directory (sorted)."""
    hdf5_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".h5") or file.endswith(".hdf5"):
                hdf5_files.append(os.path.join(root, file))
    hdf5_files.sort()
    return hdf5_files


def _normalize_iso_datetime_string(dt_str: str) -> str:
    if dt_str.endswith("Z"):
        dt_str = dt_str[:-1] + "+00:00"
    match = re.match(
        r"^(?P<prefix>.*\d)(?P<fraction>\.\d+)(?P<suffix>(?:Z|[+-]\d{2}:?\d{2})?)$",
        dt_str,
    )
    if not match:
        return dt_str
    fraction = match.group("fraction")[1:]
    padded_fraction = (fraction + "000000")[:6]
    suffix = match.group("suffix") or ""
    return f"{match.group('prefix')}.{padded_fraction}{suffix}"


def _parse_datetime_str(dt_str: str) -> datetime:
    if dt_str is None:
        raise ValueError("Datetime string cannot be None.")
    normalized = _normalize_iso_datetime_string(dt_str.strip())
    try:
        dt = datetime.fromisoformat(normalized)
    except ValueError:
        match = re.fullmatch(r"(?P<date>\d{8}) (?P<time>\d{6})(?P<fraction>\.\d+)?", normalized)
        if not match:
            raise ValueError(
                "Invalid datetime format. Expected ISO 8601 or 'YYYYMMDD HHMMSS[.ffffff]'."
            ) from None
        dt = datetime.strptime(f"{match.group('date')}{match.group('time')}", "%Y%m%d%H%M%S")
        fraction = match.group("fraction")
        if fraction:
            microsecond = int((fraction[1:] + "000000")[:6])
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
    end: Optional[str] = None,
) -> tuple[datetime, datetime]:
    """Validate and calculate start/end datetimes given two of three inputs."""
    provided_args = [start, end, duration]
    if sum(arg is not None for arg in provided_args) != 2:
        raise ValueError("Exactly two of 'start', 'end', or 'duration' must be provided.")

    if start and end:
        start_dt = _parse_datetime_str(start)
        end_dt = _parse_datetime_str(end)
    elif start and duration:
        start_dt = _parse_datetime_str(start)
        end_dt = start_dt + timedelta(seconds=duration)
    else:
        end_dt = _parse_datetime_str(end)
        start_dt = end_dt - timedelta(seconds=duration)

    if start_dt >= end_dt:
        raise ValueError("Start time must be before end time.")
    return start_dt, end_dt


def _get_file_times(date_dir: str) -> list[str]:
    try:
        file_list = os.listdir(date_dir)
    except OSError:
        return []
    file_times = [
        f[:-5] for f in file_list if f.endswith(".hdf5") and f[:-5].isdigit() and len(f[:-5]) == 6
    ]
    return sorted(file_times, reverse=True)


def _collect_files_in_range(
    exp_path: str,
    start_dt: datetime,
    end_dt: datetime,
) -> list[str]:
    """Collect HDF5 files covering the time range [start_dt, end_dt)."""
    collected_files = []
    coverage = timedelta(seconds=10)
    adjusted_start_dt = start_dt - coverage

    date_list = []
    current_date = adjusted_start_dt.date()
    end_date = end_dt.date()
    while current_date <= end_date:
        date_list.append(current_date.strftime("%Y%m%d"))
        current_date += timedelta(days=1)

    for date_str in date_list:
        date_dir = os.path.join(exp_path, date_str, "dphi")
        if not os.path.isdir(date_dir):
            continue

        file_times = _get_file_times(date_dir)
        for hhmmss in file_times:
            file_dt = datetime.strptime(f"{date_str} {hhmmss}", "%Y%m%d %H%M%S").replace(
                tzinfo=timezone.utc
            )
            if adjusted_start_dt <= file_dt < end_dt:
                collected_files.append(os.path.join(date_dir, f"{hhmmss}.hdf5"))

    collected_files.sort()
    return collected_files


def get_hdf5_file_paths_range(
    exp_path: str,
    start: Optional[str] = None,
    duration: Optional[float] = None,
    end: Optional[str] = None,
) -> list[str]:
    """Collect HDF5 file paths within a specified time range."""
    start_dt, end_dt = _calculate_time_range(start, duration, end)
    file_paths = _collect_files_in_range(exp_path, start_dt, end_dt)

    if not file_paths:
        return []

    if len(file_paths) > 1:
        start_time = start_dt.strftime("%H%M%S")
        second_file_time = os.path.splitext(os.path.basename(file_paths[1]))[0]
        header_info = get_hdf5_header(file_paths[0])
        if (second_file_time == start_time and float(header_info.t_start).is_integer()):
            file_paths = file_paths[1:]
    return file_paths
