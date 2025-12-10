import numpy as np
import pandas as pd


def denormalize_boxesn(
    boxesn: np.ndarray,
    t_start: float,
    t_end: float,
    s_start: float,
    s_end: float,
) -> np.ndarray:
    """Convert detected boxes from normalized coordinates to absolute
    coordinates in the spatial and temporal domains.

    Args:
        boxesn (np.ndarray): Array of shape (n, 4) with rows [s1_norm, t1_norm,
            s2_norm, t2_norm], where s denotes spatial and t temporal values.
        t_start (float): Start timestamp.
        t_end (float): End timestamp.
        s_start (float): Start value for the spatial channel.
        s_end (float): End value for the spatial channel.

    Returns:
        np.ndarray: Array with absolute coordinates.
    """
    boxes = boxesn.astype(np.float64, copy=True)
    boxes[:, 0] = (boxes[:, 0] * (s_end - s_start)) + s_start  # s1
    boxes[:, 2] = (boxes[:, 2] * (s_end - s_start)) + s_start  # s2
    boxes[:, 1] = (boxes[:, 1] * (t_end - t_start)) + t_start  # t1
    boxes[:, 3] = (boxes[:, 3] * (t_end - t_start)) + t_start  # t2
    return boxes


def cast_box_times_to_datetime64(boxes: np.ndarray) -> np.ndarray:
    """Convert temporal columns (t1, t2) in a (n, 4) array to numpy.datetime64
    objects, while leaving spatial columns (s1, s2) unchanged.

    Args:
        boxes (np.ndarray): Array of shape (n, 4) with rows [s1, t1, s2, t2]
            where t1 and t2 are timestamps (in seconds).

    Returns:
        np.ndarray: Object array with t1 and t2 as datetime64 objects.
    """
    boxes_dt = boxes.copy().astype(object)
    boxes_dt[:, 1] = pd.to_datetime(boxes[:, 1], unit='s', utc=True)
    boxes_dt[:, 3] = pd.to_datetime(boxes[:, 3], unit='s', utc=True)
    boxes_dt[:, 0] = boxes[:, 0].astype(np.float64)
    boxes_dt[:, 2] = boxes[:, 2].astype(np.float64)
    return boxes_dt


def build_box_df(
    boxesp: np.ndarray,  # Shape (n, 4)
    boxesn: np.ndarray,  # Shape (n, 4)
    chunk: str,
    chunk_size: int,
    additional: dict = None
) -> pd.DataFrame:
    """Create a DataFrame from detected boxes and additional information.

    Args:
        boxesp (np.ndarray): Array of shape (n, 4) with rows [s1, t1, s2, t2].
        boxesn (np.ndarray): Array of shape (n, 4) with normalized coordinates.
        chunk (str): Chunk identifier.
        chunk_size (int): Size of the chunk (number of files).
        additional (dict, optional): Additional columns to add to the
            DataFrame.
    """
    n = boxesp.shape[0]  # Number of rows

    df = pd.DataFrame({
        's1': boxesp[:, 0],
        't1': boxesp[:, 1],
        's2': boxesp[:, 2],
        't2': boxesp[:, 3],
        'x1n': boxesn[:, 0],
        'y1n': boxesn[:, 1],
        'x2n': boxesn[:, 2],
        'y2n': boxesn[:, 3],
        'chunk': [chunk] * n,
        'chunk_size': [chunk_size] * n
    })
    # Cast spatial columns to float64
    df['s1'] = df['s1'].astype(np.float64)
    df['s2'] = df['s2'].astype(np.float64)

    # Add additional columns to the DataFrame.
    if additional is not None:
        additional_df = pd.DataFrame(additional)
        df = pd.concat([df, additional_df], axis=1)

    return df
