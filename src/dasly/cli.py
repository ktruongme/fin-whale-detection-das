import time
import logging

import typer

# Set up logger
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

app = typer.Typer(help="Dasly CLI - Fin whale detection with DAS")

DEFAULT_F_MIN = 15.0
DEFAULT_F_MAX = 25.0
DEFAULT_V_MIN = 1_484.0
DEFAULT_V_MAX = 14_844.0
DEFAULT_RMS_WINDOW_SIZE = 0.5
DEFAULT_YOLO_IOU = 0.25
DEFAULT_HYPERBOLAS_NUM_POINTS = 10
DEFAULT_HYPERBOLAS_BY_CHANNEL = True


@app.command()
def whales(
    exp_path: str = typer.Option(..., help="Path to experiment directory"),
    chunk_size: int = typer.Option(
        ..., help="Number of files to process at once"
    ),
    chunk_stride: int = typer.Option(
        ..., help="Number of new files before triggering processing"
    ),
    db_table: str = typer.Option(..., help="Database table name"),
    connection_string: str = typer.Option(
        ..., help="Database connection string"
    ),
    n_start: int | None = typer.Option(
        None,
        help="Start channel index. Defaults to the minimum available channel.",
    ),
    n_end: int | None = typer.Option(
        None,
        help="End channel index (exclusive). Defaults to the full range.",
    ),
    f_min: float = typer.Option(
        DEFAULT_F_MIN, help="Minimum FK filter frequency in Hz"
    ),
    f_max: float = typer.Option(
        DEFAULT_F_MAX, help="Maximum FK filter frequency in Hz"
    ),
    v_min: float = typer.Option(
        DEFAULT_V_MIN, help="Minimum FK filter velocity in m/s"
    ),
    v_max: float = typer.Option(
        DEFAULT_V_MAX, help="Maximum FK filter velocity in m/s"
    ),
    rms_window_size: float = typer.Option(
        DEFAULT_RMS_WINDOW_SIZE,
        help="RMS window size in seconds",
    ),
    yolo_iou: float = typer.Option(
        DEFAULT_YOLO_IOU, help="YOLO IOU threshold"
    ),
    hyperbolas_num_points: int = typer.Option(
        DEFAULT_HYPERBOLAS_NUM_POINTS,
        help="Number of points kept for hyperbola fitting",
    ),
    hyperbolas_by_channel: bool = typer.Option(
        DEFAULT_HYPERBOLAS_BY_CHANNEL,
        help="Select hyperbola fitting points by channel",
    ),
):
    """Run the whale detection pipeline to process HDF5 files in real-time.

    Watches the experiment directory for new HDF5 files and runs the YOLO-based
    fin whale detection pipeline when enough files have accumulated.

    Example:
        dasly whales \\
          --exp-path /path/to/experiment \\
          --chunk-size 6 \\
          --chunk-stride 5 \\
          --db-table events_v1 \\
          --connection-string "postgresql+psycopg2://user:pass@host:5432/db"
    """
    from watchdog.observers import Observer

    from dasly.execution.online import HDF5EventHandler
    from dasly.execution.whale_detection import process_hdf5
    from dasly.loader.fsearcher import get_recent_hdf5_file_paths

    def _process(file_path: str):
        file_paths = get_recent_hdf5_file_paths(
            file_path=file_path,
            num_file=chunk_size
        )
        logger.info(
            "Processing %s files from: %s",
            len(file_paths),
            file_paths[0],
        )
        process_hdf5(
            file_paths=file_paths,
            db_table=db_table,
            connection_string=connection_string,
            n_start=n_start,
            n_end=n_end,
            f_min=f_min,
            f_max=f_max,
            v_min=v_min,
            v_max=v_max,
            rms_window_size=rms_window_size,
            yolo_iou=yolo_iou,
            hyperbolas_num_points=hyperbolas_num_points,
            hyperbolas_by_channel=hyperbolas_by_channel,
        )

    event_handler = HDF5EventHandler(
        event_thresh=chunk_stride,
        process_hdf5=_process,
    )
    observer = Observer()
    observer.schedule(
        event_handler=event_handler,
        path=exp_path,
        recursive=True
    )
    logger.info(f'Watching directory: {exp_path}')
    logger.info(f'Chunk size: {chunk_size}, Chunk stride: {chunk_stride}')
    logger.info(f'DB table: {db_table}')

    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()

    observer.join()


if __name__ == "__main__":
    app()
