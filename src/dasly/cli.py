import time
import logging

import typer

# Set up logger
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

app = typer.Typer(help="Dasly CLI - Fin whale detection with DAS")


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
):
    """Run the whale detection pipeline to process HDF5 files in real-time.

    Watches the experiment directory for new HDF5 files and runs the YOLO-based
    fin whale detection pipeline when enough files have accumulated.

    Example:
        dasly whales \\
          --exp-path /path/to/experiment \\
          --chunk-size 6 \\
          --chunk-stride 5 \\
          --db-table events_v3 \\
          --connection-string "postgresql+psycopg2://user:pass@host:5432/db"
    """
    from watchdog.observers import Observer

    from dasly.execution.online import HDF5EventHandler
    from dasly.execution.svalbard_yolo import process_hdf5
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
