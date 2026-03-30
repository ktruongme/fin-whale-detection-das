from __future__ import annotations
import os
from typing import Callable, Final
import time
import logging

import h5py
from watchdog.events import FileSystemEventHandler

# Set up logger
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def wait_until_complete(
    path: str,
    min_stable: float = 1,
    timeout: float = 10,
    poll: float = 0.25,
) -> None:
    """Block until *path* is a fully written, readable HDF5 file."""
    start: Final[float] = time.time()
    last_size: int = -1
    stable_since: float = start

    while True:
        try:
            size = os.path.getsize(path)
        except FileNotFoundError as exc:
            if time.time() - start > timeout:
                raise exc
            time.sleep(poll)
            continue

        if size != last_size:
            last_size = size
            stable_since = time.time()

        size_is_stable = (
            min_stable == 0 or time.time() - stable_since >= min_stable
        )

        if size_is_stable and size > 0:
            try:
                with h5py.File(path, "r"):
                    return
            except (OSError, IOError, BlockingIOError):
                pass

        if time.time() - start > timeout:
            raise TimeoutError(f"{path!r} not complete after {timeout}s")

        time.sleep(poll)


class HDF5EventHandler(FileSystemEventHandler):

    def __init__(
        self,
        event_thresh: int,
        process_hdf5: Callable[[str], None]
    ):
        """Initialize the event handler with an event threshold.

        Args:
            event_thresh (int): Number of events to wait for before running
                the process_hdf5 function.
            process_hdf5 (Callable[[str], None]): The function to process
                the hdf5 files.
        """
        super().__init__()
        self.event_thresh = event_thresh
        self.event_count = 0
        self.last_created = None
        self.process_hdf5 = process_hdf5

    def on_created(self, event) -> None:
        """Event handler for file creation (for testing)."""
        if (
            event.src_path.endswith('.hdf5') and
            '/dphi/' in event.src_path and
            event.src_path != self.last_created
        ):
            time.sleep(3)
            logger.info(f'New hdf5: {event.src_path}')
            self.last_created = event.src_path
            self.event_count += 1
            if self.event_count >= self.event_thresh:
                logger.info('Running dasly...')
                self.process_hdf5(event.src_path)
                self.event_count = 0

    def on_moved(self, event) -> None:
        """Event handler for file moving (production use)."""
        if (
            event.dest_path.endswith('.hdf5') and
            '/dphi/' in event.dest_path and
            event.dest_path != self.last_created
        ):
            time.sleep(5)
            try:
                wait_until_complete(event.dest_path)
            except TimeoutError as err:
                logger.warning(err)
                return

            logger.info(f'New hdf5: {event.dest_path}')
            self.last_created = event.dest_path
            self.event_count += 1
            if self.event_count >= self.event_thresh:
                logger.info('Running dasly...')
                self.process_hdf5(event.dest_path)
                self.event_count = 0
