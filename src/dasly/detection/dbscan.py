"""Clustering algorithms for DAS data."""

from typing import Callable, TYPE_CHECKING
import logging

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import cdist

from ..execution import box_saver

if TYPE_CHECKING:
    from ..core.dasarray import DASArray

# Set up logger
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def dbscan_points(
    data: np.ndarray[float],
    metric: Callable,
    eps: float,
    min_samples: int
) -> np.ndarray[int]:
    """Apply DBSCAN clustering algorithm to the data.

    Args:
        data (np.ndarray[float]): Data to cluster.
        metric (Callable): Custom distance metric function.
        eps (float): The maximum distance between two samples for one to be
            considered as in the neighborhood of the other.
        min_samples (int): The number of samples in a neighborhood for a point
            to be considered as a core point.

    Returns:
        np.ndarray[int]: Cluster labels for each point.
    """
    # Initialize DBSCAN with the custom distance metric
    dbscan_obj = DBSCAN(
        eps=eps,
        min_samples=min_samples,
        metric=metric
    )
    # Fit DBSCAN on the coordinates
    labels = dbscan_obj.fit_predict(data)
    return labels


def compute_cluster_boxesn(
    points: np.ndarray,
    cluster_labels: np.ndarray,
    quantile_trim: float,
    data_shape: tuple[int, int]
) -> np.ndarray:
    """Aggregate points by cluster and calculate bounding box coordinates
    (norminated).

    Args:
        points (np.ndarray): Array of points to aggregate. Shape (N, 2), where
            N is the number of points.
        cluster_labels (np.ndarray): Cluster labels for each point.
        quantile_trim (float): Proportion to trim the points' coordinates
            before calculating bounding box coordinates.
        data_shape (tuple[int, int]): Shape of the data (height, width).

    Returns:
        np.ndarray: Array of bounding box coordinates for each cluster.
            Shape (M, 4), where M is the number of clusters. The columns are
            (x1n, y1n, x2n, y2n) representing the normalized coordinates of the
            bounding boxes.
    """
    # Create a DataFrame for points and clusters
    points_df = (
        pd.DataFrame(points, columns=['y', 'x'])
        .assign(cluster=cluster_labels)
    )

    # Filter out noise points (cluster label -1) and group by cluster
    boxesn = (
        points_df[points_df['cluster'] != -1]
        .groupby('cluster')
        .agg(
            x1=('x', lambda x: np.quantile(x, quantile_trim)),
            y1=('y', lambda y: np.quantile(y, quantile_trim)),
            x2=('x', lambda x: np.quantile(x, 1 - quantile_trim)),
            y2=('y', lambda y: np.quantile(y, 1 - quantile_trim))
        )
        .assign(
            x1n=lambda df: df['x1'] / data_shape[1],
            y1n=lambda df: df['y1'] / data_shape[0],
            x2n=lambda df: df['x2'] / data_shape[1],
            y2n=lambda df: df['y2'] / data_shape[0]
        )
        .loc[:, ['x1n', 'y1n', 'x2n', 'y2n']]
        .to_numpy()
    )

    return boxesn


def count_point_per_cluster(cluster_labels: np.ndarray) -> np.ndarray[int]:
    """Count the number of points in each cluster.

    Args:
        cluster_labels (np.ndarray): Cluster labels for each point.

    Returns:
        np.ndarray[int]: Number of points in each cluster.
    """
    # Count the number of points in each cluster
    num_points = np.bincount(cluster_labels[cluster_labels != -1])
    return num_points


class DASDbscan:

    def dbscan_points(
        self,
        eps: float,
        min_samples: int,
        target_velocity: float,  # m/s, for scaling the space dimension
        metric: str = 'euclidean',  # 'euclidean', 'cityblock'
        quantile_trim: float = 0.01,
    ) -> 'DASArray':
        """Apply DBSCAN clustering algorithm to the data.

        Args:
            eps (float): The maximum distance between two samples for one to be
                considered as in the neighborhood of the other.
            min_samples (int): The number of samples in a neighborhood for a
                point to be considered as a core point.
            target_velocity (float): The target velocity of the data.
            metric (str): The distance metric to use. The distance function can
                be 'euclidean' or 'cityblock'. Default is 'euclidean'.
            quantile_trim (float): The proportion to trim the points
                coordinates before calculating the x1, y1, x2, y2. Default is
                0.01.

        Returns:
            DASArray: The DASArray object with the clustering results.
        """
        # Prepare the data for clustering
        #######################################################################
        # points[:, 0] is time, points[:, 1] is space dimension
        points = np.argwhere(self == 1)  # Shape (N, 2)
        if len(points) == 0:
            logger.info('No points to cluster.')
            self.meta.update(
                cluster_labels=np.array([], dtype=int),
                boxesn=np.empty((0, 4)),
                boxesp=np.empty((0, 4), dtype=object),
                cluster_sizes=np.array([], dtype=int),
            )
            return self

        scaled_points = points.copy()
        # Convert to float for multiplication
        scaled_points = scaled_points.astype('float')
        # Adjust the scale of the points in the space dimension
        # For example, if the target_velocity is 1500 m/s. We want to scale so
        # 1500 m is equivalent to 1 second (to make the velocity is 45-degree).
        # The corresponding number of pixels in the space dimension is 1500/ds,
        # and the corresponding number of pixels in the time dimension is 1/dt.
        # The scaling factor is (1/dt) / (1500/ds). We also need to adjust the
        # velocity from m/s to channnels/s. So the scaling factor is:
        # (1/dt) / (1500/dx/ds).
        scaled_points[:, 1] *= (
            (1 / self.meta.dt) /
            (target_velocity / self.meta.dx / self.meta.ds))

        if len(scaled_points) > 50_000:
            raise ValueError(f"The number of points ({len(scaled_points):,}) "
                             + "is too large for DBSCAN (max 50,000).")

        distance_matrix = cdist(scaled_points, scaled_points, metric=metric)

        # Apply DBSCAN clustering algorithm
        #######################################################################
        cluster_labels = dbscan_points(
            data=distance_matrix,
            metric='precomputed',
            eps=eps,
            min_samples=min_samples
        )

        boxesn = compute_cluster_boxesn(
            points=points,
            cluster_labels=cluster_labels,
            quantile_trim=quantile_trim,
            data_shape=self.shape
        )

        boxesd = box_saver.denormalize_boxesn(
            boxesn=boxesn,
            t_start=self.meta.timestamps[0],
            t_end=self.meta.timestamps[-1],
            s_start=self.meta.channels[0],
            s_end=self.meta.channels[-1],
        )

        boxesp = box_saver.cast_box_times_to_datetime64(boxes=boxesd)

        cluster_sizes = count_point_per_cluster(cluster_labels)

        # Update the meta attribute
        #######################################################################
        self.meta.update(
            cluster_labels=cluster_labels,
            boxesn=boxesn,
            boxesp=boxesp,
            cluster_sizes=cluster_sizes,
        )

        return self
