"""This module contains functions to plot DAS data.

The `heatmap` function plots a heatmap of the input data. The function
automatically sets the colormap and normalization based on the data type. The
function also allows users to customize the plot, such as setting the title,
labels, ticks, colorbar, and Hough transform lines.
"""

from __future__ import annotations

import logging
import io
from collections.abc import Sequence
from typing import Literal

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors, ticker
from matplotlib.image import AxesImage
import matplotlib.patches as patches
import seaborn as sns

sns.set_theme(rc={"grid.linewidth": 0.5})
# Set up logger
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def _check_data_type(data: np.ndarray | pd.DataFrame) -> str:
    """Infer an appropriate data category for colormap selection."""
    if isinstance(data, pd.DataFrame):
        array = data.to_numpy()
    else:
        array = np.asarray(data)

    if array.ndim != 2:
        raise ValueError("Input data must be two-dimensional.")

    finite_mask = np.isfinite(array)
    if not finite_mask.any():  # Degenerate case, treat as float data
        return 'float'

    # Binary data contains only 0 and 1 (ignoring NaNs)
    if np.all(np.isin(array[finite_mask], (0, 1))):
        return 'binary'

    data_min = np.nanmin(array)
    data_max = np.nanmax(array)

    if data_min == 0 and data_max == 255:
        return 'gray'
    if data_min >= 0:
        return 'positive'
    return 'float'


def _ensure_dataframe(data: pd.DataFrame | np.ndarray) -> pd.DataFrame:
    """Return a DataFrame representing the input data without mutating it."""
    if isinstance(data, pd.DataFrame):
        return data.copy()

    array = np.asarray(data)
    if array.ndim != 2:
        raise ValueError("Input data must be a two-dimensional structure.")
    return pd.DataFrame(array)


def _apply_log_scale(data: pd.DataFrame, enabled: bool) -> pd.DataFrame:
    """Apply a base-10 logarithmic transform when requested."""
    if not enabled:
        return data
    transformed = data.astype(float) + 1e-12
    return transformed.applymap(np.log10)


def _determine_color_settings(
    data: pd.DataFrame,
    vmin: float | None,
    vmax: float | None,
    percentile: float,
    log_scale: bool
) -> tuple[
    str,
    colors.Colormap | str,
    colors.Normalize | None,
    float | None,
    float | None,
]:
    """Return colormap and normalization configured for the data."""
    data_type = _check_data_type(data)

    array = data.to_numpy(dtype=float, copy=False)

    local_vmin = vmin
    local_vmax = vmax

    if data_type == 'binary':
        cmap: colors.Colormap | str = colors.ListedColormap(['black', 'white'])
        local_vmin, local_vmax = 0.0, 1.0
    elif data_type in ('gray', 'positive'):
        cmap = 'gray' if data_type == 'gray' else 'viridis'
        if local_vmin is None:
            local_vmin = 0.0
        if local_vmax is None:
            percentile_value = float(np.nanquantile(array, percentile))
            local_vmax = percentile_value
            logger.info(f'vmax: {local_vmax:.3g}')
    else:
        cmap = 'RdBu'
        if local_vmin is None or local_vmax is None:
            percentile_value = float(np.nanquantile(np.abs(array), percentile))
            local_vmin = -percentile_value
            local_vmax = percentile_value
            logger.info(f'vmin: {local_vmin:.3g}, vmax: {local_vmax:.3g}')

    if log_scale:
        norm: colors.Normalize | None = None
    elif data_type in ('gray', 'positive', 'binary'):
        norm = colors.Normalize(vmin=local_vmin, vmax=local_vmax)
    else:
        norm = colors.TwoSlopeNorm(vmin=local_vmin, vcenter=0, vmax=local_vmax)

    return data_type, cmap, norm, local_vmin, local_vmax


def _resolve_aspect(aspect: float | None, data: pd.DataFrame) -> float:
    if aspect is not None:
        return aspect
    rows, cols = data.shape
    return cols / rows if rows else 1.0


def _create_heatmap(
    data: pd.DataFrame,
    figsize: tuple[float, float],
    aspect: float,
    cmap: colors.Colormap | str,
    norm: colors.Normalize | None,
    interpolation: str
) -> tuple[plt.Figure, plt.Axes, AxesImage]:
    fig, ax = plt.subplots(figsize=figsize)
    image = ax.imshow(
        data.to_numpy(),
        aspect=aspect,
        cmap=cmap,
        norm=norm,
        interpolation=interpolation,
        origin='lower'
    )
    return fig, ax, image


def _resolve_color(rc_key: str) -> str | None:
    color = plt.rcParams.get(rc_key)
    if isinstance(color, str) and color.lower() in {'auto', 'inherit'}:
        return None
    return color


def _resolve_text(
    value: str | None,
    default: str
) -> tuple[str, bool]:
    if value is not None:
        return value, False
    return default, True


def _resolve_weight(
    explicit: str | None,
    rc_keys: Sequence[str],
    default: str = 'bold'
) -> str:
    if explicit is not None:
        return explicit

    for key in rc_keys:
        value = plt.rcParams.get(key)
        if value is None:
            continue
        if isinstance(value, str) and value.lower() in {'auto', 'inherit'}:
            continue
        default_value = plt.rcParamsDefault.get(key)
        if default_value is None:
            default_value = plt.rcParamsDefault.get('font.weight')
        if value != default_value:
            return value

    for key in rc_keys:
        value = plt.rcParams.get(key)
        if value is None:
            continue
        if isinstance(value, str) and value.lower() in {'auto', 'inherit'}:
            continue
        return value

    global_weight = plt.rcParams.get('font.weight')
    default_global = plt.rcParamsDefault.get('font.weight')
    if (
        global_weight is not None
        and global_weight != default_global
    ):
        return global_weight

    return default


def _adjust_xlabel_for_km(label: str, use_km: bool, used_default: bool) -> str:
    if use_km and used_default and label == 'Channel Distance (m)':
        return 'Channel Distance (km)'
    return label


def _set_axis_labels(
    ax: plt.Axes,
    title: str,
    titlesize: float | None,
    xlabel: str,
    ylabel: str,
    labelsize: float | None,
    ticksize: float | None,
    titleweight: str,
    labelweight: str
) -> None:
    title_kwargs: dict[str, float | str] = {}
    if titlesize is not None:
        title_kwargs['fontsize'] = titlesize
    title_kwargs['fontweight'] = titleweight
    title_color = _resolve_color('axes.titlecolor')
    if title_color is not None:
        title_kwargs['color'] = title_color
    ax.set_title(title, **title_kwargs)

    label_kwargs: dict[str, float | str] = {}
    if labelsize is not None:
        label_kwargs['fontsize'] = labelsize
    label_kwargs['fontweight'] = labelweight
    label_color = _resolve_color('axes.labelcolor')
    if label_color is not None:
        label_kwargs['color'] = label_color
    ax.set_xlabel(xlabel, **label_kwargs)
    ax.set_ylabel(ylabel, **label_kwargs)

    if ticksize is not None:
        ax.tick_params(axis='both', which='major', labelsize=ticksize)


def _apply_tick_locators(
    ax: plt.Axes,
    xticks_gap: int | None,
    yticks_gap: int | None
) -> None:
    if xticks_gap is not None:
        ax.xaxis.set_major_locator(
            ticker.MultipleLocator(xticks_gap)
        )
    if yticks_gap is not None:
        ax.yaxis.set_major_locator(
            ticker.MultipleLocator(yticks_gap)
        )


def _format_numeric_label(value: float, use_km: bool) -> str:
    scaled = value / 1000.0 if use_km else value
    return str(int(round(scaled)))


def _tick_label_kwargs(
    axis: Literal['x', 'y'],
    rotation: float | str | None,
    weight: str
) -> dict[str, float | str]:
    kwargs: dict[str, float | str] = {}
    rotation_value = rotation
    if rotation_value is None:
        rotation_value = plt.rcParams.get(
            f'{axis}tick.labelrotation',
            None
        )
    if rotation_value is not None:
        kwargs['rotation'] = rotation_value

    tick_color = _resolve_color(f'{axis}tick.color')
    if tick_color is not None:
        kwargs['color'] = tick_color

    kwargs['fontweight'] = weight
    return kwargs


def _set_tick_labels(
    ax: plt.Axes,
    axis: Literal['x', 'y'],
    positions: np.ndarray,
    labels: Sequence[str],
    rotation: float | str | None,
    tick_weight: str
) -> None:
    if axis == 'x':
        ax.set_xticks(positions)
        setter = ax.set_xticklabels
    else:
        ax.set_yticks(positions)
        setter = ax.set_yticklabels

    label_kwargs = _tick_label_kwargs(axis, rotation, tick_weight)
    setter(labels, **label_kwargs)


def _set_items_fontweight(items: Sequence[plt.Text], weight: str) -> None:
    for item in items:
        item.set_fontweight(weight)


def _apply_xtick_labels(
    ax: plt.Axes,
    data: pd.DataFrame,
    xticks_gap: int | None,
    xticks_labels: Sequence[str] | None,
    x_km: bool,
    rotation: float | str | None,
    tick_weight: str
) -> None:
    if xticks_gap is not None and xticks_labels is not None:
        max_pos = data.shape[1] - 1
        positions = np.arange(
            0,
            len(xticks_labels) * xticks_gap,
            xticks_gap,
        )
        positions = positions[positions <= max_pos]
        labels = list(xticks_labels[:len(positions)])
        _set_tick_labels(
            ax,
            'x',
            positions,
            labels,
            rotation,
            tick_weight
        )
        return

    current_xticks = ax.get_xticks()
    valid_positions = current_xticks[
        (current_xticks >= 0)
        & (current_xticks < data.shape[1])
    ]
    indices = valid_positions.astype(int)

    labels: list[str] = []
    numeric_values = pd.to_numeric(data.columns.take(indices), errors='coerce')
    for numeric, original in zip(numeric_values, data.columns.take(indices)):
        if pd.isna(numeric):
            labels.append(str(original))
        else:
            labels.append(_format_numeric_label(float(numeric), x_km))

    _set_tick_labels(
        ax,
        'x',
        valid_positions,
        labels,
        rotation,
        tick_weight
    )


def _format_index_labels(values: pd.Index, yformat: str) -> list[str]:
    if isinstance(values, pd.DatetimeIndex):
        return [ts.strftime(yformat) for ts in values]

    try:
        converted = pd.to_datetime(values, errors='coerce')
    except (TypeError, ValueError):
        return [str(v) for v in values]

    labels: list[str] = []
    for original, converted_value in zip(
        values,
        converted,
    ):
        if pd.isna(converted_value):
            labels.append(str(original))
        else:
            labels.append(converted_value.strftime(yformat))
    return labels


def _compute_elapsed_seconds(
    index: pd.Index,
    positions: np.ndarray,
) -> np.ndarray | None:
    if isinstance(index, pd.DatetimeIndex):
        base = index[0]
        return (index.take(positions) - base).total_seconds()

    try:
        converted = pd.to_datetime(index, errors='raise')
    except (TypeError, ValueError):
        return None

    base = converted[0]
    return (converted.take(positions) - base).total_seconds()


def _apply_ytick_labels(
    ax: plt.Axes,
    data: pd.DataFrame,
    y_seconds_from_start: bool,
    yformat: str,
    rotation: float | str | None,
    tick_weight: str
) -> None:
    current_yticks = ax.get_yticks()
    valid_positions = current_yticks[
        (current_yticks >= 0)
        & (current_yticks < data.shape[0])
    ]
    indices = valid_positions.astype(int)

    if not len(indices):
        return

    index = data.index
    if y_seconds_from_start:
        seconds = _compute_elapsed_seconds(index, indices)
        if seconds is not None:
            labels = [str(int(round(sec))) for sec in seconds]
            _set_tick_labels(
                ax,
                'y',
                valid_positions,
                labels,
                rotation,
                tick_weight
            )
            return

    selected = index.take(indices)
    labels = _format_index_labels(
        pd.Index(selected),
        yformat,
    )
    _set_tick_labels(
        ax,
        'y',
        valid_positions,
        labels,
        rotation,
        tick_weight
    )


def _add_binary_point_overlay(
    ax: plt.Axes,
    data: pd.DataFrame,
    data_type: str,
    marker_size: float | None,
    base_zorder: float,
) -> None:
    """Overlay white scatter markers for binary heatmaps when requested."""
    if data_type != 'binary':
        return
    if marker_size is None or marker_size <= 0:
        return

    array = data.to_numpy(dtype=float, copy=False)
    ones = np.argwhere(np.isclose(array, 1.0))
    if ones.size == 0:
        return

    ax.scatter(
        ones[:, 1],
        ones[:, 0],
        s=marker_size,
        facecolors='white',
        edgecolors='none',
        linewidths=0,
        zorder=base_zorder + 1,
    )


def _add_hough_lines(
    ax: plt.Axes,
    lines: Sequence[tuple[float, float, float, float]] | None,
    lineclusters: Sequence[int] | None,
    linestyle: str,
    linewidth: float,
    cmap_name: str,
) -> None:
    if lines is None:
        return

    lines_list = list(lines)
    if len(lines_list) == 0:
        return

    if lineclusters is not None:
        lineclusters_list = list(lineclusters)
        if len(lineclusters_list) != len(lines_list):
            raise ValueError(
                "`lineclusters` must match the number of `lines`."
            )
        cmap = plt.get_cmap(cmap_name)
        unique_clusters = np.unique(lineclusters_list)
        color_lookup = {
            cluster: cmap(i % cmap.N)
            for i, cluster in enumerate(unique_clusters)
        }
        for (x1, y1, x2, y2), cluster in zip(
            lines_list,
            lineclusters_list,
        ):
            ax.plot(
                [x1, x2],
                [y1, y2],
                linestyle=linestyle,
                linewidth=linewidth,
                color=color_lookup[cluster],
            )
        return

    for x1, y1, x2, y2 in lines_list:
        ax.plot(
            [x1, x2],
            [y1, y2],
            linestyle=linestyle,
            linewidth=linewidth,
        )


def _add_point_clusters(
    ax: plt.Axes,
    data: pd.DataFrame,
    clusters: Sequence[int] | None,
    scatter_size: float,
    clusters_legend: bool,
    markersize: float,
    fontsize: float | None,
    cmap_name: str
) -> None:
    if clusters is None:
        return

    points = np.argwhere(data.to_numpy() == 1)
    cluster_array = np.asarray(clusters)
    if len(cluster_array) != len(points):
        raise ValueError("Number of clusters does not match number of points.")

    mask = cluster_array != -1
    if not np.any(mask):
        return

    x_filtered = points[:, 1][mask]
    y_filtered = points[:, 0][mask]
    clusters_filtered = cluster_array[mask]

    cmap = plt.get_cmap(cmap_name)
    num_colors = cmap.N
    colors_plot = [cmap(label % num_colors) for label in clusters_filtered]

    ax.scatter(
        x_filtered,
        y_filtered,
        c=colors_plot,
        s=scatter_size,
        edgecolor='none',
        label='Clusters'
    )

    if not clusters_legend:
        return

    unique_labels = np.unique(clusters_filtered)
    handles = [
        plt.Line2D(
            [], [],
            marker='o',
            linestyle='None',
            color=cmap(label % num_colors),
            markersize=markersize,
            label=f'Cluster {label}'
        )
        for label in unique_labels
    ]

    if handles:
        legend_kwargs: dict[str, float | str] = {}
        if fontsize is not None:
            legend_kwargs['fontsize'] = fontsize
        ax.legend(handles=handles, **legend_kwargs)


def _add_colorbar(
    fig: plt.Figure,
    ax: plt.Axes,
    image: AxesImage,
    data_type: str,
    enabled: bool,
    label: str,
    labelsize: float | None,
    ticksize: float | None,
    labelweight: str,
    tick_weight: str
) -> None:
    if not enabled:
        return

    pos = ax.get_position()
    cbar_ax = fig.add_axes([pos.x1 + 0.03, pos.y0, 0.03, pos.height])

    if data_type == 'binary':
        cbar = fig.colorbar(image, cax=cbar_ax, ticks=[0.25, 0.75])
        cbar.ax.set_yticklabels(['0', '1'])
    else:
        cbar = fig.colorbar(image, cax=cbar_ax)

    label_kwargs: dict[str, float | str] = {}
    if labelsize is not None:
        label_kwargs['fontsize'] = labelsize
    label_kwargs['fontweight'] = labelweight
    label_color = _resolve_color('axes.labelcolor')
    if label_color is not None:
        label_kwargs['color'] = label_color
    cbar.set_label(label, **label_kwargs)

    if ticksize is not None:
        cbar.ax.tick_params(labelsize=ticksize)

    _set_items_fontweight(cbar.ax.get_xticklabels(), tick_weight)
    _set_items_fontweight(cbar.ax.get_yticklabels(), tick_weight)

    offset_text = cbar.ax.yaxis.get_offset_text()
    offset_text.set_ha('left')


def plot(
    data: pd.DataFrame,
    log_scale: bool = False,
    vmin: float | None = None,
    vmax: float | None = None,
    percentile: float = 0.95,
    figsize: tuple[float, float] = (5, 5),
    aspect: float | None = None,
    interpolation: str = 'none',
    title: str | None = None,
    titlesize: float | None = None,
    titleweight: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    labelsize: float | None = None,
    labelweight: str | None = None,
    yformat: str = '%H:%M:%S',
    y_seconds_from_start: bool = False,
    x_km: bool = False,
    xticks_rotate: float | str | None = None,
    yticks_rotate: float | str | None = None,
    ticksize: float | None = None,
    tickweight: str | None = None,
    xticks_gap: int | None = None,
    xticks_labels: Sequence[str] | None = None,
    yticks_gap: int | None = None,
    colorbar: bool = True,
    colorbar_label: str | None = 'Value',
    colorbar_labelsize: float | None = None,
    colorbar_labelweight: str | None = None,
    colorbar_ticksize: float | None = None,
    binary_marker_size: float | None = None,
    lines: Sequence[tuple[float, float, float, float]] | None = None,
    linestyle: str = '--',
    linewidth: float = 3,
    lineclusters: Sequence[int] | None = None,
    clusters: Sequence[int] | None = None,
    clusters_scatter_size: float = 2,
    clusters_legend: bool = True,
    clusters_legend_markersize: float = 10,
    clusters_legend_fontsize: float | None = None,
    clusters_cmap: str = 'tab10',
    show: bool = True
) -> tuple[plt.Figure, plt.Axes]:
    """Plot a heatmap of DAS data with optional overlays.

    The input is normalised automatically when `vmin`/`vmax` are not provided.
    For datetime-like indexes, tick labels are formatted with ``yformat``; for
    other index types, labels fall back to ``str`` representations. When
    ``y_seconds_from_start`` is requested but timestamps cannot be inferred,
    the formatter automatically falls back to ``yformat``.

    Args:
        data (pd.DataFrame): Input data.
        log_scale (bool, optional): Apply base-10 logarithmic scaling before
            plotting.
        vmin (float, optional): Minimum value for the heatmap normalisation.
        vmax (float, optional): Maximum value for the heatmap normalisation.
        percentile (float, optional): Percentile used to estimate ``vmin`` and
            ``vmax`` when they are not supplied.
        figsize (tuple[float, float], optional): Size of the created figure in
            inches.
        aspect (float, optional): Override the aspect ratio passed to
            ``imshow``.
        interpolation (str, optional): Interpolation method passed to
            ``imshow``.
        title (str, optional): Plot title. ``None`` falls back to ``'DAS'``.
        titlesize (float, optional): Title font size. ``None`` preserves the
            value from ``matplotlib.rcParams``.
        titleweight (str, optional): Title font weight. Defaults to
            bold unless overridden by ``rcParams`` or this argument.
        xlabel (str, optional): X-axis label. ``None`` falls back to
            ``'Channel Distance (m)'``.
        ylabel (str, optional): Y-axis label.
            ``None`` falls back to ``'Time (UTC)'``.
        labelsize (float, optional): Axis label font size. ``None`` preserves
            ``rcParams``.
        labelweight (str, optional): Axis label font weight. Defaults
            to bold unless overridden.
        yformat (str, optional): Datetime format string for y-axis tick labels.
        y_seconds_from_start (bool, optional): Show elapsed seconds instead of
            absolute timestamps for datetime indexes.
        x_km (bool, optional): Convert numeric x tick labels from metres to
            kilometres.
        xticks_rotate (float | str, optional): Rotation applied to x tick
            labels. ``None`` defers to ``rcParams['xtick.labelrotation']``.
        yticks_rotate (float | str, optional): Rotation applied to y tick
            labels. ``None`` defers to ``rcParams['ytick.labelrotation']``.
        ticksize (float, optional): Tick label font size. ``None`` preserves
            ``rcParams``.
        tickweight (str, optional): Tick label font weight. Defaults
            to bold unless overridden.
        xticks_gap (int, optional): Spacing between x ticks when using fixed
            tick intervals.
        xticks_labels (Sequence[str], optional): Explicit labels used when
            ``xticks_gap`` is provided.
        yticks_gap (int, optional): Spacing between y ticks when using fixed
            tick intervals.
        colorbar (bool, optional): Display a colorbar when ``True``.
        colorbar_label (str, optional): Label for the colorbar. Defaults
            to ``'Value'`` when unspecified.
        colorbar_labelsize (float, optional): Colorbar label font size.
            ``None`` preserves ``rcParams``.
        colorbar_labelweight (str, optional): Colorbar label font weight.
            Defaults to bold unless overridden.
        colorbar_ticksize (float, optional): Colorbar tick font size. ``None``
            preserves ``rcParams``.
        binary_marker_size (float, optional): Marker size for white overlays on
            binary data. ``None`` or non-positive values disable the overlay.
        lines (Sequence[tuple[float, float, float, float]], optional): Lines to
            overlay on the image.
        linestyle (str, optional): Line style for overlays.
        linewidth (float, optional): Line width for overlays.
        lineclusters (Sequence[int], optional): Cluster ids for the supplied
            lines.
        clusters (Sequence[int], optional): Cluster assignments for binary
            points in ``data``.
        clusters_scatter_size (float, optional): Scatter marker size for
            cluster points.
        clusters_legend (bool, optional): Whether to add a legend for clusters.
        clusters_legend_markersize (float, optional): Marker size within the
            cluster legend.
        clusters_legend_fontsize (float, optional): Legend font size for
            clusters. ``None`` preserves ``rcParams``.
        clusters_cmap (str, optional): Colormap used for cluster overlays.
        show (bool, optional): Display the figure immediately when ``True``.

    Returns:
        tuple[plt.Figure, plt.Axes]: Figure and Axes objects hosting the plot.
    """

    frame = _ensure_dataframe(data)
    plot_data = _apply_log_scale(frame, log_scale)
    data_type, cmap, norm, _vmin, _vmax = _determine_color_settings(
        plot_data, vmin, vmax, percentile, log_scale)

    aspect_ratio = _resolve_aspect(aspect, plot_data)
    fig, ax, image = _create_heatmap(
        plot_data,
        figsize=figsize,
        aspect=aspect_ratio,
        cmap=cmap,
        norm=norm,
        interpolation=interpolation
    )

    if norm is None and (_vmin is not None or _vmax is not None):
        image.set_clim(vmin=_vmin, vmax=_vmax)

    title_text, _ = _resolve_text(title, 'DAS')
    xlabel_text, xlabel_default = _resolve_text(xlabel, 'Channel Distance (m)')
    xlabel_text = _adjust_xlabel_for_km(xlabel_text, x_km, xlabel_default)
    ylabel_text, _ = _resolve_text(ylabel, 'Time (UTC)')
    colorbar_label_text, _ = _resolve_text(colorbar_label, 'Value')
    title_weight_resolved = _resolve_weight(
        titleweight,
        ('axes.titleweight',),
        'bold'
    )
    label_weight_resolved = _resolve_weight(
        labelweight,
        ('axes.labelweight',),
        'bold'
    )
    tick_weight_resolved = _resolve_weight(
        tickweight,
        ('font.weight',),
        'bold'
    )
    colorbar_label_weight_resolved = _resolve_weight(
        colorbar_labelweight,
        ('axes.labelweight',),
        label_weight_resolved
    )

    _set_axis_labels(
        ax,
        title=title_text,
        titlesize=titlesize,
        xlabel=xlabel_text,
        ylabel=ylabel_text,
        labelsize=labelsize,
        ticksize=ticksize,
        titleweight=title_weight_resolved,
        labelweight=label_weight_resolved
    )

    _apply_tick_locators(ax, xticks_gap=xticks_gap, yticks_gap=yticks_gap)
    _apply_xtick_labels(
        ax,
        data=plot_data,
        xticks_gap=xticks_gap,
        xticks_labels=xticks_labels,
        x_km=x_km,
        rotation=xticks_rotate,
        tick_weight=tick_weight_resolved
    )
    _apply_ytick_labels(
        ax,
        data=plot_data,
        y_seconds_from_start=y_seconds_from_start,
        yformat=yformat,
        rotation=yticks_rotate,
        tick_weight=tick_weight_resolved
    )

    _add_binary_point_overlay(
        ax,
        data=plot_data,
        data_type=data_type,
        marker_size=binary_marker_size,
        base_zorder=image.get_zorder(),
    )

    _add_hough_lines(
        ax,
        lines=lines,
        lineclusters=lineclusters,
        linestyle=linestyle,
        linewidth=linewidth,
        cmap_name=clusters_cmap
    )

    _add_point_clusters(
        ax,
        data=plot_data,
        clusters=clusters,
        scatter_size=clusters_scatter_size,
        clusters_legend=clusters_legend,
        markersize=clusters_legend_markersize,
        fontsize=clusters_legend_fontsize,
        cmap_name=clusters_cmap
    )

    _add_colorbar(
        fig,
        ax,
        image,
        data_type=data_type,
        enabled=colorbar,
        label=colorbar_label_text,
        labelsize=colorbar_labelsize,
        ticksize=colorbar_ticksize,
        labelweight=colorbar_label_weight_resolved,
        tick_weight=tick_weight_resolved
    )

    if not show:
        plt.close(fig)

    return fig, ax


def plot_hyperbolas(
    fig: plt.Figure,
    hyperbolas: np.ndarray,
    num_points: int = 1_000,
    colormap: str = 'tab10',
    linewidth: float = 1.5,
    branch: str = 'upper',
    **kwargs
) -> plt.Figure:
    """Plot the upper branches of hyperbolas on a given figure.

    Args:
        fig (plt.Figure): The figure to plot the hyperbolas on.
        hyperbolas (np.ndarray): Array having shape (N, 4) of hyperbolas in the
            form (a, b, h, k).
        num_points (int, optional): Number of points to generate for each
            hyperbola. Defaults to 1_000.
        colormap (str, optional): Name of the colormap to use for the hyperbola
            colors. Defaults to 'tab10'.
        linewidth (float, optional): Line width of the hyperbolas. Defaults to
            1.5.
        branch (str, optional): Branch of the hyperbola to plot, either 'upper'
            or 'lower'. Defaults to 'upper'.
        **kwargs: Additional keyword arguments to pass to the plot function.

    Raises:
        ValueError: If the hyperbolas array is not a 2D array with shape
            (N, 4).
        ValueError: If the figure does not contain any Axes.

    Returns:
        plt.Figure: The input figure with the hyperbolas plotted.
    """
    # Ensure hyperbolas is a NumPy array
    hyperbolas = np.asarray(hyperbolas)
    if hyperbolas.ndim != 2 or hyperbolas.shape[1] != 4:
        raise ValueError("hyperbolas must be a 2D array with shape (N, 4)")

    # Get the current limits of the Axes
    ax = fig.get_axes()[0]
    data_shape = ax.get_images()[0].get_array().shape
    x_min, x_max = 1, data_shape[1] - 1  # Avoid mutiple axes appearance
    y_min, y_max = 1, data_shape[0] - 1  # Avoid mutiple axes appearance

    # Get colors from the specified colormap
    cmap = plt.get_cmap(colormap)
    num_colors = cmap.N
    colors = [cmap(i % num_colors) for i in range(len(hyperbolas))]

    # Iterate over each hyperbola and plot
    for idx, ((a, b, h, k), color) in enumerate(zip(hyperbolas, colors)):
        if a == 0 or b == 0:
            logger.info(f"Skipping hyperbola at index {idx} due to zero a or" +
                        " b value.")
            continue  # Avoid division by zero

        # Generate x values within the axes limits
        x = np.linspace(x_min, x_max, num_points)

        # Compute the corresponding y values for the specified branch
        try:
            if branch == 'upper':
                y = k + b * np.sqrt(1 + ((x - h) / a) ** 2)
            else:  # branch == 'lower'
                y = k - b * np.sqrt(1 + ((x - h) / a) ** 2)
        except FloatingPointError:
            logger.info("Numerical issue encountered for hyperbola at index " +
                        f"{idx}. Skipping.")
            continue

        # Determine which points are within the y-limits
        valid = (y >= y_min) & (y <= y_max)
        x_valid = x[valid]
        y_valid = y[valid]

        # Access the first Axes in the Figure
        if not fig.axes:
            raise ValueError("The figure does not contain any Axes.")
        ax = fig.axes[0]

        # Plot the valid portion of the hyperbola
        ax.plot(x_valid, y_valid, color=color, linewidth=linewidth, **kwargs)

    # Optionally, redraw the figure to update the plot
    fig.canvas.draw()

    return fig


def show_figures(
    figures: list[plt.Figure],
    layout: tuple[int, int] = None,
    width: int = 5,
    height: int = 5
) -> None:
    """Display multiple figures side by side.

    Args:
        figures (list): List of figures to display.
        layout (tuple, optional): Layout of the figures (nrows, ncols). If
            None, defaults to a single row with all figures side by side.
            Defaults to None.
        width (int, optional): Width of each figure in inches. Defaults to 5.
        height (int, optional): Height of each figure in inches. Defaults to 5.

    Raises:
        ValueError: If the layout is too small for the number of figures.

    Returns:
        None
    """
    n_figures = len(figures)

    if layout is None:
        # Default to one row with all figures side by side
        nrows, ncols = 1, n_figures
    else:
        if not isinstance(layout, tuple) or len(layout) != 2:
            raise ValueError("`layout` must be a tuple of two integers, e.g., "
                             + "(2, 3).")
        nrows, ncols = layout
        if nrows * ncols < n_figures:
            raise ValueError(f"Layout {layout} is too small for {n_figures} "
                             + "figures.")

    # Create a new figure with the specified layout
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(width * ncols, height * nrows))

    # If there's only one subplot, make axes iterable
    if nrows * ncols == 1:
        axes = [axes]
    else:
        # Flatten the axes array for easy iteration
        axes = axes.flatten()

    for idx, fig_i in enumerate(figures):
        ax = axes[idx]

        # Save the individual figure to a BytesIO buffer
        buf = io.BytesIO()
        fig_i.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)

        # Read the image from the buffer
        img = plt.imread(buf)

        # Display the image on the current axis
        ax.imshow(img)
        ax.axis('off')  # Hide axes ticks and labels

    # Turn off any unused subplots
    for idx in range(n_figures, nrows * ncols):
        axes[idx].axis('off')

    plt.tight_layout()
    plt.show()


def save_figure(
    figure: plt.Figure,
    axes: plt.Axes,
    filename: str,
    dpi: int | str = 'figure'
) -> None:
    """Save a figure to a file with tight bounding box and no padding. No axes,
    colorbars, titles, or labels will be included in the saved figure.

    Args:
        figure (plt.Figure): The figure to save.
        axes (plt.Axes): The main axes of the figure.
        filename (str): The name of the file to save the figure to.
        dpi (int | str, optional): The resolution of the saved figure. If
            'figure', uses the DPI of the input figure. Defaults to 'figure'.

    Returns:
        None
    """
    # 1. Remove Axes
    axes.axis('off')  # Hides the axes (ticks, labels, etc.)

    # 2. Remove Colorbars (if any)
    # It's common for colorbars to be separate axes within the figure.
    # We'll iterate through all axes and remove any that are not the main plot.

    # Identify and remove colorbar axes
    for ax in figure.axes:
        if ax is not axes:
            # Optionally, check if the axes is a colorbar by its properties
            # For example, colorbars often have a different aspect ratio
            # Here, we assume any additional axes are colorbars
            ax.remove()

    # 3. Remove Titles and Labels (if any)
    axes.set_title('')      # Removes the title
    axes.set_xlabel('')     # Removes the x-axis label
    axes.set_ylabel('')     # Removes the y-axis label

    # 4. Save the Figure
    # Save with tight bounding box and no padding
    figure.savefig(
        filename,
        bbox_inches='tight',
        pad_inches=0,
        transparent=True,  # Set to False if you prefer a solid background
        dpi=dpi
    )

    # 5. Close the Figure (optional but recommended to free up memory)
    plt.close(figure)


def add_boxes(
    fig: plt.Figure,
    boxesn: np.ndarray,
    line_width: int = 2,
    color: str = 'red',
    num_boxes: bool = False
) -> None:
    """Adds bounding boxes (in normalized coordinates) to an imshow plot.

    Args:
        fig (plt.Figure): A matplotlib Figure containing an imshow plot.
        boxesn (np.ndarray): Array of shape (N, 4) representing bounding boxes
            in normalized coordinates (x1n, y1n, x2n, y2n).
        line_width (int, optional): Line width for the rectangle. Defaults to
            2.
        color (str, optional): Color of the box. Defaults to 'red'.
        num_boxes (bool, optional): If True, add the index of each box above
            the box. Defaults to False.
    """
    for ax in fig.get_axes():
        images = ax.get_images()
        if not images:
            continue

        # Get image shape (height, width)
        img_shape = images[0].get_array().shape
        height, width = img_shape

        for i, boxn in enumerate(boxesn):
            x1n, y1n, x2n, y2n = boxn
            x1 = x1n * width
            y1 = y1n * height
            x2 = x2n * width
            y2 = y2n * height
            width_box = x2 - x1
            height_box = y2 - y1

            rect = patches.Rectangle(
                (x1, y1), width_box, height_box,
                linewidth=line_width, edgecolor=color, facecolor='none'
            )
            ax.add_patch(rect)

            # Optionally add index above the box
            if num_boxes:
                # Adjust the position of the text to be above the box
                ax.text(
                    x1, y2 + 2, f"{i}",
                    fontsize=12,
                    color='white',
                    ha='left',
                    va='bottom',
                    bbox=dict(facecolor='red', edgecolor='none', pad=1)
                )
    fig.canvas.draw_idle()


class DASPlotter:

    def plot(self, **kwargs):
        return plot(self.to_df(), **kwargs)
