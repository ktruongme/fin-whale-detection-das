from importlib.resources import as_file, files

from ..core.dasarray import DASArray
from ..fitting.hyperbola_fitter import (
    fit_multiple_hyperbolas_least_squares,
    derive_hyperbola_metrics
)
from .box_saver import save_to_db, build_box_df
from ..loader.fsearcher import parse_file_path

DEFAULT_MODEL_FILENAME = "fin_whale_detection_weights.pt"
DEFAULT_TRAIN_WIDTH = 640
DEFAULT_TRAIN_HEIGHT = 640
DEFAULT_TRAIN_PHYSICAL_WIDTH = 110_000.0
DEFAULT_TRAIN_PHYSICAL_HEIGHT = 30.0
DEFAULT_GRAYSCALE_BY_COLUMN = True


def process_hdf5(
    file_paths: list[str],
    db_table: str,
    connection_string: str,
    *,
    n_start: int | None,
    n_end: int | None,
    f_min: float,
    f_max: float,
    v_min: float,
    v_max: float,
    rms_window_size: float,
    yolo_iou: float,
    hyperbolas_num_points: int,
    hyperbolas_by_channel: bool,
) -> None:
    das_rms = (
        DASArray()
        .load(file_paths=file_paths, n_start=n_start, n_end=n_end)
        .fk_filter_real(f_min=f_min, f_max=f_max, v_min=v_min, v_max=v_max)
        .rms(window_size_second=rms_window_size)
    )

    with as_file(
        files("dasly.models").joinpath(DEFAULT_MODEL_FILENAME)
    ) as resolved_model_path:
        das = (
            das_rms
            .match_train_scale(
                train_dn=DEFAULT_TRAIN_PHYSICAL_WIDTH / DEFAULT_TRAIN_WIDTH,
                train_dt=DEFAULT_TRAIN_PHYSICAL_HEIGHT / DEFAULT_TRAIN_HEIGHT
            )
            .grayscale_transform(by_column=DEFAULT_GRAYSCALE_BY_COLUMN)
            .rgb_transform()
            .yolo(model=str(resolved_model_path), iou=yolo_iou)
        )

    hyperbolas = fit_multiple_hyperbolas_least_squares(
        array=das_rms,
        boxesn=das.meta.boxesn,
        num_points=hyperbolas_num_points,
        by_channel=hyperbolas_by_channel
    )

    hyper_metrics = derive_hyperbola_metrics(
        hyperbolas=hyperbolas,
        dn=das_rms.meta.dn,
        dxn=das_rms.meta.dxn
    )

    _, yyyymmdd, hhmmss, _ = parse_file_path(file_paths[0])
    chunk = f'{yyyymmdd}T{hhmmss}'
    additional = {
        'source_distance': hyper_metrics['source_distance'],
        'hyper_rmse_norm': hyper_metrics['hyper_rmse_norm'],
        'hyper_mae_norm': hyper_metrics['hyper_mae_norm'],
        'confidence': das.meta.boxes_conf
    }

    boxes_df = build_box_df(
        boxesp=das.meta.boxesp,
        boxesn=das.meta.boxesn,
        chunk=chunk,
        chunk_size=len(file_paths),
        additional=additional
    )

    save_to_db(
        df=boxes_df,
        table_name=db_table,
        connection_string=connection_string
    )
