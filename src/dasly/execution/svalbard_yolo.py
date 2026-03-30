from ..core.dasarray import DASArray
from ..fitting.hyperbola_fitter import (
    fit_multiple_hyperbolas_least_squares,
    derive_hyperbola_metrics
)
from .box_saver import save_to_db, build_box_df
from ..loader.fsearcher import parse_file_path


n_start, n_end = 5_000, 115_000  # Start and end channel indices
f_min, f_max = 15, 25  # Hz
v_min, v_max = 1_484, 1_484.4  # m/s
rms_window_size = 0.5  # Window size for RMS calculation in seconds
train_size = (640, 640)  # Training size in pixels
train_physical = (110_000, 30)  # Training size in physical units (m, s)
grayscale_by_column = True  # Grayscale transform by column
model_path = 'models/fin_whale_detection_weights.pt'  # Path to YOLO model
yolo_iou = 0.25  # IOU threshold for YOLO
hyperbolas_num_points = 10  # Number of binary transformed points preserved
hyperbolas_by_channel = True  # Binary transformed points by channel


# Define the processing function
def process_hdf5(
    file_paths: list[str],
    db_table: str,
    connection_string: str,
) -> None:
    das_rms = (
        DASArray()
        .load(file_paths=file_paths, n_start=n_start, n_end=n_end)
        .fk_filter_real(f_min=f_min, f_max=f_max, v_min=v_min, v_max=v_max)
        .rms(window_size_second=rms_window_size)
    )

    das = (
        das_rms
        .match_train_scale(
            train_dn=train_physical[0] / train_size[0],
            train_dt=train_physical[1] / train_size[1]
        )
        .grayscale_transform(by_column=grayscale_by_column)
        .rgb_transform()
        .yolo(model=model_path, iou=yolo_iou)
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
