from typing import TYPE_CHECKING

import numpy as np
import torch
from ultralytics import YOLO

from ..execution import box_saver


if TYPE_CHECKING:
    from ..core.dasarray import DASArray


def yolo(
    data: np.ndarray,
    model: str,
    conf: float = 0.25,
    iou: float = 0.7,
    reverse_data: bool = True
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Perform object detection using YOLO.

    Args:
        data (np.ndarray): Image data to perform object detection.
        model (str): Model path to use for object detection.
        conf (float): Confidence threshold. Objects detected with confidence
            below this threshold will be disregarded. Default is 0.25.
        iou (float): Intersection over union threshold for Non-Maximum
            Suppression (NMS). Lower values result in fewer detections by
            eliminating overlapping boxes. Default is 0.7.
        reverse_data (bool): Reverse the order of the time axis. If the model
            was trained on images with time axis reversed to the input data,
            set this to True to ensure that the model performs correctly.
            Default is True.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: Detected boxes, normalized
            boxes, and confidence values.
    """
    # Load a model
    model = YOLO(model)  # pretrained model
    # Use the GPU if available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)  # Move the model to the device

    # Rounded the data to the lower nearest multiple of 32
    data_shape = data.shape
    adjusted_shape = (
        (data_shape[0] // 32) * 32,  # Ensure width is a multiple of 32
        (data_shape[1] // 32) * 32   # Ensure height is a multiple of 32
    )

    if reverse_data:
        results = model(data[::-1, :, :],
                        conf=conf,
                        iou=iou,
                        imgsz=adjusted_shape)
    else:
        results = model(data, conf=conf, iou=iou, imgsz=adjusted_shape)

    boxes = results[0].boxes.xyxy.cpu().numpy()
    boxesn = results[0].boxes.xyxyn.cpu().numpy()
    conf = results[0].boxes.conf.cpu().numpy()

    # Reverse box coordinates if data was reversed
    if reverse_data and boxes.size > 0:
        boxes[:, [1, 3]] = data.shape[0] - boxes[:, [3, 1]]
        boxesn[:, [1, 3]] = 1 - boxesn[:, [3, 1]]

    return boxes, boxesn, conf


class DASYolo:

    def yolo(
        self,
        model: str,
        conf: float = 0.25,
        iou: float = 0.7,
        reverse_data: bool = True
    ) -> 'DASArray':
        """Perform object detection using YOLO.

        Args:
            model (str): Model path to use for object detection.
            conf (float): Confidence threshold. Objects detected with
                confidence below this threshold will be disregarded. Default
                is 0.25.
            iou (float): Intersection over union threshold for Non-Maximum
                Suppression (NMS). Lower values result in fewer detections by
                eliminating overlapping boxes. Default is 0.7.
            reverse_data (bool): Reverse the order of the time axis. If the
                model was trained on images with time axis reversed to the
                input data, set this to True to ensure that the model performs
                correctly. Default is True.

        Returns:
            DASArray: Array with detected boxes and their confidence in
                metadata.
        """
        boxes, boxesn, conf = yolo(
            self,
            model=model,
            conf=conf,
            iou=iou,
            reverse_data=reverse_data
        )

        boxesd = box_saver.denormalize_boxesn(
            boxesn=boxesn,
            t_start=self.meta.timestamps[0],
            t_end=self.meta.timestamps[-1],
            n_start=self.meta.channels[0],
            n_end=self.meta.channels[-1],
        )

        boxesp = box_saver.cast_box_times_to_datetime64(boxes=boxesd)

        self.meta.update(
            boxesn=boxesn,
            boxesp=boxesp,
            boxes_conf=conf
        )
        return self
