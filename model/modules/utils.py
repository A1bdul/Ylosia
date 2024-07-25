from typing import List
import torch
import numpy as np

from typing import Tuple, Any, Union

def autopad(kernel_size: int, padding: int = None):
    if padding is None:
        return kernel_size // 2 if isinstance(kernel_size, int) else [k // 2 for k in kernel_size]
    return padding

def make_anchors(features: List[torch.Tensor], strides: List[torch.Tensor], device: str = 'cpu'):
    """
    Generate anchor points and stride tensors for a given set of feature maps.

    Args:
        features (list of torch.Tensor): List of feature map tensors with shapes (batch, channels, height, width).
        strides (torch.Tensor): Tensor of stride values for each feature map.
        device (str): Device to perform tensor operations ('cpu' or 'cuda').

    Returns:
        tuple: A tuple containing:
            - anchor_points (torch.Tensor): The anchor points tensor.
            - stride_tensor (torch.Tensor): The stride tensor.
    """
    anchor_points = []
    stride_tensor = []

    for i, feature in enumerate(features):
        batch_size, _, height, width = feature.shape
        stride = strides[i]

        # Generate grid of anchor points for the current feature map
        grid_y, grid_x = torch.meshgrid(torch.arange(height, device=device), torch.arange(width, device=device), indexing='ij')
        grid = torch.stack((grid_x, grid_y), dim=2).float().to(device)

        # Normalize grid to get anchor points
        anchor_points.append((grid.view(-1, 2) * stride).to(device))
        stride_tensor.append(torch.full((height * width, 1), stride, device=device))

    # Concatenate anchor points and stride tensor from all feature maps
    anchor_points = torch.cat(anchor_points, dim=0).to(device)
    stride_tensor = torch.cat(stride_tensor, dim=0).to(device)

    return anchor_points, stride_tensor

def dist2bbox(distance: torch.Tensor, anchor_points: torch.Tensor, xywh: bool = True, dim: int = -1, device: str = 'cpu'):
    """
    Transform distance in (ltrb) to bounding box (xywh) or (xyxy)
    """
    distance = distance.to(device)
    anchor_points = anchor_points.to(device)

    lt, rb = torch.chunk(distance, 2, dim=dim)
    xy_lt = anchor_points - lt
    xy_rb = anchor_points + rb

    if xywh:
        center = (xy_lt + xy_rb) / 2
        wh = xy_rb - xy_lt
        return torch.cat((center, wh), dim=dim).to(device)

    return torch.cat((xy_lt, xy_rb), dim=dim).to(device)

def bbox2dist(bbox: torch.Tensor, anchor_points: torch.Tensor, reg_max: float, device: str = 'cpu'):
    """
    Transform bounding box (xyxy) to distance (ltrb)
    """
    bbox = bbox.to(device)
    anchor_points = anchor_points.to(device)

    xy_lt, xy_rb = torch.chunk(bbox, 2, dim=-1)
    lt = anchor_points - xy_lt
    rb = xy_rb - anchor_points
    return torch.cat((lt, rb), dim=-1).clamp(max=reg_max - 0.01).to(device)

def xywh2xyxy(xywh: torch.Tensor, device: str = 'cpu'):
    """
    Convert bounding box coordinates from (xywh) to (xyxy)
    """
    xywh = xywh.to(device)

    xy, wh = torch.chunk(xywh, 2, dim=-1)
    return torch.cat((xy - wh / 2, xy + wh / 2), dim=-1).to(device)

def xyxy2xywh(xyxy: torch.Tensor, device: str = 'cpu'):
    """
    Convert bounding box coordinates from (xyxy) to (xywh)
    """
    xyxy = torch.from_numpy(xyxy).to(device)

    xy_lt, xy_rb = torch.chunk(xyxy, 2, dim=-1)
    return torch.cat(((xy_lt + xy_rb) / 2, xy_rb - xy_lt), dim=-1).to(device)


def box_iou_batch(gt_boxes: np.ndarray, pred_boxes: np.ndarray) -> np.ndarray:
    """
    Compute Intersection over Union (IoU) of two sets of bounding boxes -
        `gt_boxes` and `pred_boxes`. Both sets
        of boxes are expected to be in `(xyxy)` format.

    Args:
        gt_boxes (np.ndarray): 2D `np.ndarray` representing ground-truth boxes.
            `shape = (N, 4)` where `N` is number of true objects.
        pred_boxes (np.ndarray): 2D `np.ndarray` representing detection boxes.
            `shape = (M, 4)` where `M` is number of detected objects.

    Returns:
        np.ndarray: Pairwise IoU of boxes from `gt_boxes` and `pred_boxes`.
            `shape = (N, M)` where `N` is number of true objects and
            `M` is number of detected objects.
    """

    def box_area(box):
        return (box[2] - box[0]) * (box[3] - box[1])

    area_true = box_area(gt_boxes.T)
    area_detection = box_area(pred_boxes.T)

    top_left = np.maximum(gt_boxes[:, None, :2], pred_boxes[:, :2])
    bottom_right = np.minimum(gt_boxes[:, None, 2:], pred_boxes[:, 2:])

    area_inter = np.prod(np.clip(bottom_right - top_left, a_min=0, a_max=None), 2)
    return area_inter / (area_true[:, None] + area_detection - area_inter)


def non_max_suppression(predictions: np.ndarray, iou_threshold: float = 0.5) -> np.ndarray:
    """
    Perform Non-Maximum Suppression (NMS) on object detection predictions.

    Args:
        predictions (np.ndarray): An array of object detection predictions in
            the format of `(x_min, y_min, x_max, y_max, score)`
            or `(x_min, y_min, x_max, y_max, score, class)`.
        iou_threshold (float, optional): The intersection-over-union threshold
            to use for non-maximum suppression.

    Returns:
        np.ndarray: A boolean array indicating which predictions to keep after n
            on-maximum suppression.

    Raises:
        AssertionError: If `iou_threshold` is not within the
            closed range from `0` to `1`.
    """
    assert 0 <= iou_threshold <= 1, (
        "Value of `iou_threshold` must be in the closed range from 0 to 1, "
        f"{iou_threshold} given."
    )
    rows, columns = predictions.shape

    # add column #5 - category filled with zeros for agnostic nms
    if columns == 5:
        predictions = np.c_[predictions, np.zeros(rows)]

    # sort predictions column #4 - score
    sort_index = np.flip(predictions[:, 4].argsort())
    predictions = predictions[sort_index]

    boxes = predictions[:, :4]
    categories = predictions[:, 5]
    ious = box_iou_batch(boxes, boxes)
    ious = ious - np.eye(rows)

    keep = np.ones(rows, dtype=bool)

    for index, (iou, category) in enumerate(zip(ious, categories)):
        if not keep[index]:
            continue

        # drop detections with iou > iou_threshold and
        # same category as current detections
        condition = (iou > iou_threshold) & (categories == category)
        keep = keep & ~condition

    return keep[sort_index.argsort()]

### Data validation for Detections class
def validate_xyxy(xyxy:Any) -> None:
    expected_shape = "(_, 4)"
    actual_shape = str(getattr(xyxy, "shape", None))
    is_valid = isinstance(xyxy, np.ndarray) and xyxy.ndim == 2 and xyxy.shape[1] == 4
    if not is_valid:
        raise ValueError(
            f"xyxy must be a 2D np.ndarray with shape {expected_shape}, but got shape "
            f"{actual_shape}"
        )

def validate_mask(mask:Any, n:int) -> None:
    expected_shape = f"({n}, H, W)"
    actual_shape = str(getattr(mask, "shape", None))
    is_valid = mask is None or (
        isinstance(mask, np.ndarray) and len(mask.shape) == 3 and mask.shape[0] == n
    )
    if not is_valid:
        raise ValueError(
            f"mask must be a 3D np.ndarray with shape {expected_shape}, but got shape "
            f"{actual_shape}"
        )


def validate_class_id(class_id:Any, n:int) -> None:
    expected_shape = f"({n},)"
    actual_shape = str(getattr(class_id, "shape", None))
    is_valid = class_id is None or (
        isinstance(class_id, np.ndarray) and class_id.shape == (n,)
    )
    if not is_valid:
        raise ValueError(
            f"class_id must be a 1D np.ndarray with shape {expected_shape}, but got "
            f"shape {actual_shape}"
        )


def validate_confidence(confidence:Any, n:int) -> None:
    expected_shape = f"({n},)"
    actual_shape = str(getattr(confidence, "shape", None))
    is_valid = confidence is None or (
        isinstance(confidence, np.ndarray) and confidence.shape == (n,)
    )
    if not is_valid:
        raise ValueError(
            f"confidence must be a 1D np.ndarray with shape {expected_shape}, but got "
            f"shape {actual_shape}"
        )


def validate_tracker_id(tracker_id:Any, n:int) -> None:
    expected_shape = f"({n},)"
    actual_shape = str(getattr(tracker_id, "shape", None))
    is_valid = tracker_id is None or (
        isinstance(tracker_id, np.ndarray) and tracker_id.shape == (n,)
    )
    if not is_valid:
        raise ValueError(
            f"tracker_id must be a 1D np.ndarray with shape {expected_shape}, but got "
            f"shape {actual_shape}"
        )

def validate_detections_fields(
    xyxy: Any,
    mask: Any,
    class_id: Any,
    confidence: Any,
    tracker_id: Any
) -> None:
    validate_xyxy(xyxy)
    n = len(xyxy)
    validate_mask(mask, n)
    validate_class_id(class_id, n)
    validate_confidence(confidence, n)
    validate_tracker_id(tracker_id, n)
