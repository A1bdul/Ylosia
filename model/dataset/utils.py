import torch
import torch.nn.functional as F
from typing import Tuple, Union
import numpy as np


def pad_to(x: torch.Tensor, stride: int = None, shape: Tuple[int, int] = None):
    """
    Pads an image with zeros to make it divisible by stride
    (Pads both top/bottom and left/right evenly) or pads to
    specified shape.

    Args:
        x (Tensor): image tensor of shape (..., h, w)
        stride (optional, int): stride of model
        shape (optional, Tuple[int,int]): shape to pad image to
    """
    h, w = x.shape[-2:]

    if stride is not None:
        h_new = h if h % stride == 0 else h + stride - h % stride
        w_new = w if w % stride == 0 else w + stride - w % stride
    elif shape is not None:
        h_new, w_new = shape

    t, b = int((h_new - h) / 2), int(h_new - h) - int((h_new - h) / 2)
    l, r = int((w_new - w) / 2), int(w_new - w) - int((w_new - w) / 2)
    pads = (l, r, t, b)

    x_padded = F.pad(x, pads, "constant", 0)

    return x_padded, pads


def pad_xywh(xywh: Union[np.ndarray, torch.Tensor],
             pads: Tuple[int, int, int, int],
             im_size: Tuple[int, int] = None,
             return_norm: bool = False):
    """
    Add padding to the bounding boxes based on image padding

    Args:
        xywh: The bounding boxes in the format of `(x, y, w, h)`.
            if `im_size` is provided, assume this is normalized coordinates
        pad: The padding added to the image in the format
            of `(left, right, top, bottom)`.
        im_size: The size of the original image in the format of `(height, width)`.
        return_norm: Whether to return normalized coordinates
    """
    l, r, t, b = pads
    if return_norm and im_size is None:
        raise ValueError("im_size must be provided if return_norm is True")

    if im_size is not None:
        h, w = im_size
        hpad, wpad = h + b + t, w + l + r

    if isinstance(xywh, np.ndarray):
        xywh_unnorm = xywh * np.array([w, h, w, h],
                                      dtype=xywh.dtype) if im_size else xywh
        padded = xywh_unnorm + np.array([l, t, 0, 0], dtype=xywh.dtype)
        if return_norm:
            padded /= np.array([wpad, hpad, wpad, hpad], dtype=xywh.dtype)
        return padded

    xywh_unnorm = xywh * torch.tensor([w, h, w, h],
                                      dtype=xywh.dtype,
                                      device=xywh.device) if im_size else xywh
    padded = xywh_unnorm + torch.tensor(
        [l, t, 0, 0], dtype=xywh.dtype, device=xywh.device)
    if return_norm:
        padded /= torch.tensor([wpad, hpad, wpad, hpad],
                               dtype=xywh.dtype,
                               device=xywh.device)
    return padded
