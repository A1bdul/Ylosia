from typing import List
import torch

def autopad(kernel_size:int, padding:int):
    if padding is None:
        return kernel_size // 2 if isinstance(kernel_size, int) else [k // 2 for k in kernel_size]
    return padding

def make_anchors(features:List[torch.tensor], strides:List[torch.tensor]):
    """
    Generate anchor points and stride tensors for a given set of feature maps.
    
    Args:
        features (list of torch.Tensor): List of feature map tensors with shapes (batch, channels, height, width).
        strides (torch.Tensor): Tensor of stride values for each feature map.
        
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
        grid_y, grid_x = torch.meshgrid(torch.arange(height), torch.arange(width), indexing='ij')
        grid = torch.stack((grid_x, grid_y), dim=2).float()
        
        # Normalize grid to get anchor points
        anchor_points.append(grid.view(-1, 2) * stride)
        stride_tensor.append(torch.full((height * width, 1), stride))
    
    # Concatenate anchor points and stride tensor from all feature maps
    anchor_points = torch.cat(anchor_points, dim=0)
    stride_tensor = torch.cat(stride_tensor, dim=0)
    
    return anchor_points, stride_tensor

def dist2bbox(distance:torch.Tensor, anchor_points:torch.Tensor, xywh:bool=True, dim:int=-1):
    """
    Transform distance in (ltrb) to bounding box (xywh) or (xyxy)
    """

    lt, rb = torch.chunk(distance, 2, dim=dim)
    xy_lt = anchor_points - lt
    xy_rb = anchor_points + rb

    if xywh:
        center = (xy_lt + xy_rb) / 2
        wh = xy_rb - xy_lt
        return torch.cat((center, wh), dim=dim)
    
    return torch.cat((xy_lt, xy_rb), dim=dim)

def bbox2dist(bbox: torch.Tensor, anchor_points: torch.Tensor, reg_max: float):  # reg_max is now a float
    """
    Transform bounding box (xyxy) to distance (ltrb)
    """
    xy_lt, xy_rb = torch.chunk(bbox, 2, dim=-1)
    lt = anchor_points - xy_lt
    rb = xy_rb - anchor_points
    return torch.cat((lt, rb), dim=-1).clamp(max=reg_max - 0.01)

def xywh2xyxy(xywh:torch.Tensor):
    """
    Convert bounding box coordinates from (xywh) to (xyxy)
    """
    xy, wh = torch.chunk(xywh, 2, dim=-1)
    return torch.cat((xy - wh / 2, xy + wh / 2), dim=-1)

def xyxy2xywh(xyxy:torch.Tensor):
    """
    Convert bounding box coordinates from (xyxy) to (xywh)
    """
    xy_lt, xy_rb = torch.chunk(xyxy, 2, dim=-1)
    return torch.cat(((xy_lt + xy_rb) / 2, xy_rb - xy_lt), dim=-1)
