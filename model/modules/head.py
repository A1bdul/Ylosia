import torch
import torch.nn as nn
from math import log
from typing import List
from .block import DFL, Conv
from .utils import dist2bbox, make_anchors


class DetectionHead(nn.Module):
    anchors = torch.empty(0)
    strides = torch.empty(0)
    shape = None

    def __init__(self, num_classes: int = 1, in_channels: List[int] = []):
        super().__init__()
        self.nc = num_classes
        self.n_layers = len(in_channels)
        self.reg_max = 16
        self.n_outputs = 4 * self.reg_max + self.nc
        self.stride = torch.zeros(self.n_layers)

        # print(in_channels)

        c2 = max(16, in_channels[0] // 4, self.reg_max * 4)
        c3 = max(in_channels[0], min(self.nc, 100))

        self.box_conv = nn.ModuleList(
            nn.Sequential(Conv(c, c2, kernel_size=3),
                          Conv(c2, c2, kernel_size=3),
                          nn.Conv2d(c2, 4 * self.reg_max, kernel_size=1))
            for c in in_channels)

        self.cls_convs = nn.ModuleList(
            nn.Sequential(Conv(c, c3, kernel_size=3),
                          Conv(c3, c3, kernel_size=3),
                          nn.Conv2d(c3, self.nc, kernel_size=1))
            for c in in_channels)

        self.dfl = DFL(in_channels=self.reg_max)

    def forward(self, x):
        device = x[0].device
        
        for i in range(self.n_layers):
            x[i] = torch.cat((self.box_conv[i](x[i]), self.cls_convs[i](x[i])),
                             dim=1)

        if self.training:
            return x

        shape = x[0].shape  # (batch, channels, height, width)

        # (batch, 4 * reg_max + nc, n_layers * height * width)
        x_cat = torch.cat([xi.view(shape[0], self.n_outputs, -1) for xi in x],
                          dim=2)

        if self.shape != shape:
            self.anchors, self.strides = make_anchors(x, self.stride, device=device)
            self.anchors.transpose_(0, 1)
            self.strides.transpose_(0, 1)
            self.shape = shape

        # (batch, 4 * reg_max, n_layers * height * width), (batch, nc, n_layers * height * width)
        box, cls = x_cat.split((4 * self.reg_max, self.nc), dim=1)
        # (batch, 4, n_layers * height * width) (ltrb) -> (xywh)

        bbox = dist2bbox(self.dfl(box), self.anchors.unsqueeze(0),
                         dim=1) * self.strides

        # (batch, 4 + nc, n_layers * height * width)
        out = torch.cat((bbox, torch.sigmoid(cls)), dim=1)
        return out

    def _bias_init(self) -> None:
        """
        Initialize biases for Conv2d layers

        Must set stride before calling this method
        """
        if self.stride is None:
            raise ValueError('stride is not set')

        for b_list, c_list, s in zip(self.box_conv, self.cls_convs,
                                     self.stride):
            b_list[-1].bias.data[:] = 1.0
            c_list[-1].bias.data[:self.nc] = log(5 / (self.nc * (640 / s)**2))
            c_list[-1].bias.data[:self.nc] = log(5 / (self.nc * (640 / s)**2))
