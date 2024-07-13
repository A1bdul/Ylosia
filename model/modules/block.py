import torch
import torch.nn as nn
from utils import autopad

class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size:int =3, stride:int=1, padding:int=None, bn_act:bool=True, debug=False):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, autopad(kernel_size, padding))
        self.bn = nn.BatchNorm2d(out_channels)
        self.silu = nn.SiLU()
        self.use_bn_act = bn_act
        
    def forward(self, x):
        return self.silu(self.bn(self.conv(x))) if self.use_bn_act else self.conv(x)
    

class C2f(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size:int=1, stride:int=1, padding:int=0, expansion:float=0.5, n:int=1, shortcut:bool=False):
        super().__init__()
        hidden_size = int(out_channels * expansion)
        self.conv1 = Conv(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bottlenecks = nn.ModuleList(BottleNeck(hidden_size, hidden_size, shortcut=shortcut, expansion=expansion) for _ in range(n))
        self.conv2 = Conv((n+2)*hidden_size, out_channels,  kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        module_outputs = list(self.conv1(x).chunk(2, dim=1))
        module_outputs.extend(b(module_outputs[-1]) for b in self.bottlenecks)
        return self.conv2(torch.cat(module_outputs, dim=1))

class BottleNeck(nn.Module):
    """
    Bottleneck block for compressing information in a convolutional neural netoek
    """
    def __init__(self, in_channels:int, out_channels:int, shortcut:bool=True, kernel_size:int=3, stride:int=1, expansion:float=0.5) -> None:
        super().__init__()
        hidden_size = int(out_channels * expansion)
        self.conv1 = Conv(in_channels, hidden_size, kernel_size, stride)
        self.conv2 = Conv(hidden_size, out_channels, kernel_size, stride)

        self.residual = shortcut and in_channels == out_channels and stride != 1

    def forward(self, x):
        return x + self.conv2(self.conv1(x)) if self.residual else self.conv2(self.conv1(x))
    

class SPPF(nn.Module):
    """
    Spatial Pyramid Pooling - Fast
    
    Pool features at 4 different spatial scales, and then concatenate them together.

    Allows model to process features at different scales.
    """
    def __init__(self, in_channels:int, out_channels:int, kernel_size:int=5, hidden_size:int=None):
        super().__init__()
        hidden_size = in_channels // 2 if hidden_size is not None else hidden_size
        self.conv1 = Conv(in_channels, hidden_size, kernel_size=1, stride=1)
        self.conv2 = Conv(hidden_size*4, out_channels, kernel_size=1, stride=1)
        self.maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=kernel_size // 2)

    def forward(self, x:torch.Tensor):
        x = self.conv1(x)
        y1 = self.maxpool(x)
        y2 = self.maxpool(y1)
        return self.conv2(torch.cat((x, y1, y2, self.maxpool(y2)), dim=1))
    
class DFL(nn.Module):
    """
    Module for Distribution Focal Loss

    https://arxiv.org/abs/2006.04388
    """
    def __init__(self, in_channels:int=16):
        super().__init__()
        self.in_channels = in_channels
        self.conv = nn.Conv2d(in_channels, out_channels=1, kernel_size=1, bias=False).requires_grad_(False)
        x = torch.arange(in_channels, dtype=torch.float)
        self.conv.weight.data = nn.Parameter(x.view(1, in_channels, 1, 1))

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        b, _, a = x.shape  # (batch, channels, anchors)
        # (b, 4*reg_max, anchors) -> (b, 4, reg_max, anchors) -> (b, reg_max, 4, anchors)
        # -> (b, 1, 4, anchors) -> (b, 4, anchors)
        return self.conv(x.view(b, 4, self.in_channels, a).transpose(2,1).softmax(dim=1)).view(b, 4, a)
        

