import torch
import torch.nn as nn
from .utils import autopad


class Conv(nn.Module):
    """
    Convolutional layer with optional BatchNorm and SiLU activation.

    Args:
    - in_channels (int): Number of input channels.
    - out_channels (int): Number of output channels.
    - kernel_size (int, optional): Size of the convolutional kernel. Default is 3.
    - stride (int, optional): Stride of the convolution. Default is 1.
    - padding (int, optional): Padding size for the convolution. Default is None.
    - bn_act (bool, optional): Whether to apply BatchNorm and SiLU activation. Default is True.
    """
    def __init__(self, in_channels:int, out_channels:int, kernel_size:int =3, stride:int=1, padding:int=None, bn_act:bool=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, autopad(kernel_size, padding))
        self.bn = nn.BatchNorm2d(out_channels)
        self.silu = nn.SiLU()
        self.use_bn_act = bn_act
        
    def forward(self, x):
        """
        Forward pass through the Conv module.

        Args:
        - x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
        - torch.Tensor: Output tensor after applying Convolution, BatchNorm, and SiLU activation (if enabled).
        """
        return self.silu(self.bn(self.conv(x))) if self.use_bn_act else self.conv(x)

class C2f(nn.Module):
    """
    C2f module consisting of initial convolution, bottleneck blocks, and final convolution.

    Args:
    - in_channels (int): Number of input channels.
    - out_channels (int): Number of output channels.
    - kernel_size (int, optional): Size of the convolutional kernel for initial and final convolutions. Default is 1.
    - stride (int, optional): Stride for initial and final convolutions. Default is 1.
    - padding (int, optional): Padding size for initial and final convolutions. Default is 0.
    - expansion (float, optional): Expansion factor for bottleneck blocks. Default is 0.5.
    - n (int, optional): Number of bottleneck blocks. Default is 1.
    - shortcut (bool, optional): Whether to include shortcut connections in bottleneck blocks. Default is False.
    """
    def __init__(self, in_channels:int, out_channels:int, n:int=1, shortcut:bool=False, expansion:float=0.5, hidden_size:int=None,
                     kernel_size:int=1, stride:int=1, padding:int=0):
        super().__init__()
        if isinstance(hidden_size, int):
            self.hidden_size = hidden_size
        else:
            # Adjust hidden_size calculation to keep channels consistent
            self.hidden_size = int(out_channels * expansion)
        self.conv1 = Conv(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bottlenecks = nn.ModuleList(BottleNeck(self.hidden_size, self.hidden_size, shortcut=shortcut, expansion=expansion) for _ in range(n))
        self.conv2 = Conv((n+2)*self.hidden_size, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        """
        Forward pass through the C2f module.

        Args:
        - x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
        - torch.Tensor: Output tensor after passing through the C2f module.
        """
        module_outputs = list(self.conv1(x).chunk(2, dim=1))
        module_outputs.extend(b(module_outputs[-1]) for b in self.bottlenecks)
        return self.conv2(torch.cat(module_outputs, dim=1))

class BottleNeck(nn.Module):
    """
    Bottleneck block for compressing information in a convolutional neural network.

    Args:
    - in_channels (int): Number of input channels.
    - out_channels (int): Number of output channels.
    - shortcut (bool, optional): Whether to include shortcut connection. Default is True.
    - kernel_size (int, optional): Size of the convolutional kernel. Default is 3.
    - stride (int, optional): Stride for convolutional layers. Default is 1.
    - expansion (float, optional): Expansion factor for bottleneck block. Default is 0.5.
    """
    def __init__(self, in_channels:int, out_channels:int, shortcut:bool=True, kernel_size:int=3, stride:int=1, expansion:float=0.5) -> None:
        super().__init__()
        hidden_size = int(out_channels * expansion)
        self.conv1 = Conv(in_channels, hidden_size, kernel_size, stride)
        self.conv2 = Conv(hidden_size, out_channels, kernel_size, stride)
        self.residual = shortcut and in_channels == out_channels and stride != 1

    def forward(self, x):
        """
        Forward pass through the BottleNeck block.

        Args:
        - x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
        - torch.Tensor: Output tensor after passing through the BottleNeck block.
        """
        return x + self.conv2(self.conv1(x)) if self.residual else self.conv2(self.conv1(x))    

class SPPF(nn.Module):
    """
    Spatial Pyramid Pooling - Fast module for pooling features at multiple scales.

    Args:
    - in_channels (int): Number of input channels.
    - out_channels (int): Number of output channels.
    - kernel_size (int, optional): Size of the pooling kernel. Default is 5.
    - hidden_size (int, optional): Number of hidden channels. Default is half of input channels.
    """
    def __init__(self, in_channels:int, out_channels:int, kernel_size:int=5, hidden_size:int=None):
        super().__init__()
        hidden_size = in_channels // 2 if hidden_size is None else hidden_size
        self.conv1 = Conv(in_channels, hidden_size, kernel_size=1, stride=1)
        self.conv2 = Conv(hidden_size*4, out_channels, kernel_size=1, stride=1)
        self.maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=kernel_size // 2)

    def forward(self, x:torch.Tensor):
        """
        Forward pass through the SPPF module.

        Args:
        - x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
        - torch.Tensor: Output tensor after passing through the SPPF module.
        """
        x = self.conv1(x)
        y1 = self.maxpool(x)
        y2 = self.maxpool(y1)
        return self.conv2(torch.cat((x, y1, y2, self.maxpool(y2)), dim=1))
     
class DFL(nn.Module):
    """
    Distribution Focal Loss module for object detection.

    Args:
    - in_channels (int, optional): Number of input channels. Default is 16.
    """
    def __init__(self, in_channels:int=16):
        super().__init__()
        self.in_channels = in_channels
        self.conv = nn.Conv2d(in_channels, out_channels=1, kernel_size=1, bias=False).requires_grad_(False)
        x = torch.arange(in_channels, dtype=torch.float)
        self.conv.weight.data = nn.Parameter(x.view(1, in_channels, 1, 1))

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the DFL module.

        Args:
        - x (torch.Tensor): Input tensor of shape (batch_size, 4*reg_max, anchors).

        Returns:
        - torch.Tensor: Output tensor after passing through the DFL module.
        """
        b, _, a = x.shape  # (batch_size, 4*reg_max, anchors)
        # Reshape and apply softmax
        return self.conv(x.view(b, 4, self.in_channels, a).transpose(2,1).softmax(dim=1)).view(b, 4, a)
