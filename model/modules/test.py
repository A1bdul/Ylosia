import torch
import unittest
from block import C2f, Conv, BottleNeck
from head import DetectionHead

class TestConv(unittest.TestCase):
    conv = Conv(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
    x = torch.randn(1, 3, 32, 32)
    y = conv(x)
    assert y.shape == (1, 16, 32, 32), "Conv module output shape mismatch"
    
    # Test BottleNeck module
    bottleneck = BottleNeck(in_channels=16, out_channels=16, kernel_size=3, stride=1)
    y = bottleneck(y)
    assert y.shape == (1, 16, 32, 32), "BottleNeck module output shape mismatch"
    
    # Test C2f module
    c2f = C2f(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1, expansion=0.5, n=2)
    y = c2f(y)
    assert y.shape == (1, 32, 32, 32), "C2f module output shape mismatch"
    
    print("All tests passed.")

class TestDetectionHead(unittest.TestCase):
        
    def test_detection_head(self):
        num_classes = 80
        in_channels = [256, 512, 1024]
        model = DetectionHead(num_classes=num_classes, in_channels=in_channels)
        
        # Generate random input tensors for each layer
        x = [torch.randn(1, c, 20, 20) for c in in_channels]
        # Test forward pass in training mode
        model.train()
        out_train = model(x)
        assert all([o.shape == (1, model.n_outputs, 20, 20) for o in out_train]), "Training output shape mismatch"

        # Test forward pass in evaluation mode
        model.eval()
        x = [torch.randn(1, c, 20, 20) for c in in_channels]
        out_eval = model(x)
        assert out_eval.shape[1] == 4 + num_classes, "Evaluation output shape mismatch"
        assert out_eval.shape[2] == sum([20*20 for _ in in_channels]), "Evaluation output shape mismatch"
        
        print("All tests passed.")


if __name__ == "__main__":
    unittest.main()

