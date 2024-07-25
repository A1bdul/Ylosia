import yaml
import torch

from .base_model import BaseModel
from model.modules.head import DetectionHead
from .loss import DetectionLoss

from model.misc import parse_from_config
from .utils import init_weights, nms

class DetectionModel(BaseModel):
    """A YOLOv8 model for object detection."""

    def __init__(self, config: str, verbose: bool = True, *args, **kwargs):
        super().__init__(*args, **kwargs)

        config = config if isinstance(config, dict) else yaml.safe_load(open(config, 'r'))
        in_channels = config.get('in_channels', 3)

        self.model, self.save_idxs = parse_from_config(config_dict=config)
        self.model.to(self.device)
        self.inplace = config.get('inplace', True)

        detect_head = self.model[-1]
        if isinstance(detect_head, DetectionHead):
            detect_head.inplace = True
            s = 256
            detect_head.stride = torch.tensor(
                [s / x.shape[-2] for x in self.forward(torch.zeros(1, in_channels, s, s, device=self.device))]
            )
            self.stride = detect_head.stride
            detect_head._bias_init()
            
        init_weights(self)
        self.loss_gains = config.get('loss_gains', None)
        self.loss_fn = DetectionLoss(self, self.device)

    def postprocess(self, preds: torch.Tensor):
        """Post-processes detection predictions."""
        return nms(preds, device=self.device)
        
        