import os
import torch
import torch.nn as nn
from model.modules.head import DetectionHead
from model.modules.block import Conv, C2f, SPPF

def parse_from_config(config_dict: dict):
    """Parses a YOLOv8 configuration dictionary and creates a model.

    Args:
        config_dict (dict): The YOLOv8 configuration dictionary.
        
    Returns:
        Tuple[nn.Module, set]: A tuple containing the constructed YOLOv8 model and a set of indices to save.
    """

    num_classes = config_dict.get('num_classes', 80)
    depth, width, max_channels = config_dict['scale']
    channels = [config_dict.get('in_channels', 3)]

    modules = []
    save_idx = set()

    for i, (module_name, f, r,
            args) in enumerate(config_dict['backbone'] + config_dict['head']):
        module = getattr(torch.nn, module_name[3:]) if module_name.startswith(
            'nn.') else globals()[module_name]
        if module in (Conv, C2f, SPPF):
            in_channels = channels[f] if isinstance(f, int) else sum(
                [channels[idx] for idx in f])
            out_channels = args[0]  # Output channels from config

            if out_channels != num_classes:
                out_channels = int(min(out_channels, max_channels) * width)

            if module == C2f:
                args = [
                    in_channels, out_channels,
                    max(round(r * depth), 1), *args[1:]
                ]
            else:
                args = [in_channels, out_channels, *args[1:]]

        elif module in (DetectionHead, ):
            # Special handling for DetectionHead to pass in input channels
            args = [num_classes]
            args.append([channels[idx] for idx in f])
        m_ = module(*args)
        modules.append(m_)
        m_.i, m_.f = i, f  # Store index and "from" index for later use

        # Update the set of indices to save
        save_idx.update([f] if isinstance(f, int) else f)

        if i == 0:
            channels = []  # Reset channel list
        channels.append(out_channels)
        # Remove the initial "from" index (-1) as it's not a valid index
    save_idx.remove(-1)

    return nn.Sequential(*modules), save_idx
