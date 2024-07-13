import os
import torch

import argparse
from tqdm import trange
from torchinfo import summary
from model.dataset.data_loader import Dataset
from model.models.detection_model import DetectionModel
from torch.utils.data import DataLoader

import yaml

def get_args():
    parser = argparse.ArgumentParser(description='YOLOv8 model training')
    parser.add_argument(
        '--model-config',
        type=str,
        default='model/config/models/yolov8n.yaml',
        help='path to model config file'
    )
    parser.add_argument(
        '--weights',
        type=str,
        help='path to weights file'
    )

    parser.add_argument(
        '--train-config',
        type=str,
        default='model/config/training/train.yaml',
        help='path to training config file'
    )

    dataset_args = parser.add_argument_group('Dataset')
    dataset_args.add_argument(
        '--dataset',
        type=str,
        default='dataset/data.yaml',
        help='path to dataset config file'
    )
    dataset_args.add_argument(
        '--dataset-mode',
        type=str,
        default='train',
        help='dataset mode'
    )

    parser.add_argument(
        '--device',
        '-d',
        type=str,
        default='cuda',
        help='device to model on'
    )

    parser.add_argument(
        '--save',
        '-s',
        action='store_true',
        help='save trained model weights'
    )

    return parser.parse_args()

def main(args):
    # Load the YAML files
    with open(args.dataset, 'r') as config1_file, open(args.train_config, 'r') as config2_file:
        config1 = yaml.safe_load(config1_file)
        config2 = yaml.safe_load(config2_file)

    # Move 'path' from config1 to config2
    if 'path' in config1:
        config2['num_classes'] = config1.pop('nc')

    # Save the updated data back to the YAML files
    with open(args.dataset, 'w') as config1_file, open(args.train_config, 'w') as config2_file:
        yaml.safe_dump(config1, config1_file)
        yaml.safe_dump(config2, config2_file)

    train_config = yaml.safe_load(open(args.train_config, 'r'))

    device = torch.device(args.device)
    model = DetectionModel(args.model_config, device=device)
    summary(model, input_size=[1, 3, 640, 640])
    dataset = Dataset(args.dataset, args.dataset_mode, batch_size=train_config['batch_size'])
    dataloader = DataLoader(dataset, batch_size=train_config['batch_size'], shuffle=True, 
                            collate_fn=Dataset.collate_fn)
    for epoch in trange(train_config['epochs']):
        for batch in dataloader:
            print(batch.keys())

if __name__ == "__main__":
    args = get_args()
    main(args)
