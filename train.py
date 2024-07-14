import re
import os
import torch

import argparse
from tqdm import trange
from torchinfo import summary
from model.dataset.data_loader import Dataset
from model.models.detection_model import DetectionModel
from torch.utils.data import DataLoader

import yaml
import logging
import matplotlib.pyplot as plt
import numpy as np

log = logging.getLogger("Ylosia")
logging.basicConfig(level=logging.INFO)

def get_args():
    parser = argparse.ArgumentParser(description='YOLOv8 model training')
    parser.add_argument('--model-config',
                        type=str,
                        default='model/config/models/yolov8n.yaml',
                        help='path to model config file')
    parser.add_argument('--weights', type=str, help='path to weights file')

    parser.add_argument('--train-config',
                        type=str,
                        default='model/config/training/train.yaml',
                        help='path to training config file')

    dataset_args = parser.add_argument_group('Dataset')
    dataset_args.add_argument('--dataset',
                              type=str,
                              required=True,
                              help='path to dataset config file')
    dataset_args.add_argument('--dataset-mode',
                              type=str,
                              default='train',
                              help='dataset mode')

    parser.add_argument('--device',
                        '-d',
                        type=str,
                        default='cuda',
                        help='device to model on')

    parser.add_argument('--save',
                        '-s',
                        action='store_true',
                        help='save trained model weights')

    return parser.parse_args()


def main(args):
    # Read config1.yaml
    with open(args.dataset, 'r') as file:
        config1 = yaml.safe_load(file)

    # Read config2.yaml
    with open(args.model_config, 'r') as file1:
        config2_str = file1.read()

    if config1['nc']:
        num_classes = f"num_classes: {config1['nc']}\n"
        config2_str = re.sub(r'^num_classes:\s*\d+\s*$', num_classes, config2_str, flags=re.MULTILINE)
        with open(args.model_config, 'w') as file:
            file.write(config2_str)

    train_config = yaml.safe_load(open(args.train_config, 'r'))

    device = torch.device(args.device)
    model = DetectionModel(args.model_config, device=device)
    summary(model, input_size=[1, 3, 640, 640], device=device)

    optimizer = torch.optim.Adam(model.parameters(), lr=train_config['lr'])
    dataset = Dataset(config=args.dataset,
                      mode=args.dataset_mode,
                      batch_size=train_config['batch_size'],
                      device=device)

    dataloader = DataLoader(dataset,
                            batch_size=train_config['batch_size'],
                            shuffle=True,
                            collate_fn=Dataset.collate_fn)

    if args.save:
        save_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                 train_config['save_dir'],
                                 os.path.splitext(os.path.basename(args.model_config))[0])
        os.makedirs(save_path, exist_ok=True)

    # Initialize lists to store epoch and loss values for plotting
    epochs = []
    losses = []

    for epoch in trange(train_config['epochs']):
        epoch_losses = []
        for batch in dataloader:
            loss = model.loss(batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        # Calculate average loss for the epoch
        epoch_loss_avg = np.mean(epoch_losses)
        epochs.append(epoch + 1)
        losses.append(epoch_loss_avg)

        if (epoch + 1) % train_config['save_freq'] == 0 and args.save:
            log.info('Saving model to checkpoint...')
            print('Saving model to checkpoint...')
            model.save(os.path.join(save_path, f'{epoch + 1}.pt'))

            # Plot and save loss graph in the same directory as the weights
            plt.figure(figsize=(10, 5))
            plt.plot(epochs, losses, label='Training Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training Loss Over Epochs')
            plt.grid(True)
            plt.legend()
            plt.savefig(os.path.join(save_path, f'loss_plot_epoch_{epoch + 1}.png'))
            plt.close()

    # Final plot of the overall loss curve
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Epochs')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(save_path, 'loss_plot_final.png'))
    plt.show()


if __name__ == "__main__":
    args = get_args()
    main(args)
