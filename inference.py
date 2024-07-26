import os
import torch
import cv2
import argparse

from tqdm import trange
from model.dataset.data_loader import Dataset
from model.detection import Detections
from model.models.detection_model import DetectionModel
from torch.utils.data import DataLoader

from matplotlib import colormaps as cm

import logging

log = logging.getLogger("Ylosia")
logging.basicConfig(level=logging.INFO)


import argparse

def get_args():
    parser = argparse.ArgumentParser(prog='Ylosia', description='Object detection')

    parser.add_argument(
        '--config',
        type=str,
        default='model/config/models/yolov8n.yaml',
        help='path to model config file'
    )
    images = parser.add_argument_group('Images', 'Path to the Image or folder of images')
    exclusive_group = images.add_mutually_exclusive_group(required=True)
    exclusive_group.add_argument('--image',
                                 type=str,
                                 help='path to input image')
    exclusive_group.add_argument('--dataset',
                                 type=str,
                                 help='path to dataset config file')

    parser.add_argument('--dataset-mode', type=str, default='val', help='dataset mode - (train/val/test)')
    parser.add_argument('--weights', type=str, default="C:/Users/User/Projects/ML/Ylosia/weights/best.pt", help='path to model weights file')
    parser.add_argument('--device', '-d', type=str, default='cpu', help='device to run the model on')
    parser.add_argument('--threshold', type=float, default=0.5, help='confidence threshold for detections')
    parser.add_argument('--save', action='store_true', help='save trained model weights')

    return parser.parse_args()


def load_model(config, weights_path, device):
    model = DetectionModel(config, device=device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()  # Set the model to evaluation mode
    model.mode = 'eval'

    return model


def preprocess_image(image_path, device):
    dataloader = []
    batch = {}
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (640, 640))
    image_tensor = torch.from_numpy(image_resized).permute(2, 0,
                                                           1).float() / 255.0
    batch['images'] = image_tensor.unsqueeze(0).to(device)  # Add batch dimension and move to device
    dataloader.append(batch)
    return dataloader


def main(args):
    device = torch.device(args.device)
    model = load_model(args.config, args.weights, device)

    if args.dataset:
        dataset = Dataset(args.dataset, mode=args.dataset_mode)
        dataloader = DataLoader(dataset, batch_size=dataset.batch_size, shuffle=False, collate_fn=Dataset.collate_fn)
    else:        
        dataloader = preprocess_image(args.image, device=device)

    if args.save:
        label_path = os.path.join('results', args.dataset_mode, 'labels')
        save_path = os.path.join('results', args.dataset_mode, 'images')

        cmap = cm['jet']
        os.makedirs(label_path, exist_ok=True)
        os.makedirs(save_path, exist_ok=True)
       
    for batch in dataloader:
        preds = model(batch['images'].to(device))
        
        for i in trange(len(preds)):
            detections = Detections.from_yolo(preds[i])

            if args.save:
                detections.save(os.path.join(label_path, batch['ids'][i]+'.txt'), pads=batch['padding'][i], im_size=batch['orig_shapes'][i])
                image = batch['images'][i].detach().cpu().numpy().transpose((1, 2, 0))
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                # image = detections.view(image, classes_dict=dataset.config['names'], cmap=cmap, pads=batch['padding'][i])

                # cv2.waitKey(0)
                # visualized_image_path = os.path.join(save_path, batch['ids'][i] + '_visualized.jpg')
                # cv2.imwrite(visualized_image_path, image)

if __name__ == "__main__":
    args = get_args()
    main(args)
