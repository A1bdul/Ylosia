import os
import yaml
from glob import glob
import logging

import cv2
import numpy as np
import torch
from math import ceil

from .utils import pad_to, pad_xywh

from typing import Tuple

log = logging.getLogger("dataset")
logging.basicConfig(level=logging.DEBUG)


class Dataset(torch.utils.data.Dataset):
    """
    Dataset class for loading images and annotations.

    Args:
        config (str): path to dataset config file
        batch_size (optional, int): batch size for dataloader
        mode (optional, str): dataset mode (train, val, test)
        img_size (optional, Tuple[int,int]): image size to pad images to
        device (optional, str): device to use for tensor operations ('cpu' or 'cuda')
    """

    def __init__(self,
                 config: str,
                 batch_size: int = 8,
                 mode: str = 'train',
                 img_size: Tuple[int, int] = (640, 640),
                 device: str = 'cpu'):
        super().__init__()
        self.config = yaml.safe_load(open(config, 'r'))
        self.dataset_path = os.path.dirname(config)
        self.batch_size = batch_size
        self.img_size = img_size
        self.device = torch.device(device)

        assert mode in ('train', 'valid', 'val', 'test'), f'Invalid mode: {mode}'
        self.mode = mode

        # Remove empty labels and corresponding images if in training mode
        if self.mode == 'train':
            self.im_files, self.label_files = self.remove_empty_labels()
        else:
            self.im_files = self.get_image_paths()
            self.label_files = self.get_label_paths()

        print(
            f'Found {len(self.im_files)} images in {os.path.join(self.dataset_path, self.mode+"/images")}'
        )

        if self.mode == 'train':
            print(
                f'Found {len(self.label_files)} labels in {os.path.join(self.dataset_path, self.mode + "/labels")}'
            )

        self.labels = self.get_labels() if self.label_files else [{} for _ in range(len(self.im_files))]
        self.seen_idxs = set()

    def remove_empty_labels(self):
        """
        Removes empty label files and corresponding image files from the dataset.
        """
        im_dir = os.path.join(self.dataset_path, f'{self.mode}/images')
        label_dir = os.path.join(self.dataset_path,
                                 f'{self.mode}/labels')

        image_paths = glob(os.path.join(im_dir, '*.jpg')) + \
                      glob(os.path.join(im_dir, '*.png')) + \
                      glob(os.path.join(im_dir, '*.jpeg'))

        label_paths = glob(os.path.join(label_dir, '*.txt'))

        # Create a list to store the filtered image and label paths
        filtered_image_paths = []
        filtered_label_paths = []

        for image_path in image_paths:
            # Get the corresponding label file path
            label_path = os.path.join(
                label_dir,
                os.path.splitext(os.path.basename(image_path))[0] + ".txt")

            removed = False
            # Check if the label file exists and is not empty
            if os.path.isfile(label_path) and os.stat(label_path).st_size > 0:
                filtered_image_paths.append(image_path)
                filtered_label_paths.append(label_path)
            else:
                # Delete the corresponding image and label files
                removed = True
                os.remove(image_path)
                os.remove(label_path)
        if removed:
            print(f'Empty labels with corresponding images is removed')

        return filtered_image_paths, filtered_label_paths

    def get_image_paths(self):
        """
        Get image paths from dataset directory

        Searches recursively for .jpg, .png, and .jpeg files.
        """
        im_dir = os.path.join(self.dataset_path, self.mode+"/images")

        image_paths = glob(os.path.join(im_dir, '*.jpg')) + \
                      glob(os.path.join(im_dir, '*.png')) + \
                      glob(os.path.join(im_dir, '*.jpeg'))

        return image_paths

    def get_label_paths(self):
        """
        Get label paths from dataset directory

        Uses ids from image paths to find corresponding label files.

        If no label directory is found, returns None.
        """
        label_dir = os.path.join(self.dataset_path, self.mode + '/labels')
        if os.path.isdir(label_dir):
            return [
                os.path.join(label_dir,
                             os.path.splitext(os.path.basename(p))[0] + ".txt")
                for p in self.im_files
            ]
        return None

    def get_labels(self):
        """
        Gets labels from label files (assumes YOLOv8 format - txt files)
        Returns a list of dictionaries, one for each image:
            {
                'bboxes': torch.Tensor of shape (num_boxes, 4) in (xywh) format
                'cls': torch.Tensor of shape (num_boxes,)
            }
        If no label files were found, returns a list of empty dictionaries.
        """
        if self.label_files is None:
            return [{} for _ in range(len(self.im_files))]

        labels = []
        # for label_file in self.label_files:
        #     with open(label_file, 'r') as f:
        #         annotations = f.readlines()

        #     boxes = []
        #     class_ids = []

        #     for ann in annotations:
        #         ann = ann.strip().split()
        #         class_id = int(ann[0])

        #         if len(ann) == 5:  # Bounding box annotation
        #             x_center, y_center, w, h = map(float, ann[1:])
        #             boxes.append([x_center, y_center, w, h])

        #         elif len(ann) > 5:  # Polygon annotation
        #             polygon_coords = np.array(ann[1:],
        #                                       dtype=float).reshape(-1, 2)
        #             x_min, y_min = np.min(polygon_coords, axis=0)
        #             x_max, y_max = np.max(polygon_coords, axis=0)
        #             w = x_max - x_min
        #             h = y_max - y_min
        #             x_center = x_min + w / 2
        #             y_center = y_min + h / 2
        #             boxes.append([x_center, y_center, w, h])

        #         class_ids.append(class_id)

        #     # Convert to tensors and move to device
        #     label_data = {
        #         'bboxes': torch.tensor(boxes, dtype=torch.float32, device=self.device),
        #         'cls': torch.tensor(class_ids, dtype=torch.long, device=self.device)
        #     }

        #     labels.append(label_data)
        for label_file in self.label_files:
            annotations = open(label_file, 'r').readlines()
            cls, boxes = [], []
            for ann in annotations:
                ann = ann.strip('\n').split(' ')
                cls.append(int(ann[0]))

                # box provided in xywh format
                boxes.append(torch.from_numpy(np.array(ann[1:5], dtype=float)))

            labels.append({
                'cls': torch.tensor(cls),
                'bboxes': torch.vstack(boxes)
            })
            
        return labels

    def load_image(self, idx):
        """
        Loads image at specified index and prepares for model input.

        Changes image shape to be specified img_size, but preserves aspect ratio.
        """
        im_file = self.im_files[idx]
        im_id = os.path.splitext(os.path.basename(im_file))[0]
        image = cv2.cvtColor(cv2.imread(im_file), cv2.COLOR_BGR2RGB)

        h0, w0 = image.shape[:2]

        if h0 > self.img_size[0] or w0 > self.img_size[1]:
            # Resize to have max dimension of img_size, but preserve aspect ratio
            ratio = min(self.img_size[0] / h0, self.img_size[1] / w0)
            h, w = min(ceil(h0 * ratio),
                       self.img_size[0]), min(ceil(w0 * ratio),
                                              self.img_size[1])
            image = cv2.resize(image, (h, w), interpolation=cv2.INTER_LINEAR)

        image = image.transpose((2, 0, 1))  # (h, w, 3) -> (3, h, w)

        image = torch.from_numpy(image).float().to(self.device) / 255.0
        # Pad image with black bars to desired img_size
        image, pads = pad_to(image, shape=self.img_size)

        return image, pads, (h0, w0), im_id

    def get_image_and_label(self, idx):
        """
        Gets image and annotations at specified index
        """
        label = self.labels[idx]
        if idx in self.seen_idxs:
            return label
        label['images'], label['padding'], label['orig_shapes'], label[
            'ids'] = self.load_image(idx)

        if 'bboxes' in label:
            label['bboxes'] = pad_xywh(label['bboxes'],
                                       label['padding'],
                                       label['orig_shapes'],
                                       return_norm=True)

        self.seen_idxs.add(idx)

        return label

    def __len__(self) -> int:
        return len(self.im_files)

    def __getitem__(self, index):
        return self.get_image_and_label(index)

    @staticmethod
    def collate_fn(batch):
        """
        Collate function to specify how to combine a list of samples into a batch
        """
        collated_batch = {}
        for k in batch[0].keys():
            if k == "images":
                collated_batch[k] = torch.stack([b[k] for b in batch], dim=0)
            elif k in ('cls', 'bboxes'):
                collated_batch[k] = torch.cat([b[k] for b in batch], dim=0)
            elif k in ('padding', 'orig_shapes', 'ids'):
                collated_batch[k] = [b[k] for b in batch]

        if 'cls' in collated_batch:
            collated_batch['batch_idx'] = [
                torch.full((batch[i]['cls'].shape[0], ), i, device=batch[i]['cls'].device)
                for i in range(len(batch))
            ]
            collated_batch['batch_idx'] = torch.cat(collated_batch['batch_idx'],
                                                    dim=0)

        return collated_batch
