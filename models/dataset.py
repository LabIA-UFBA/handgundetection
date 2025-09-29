import os
from PIL import Image
import pandas as pd
import torch
import torchvision.ops as ops

class YoloDataset:
    def __init__(self, folder):
        self.images_folder = os.path.join(folder, 'images')
        self.labels_folder = os.path.join(folder, 'labels')
        self.image_files = [f for f in os.listdir(self.images_folder) if f.endswith(('.jpg', '.JPG', '.png', '.PNG'))]
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_file = self.image_files[idx]
        image_path = os.path.join(self.images_folder, image_file)
        labels_path = os.path.join(self.labels_folder, os.path.splitext(image_file)[0] + '.txt')

        img = Image.open(image_path)

        try:
            labels = pd.read_csv(labels_path, names=['class', 'x', 'y', 'w', 'h'], sep=' ')
            labels = torch.tensor(labels.values, dtype=torch.float32)[:, 1:].to(self.device)
            width, height = img.size
            labels = self._convert(labels, width, height)
        except:
            labels = torch.tensor([], dtype=torch.float32).to(self.device)

        return img, labels
    
    def __iter__(self):
        for i in range(len(self)):
            yield self[i]
    
    def _convert(self, labels, width, height):
        xyxy = ops.box_convert(labels, in_fmt='cxcywh', out_fmt='xyxy')
        xyxy[:, 0] *= width
        xyxy[:, 1] *= height
        xyxy[:, 2] *= width
        xyxy[:, 3] *= height

        return xyxy
