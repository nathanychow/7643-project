import os
import numpy as np
import pandas as pd
import torch
from torchvision.io import read_image
from torch.utils.data import Dataset

class DriversDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)


    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 2]) # 2 because img name is 3rd column.
        image = read_image(img_path)
        #print('image', image.shape)
        image = torch.from_numpy(np.array(image).astype(np.float32))
        #print('image', image.shape)
        label = int(self.img_labels.iloc[idx, 1][1]) # the second [1] is to grab the number from c0 - c9
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label