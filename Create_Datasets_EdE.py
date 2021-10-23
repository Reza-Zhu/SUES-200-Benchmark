import PIL.Image
import os
import glob
import torch
from torchvision import datasets, transforms
from torchvision.io import read_image
from PIL import Image
import numpy as np

# from torch.utils.data.dataloader
class EdE_Dataset(torch.utils.data.Dataset):
    def __init__(self, sample_path, label_path, transform=None, target_transform=None):
        self.label_path = label_path
        self.sample_path = sample_path

        self.sample_img_list = glob.glob(os.path.join(sample_path, "*"))
        self.label_img_list = glob.glob(os.path.join(label_path, "*"))

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.label_img_list)

    def __getitem__(self, item):
        imgs_dir = glob.glob(os.path.join(self.sample_img_list[item],"*"))
        sample = []
        for i in imgs_dir:
            img = Image.open(i)
            if self.transform:
                img = self.transform(img)
            sample.append(torch.from_numpy(np.array(img)))
        labels_dir = glob.glob(os.path.join(self.label_img_list[item],"*"))
        label_img = labels_dir[0]
        label_img = Image.open(label_img)
        if self.target_transform:
            label_img = self.target_transform(label_img)
        label = [torch.from_numpy(np.array(label_img))]
        return sample, label


