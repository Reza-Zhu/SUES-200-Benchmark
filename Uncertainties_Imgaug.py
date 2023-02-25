import os
import re
import glob
import numpy as np
import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from PIL import Image
import imgaug.augmenters as iaa

class AddBlock(object):
    def __init__(self, gap, type):
        self.gap = gap
        self.type = type
    def __call__(self, img):
        height = img.height
        img = img.crop((0, 0, height - self.gap, height))

        if self.type == "flip":
            crop = img.crop((0, 0, self.gap, height))
            crop = crop.transpose(Image.FLIP_LEFT_RIGHT)
        else:
            crop = Image.new("RGB", (self.gap, height), self.type)

        joint = Image.new("RGB", (height, height))
        joint.paste(crop, (0, 0, self.gap, height))
        joint.paste(img, (self.gap, 0, height, height))

        return joint

class Weather(object):
    def __init__(self, type):

        if type == "snow":
            self.seq = iaa.imgcorruptlike.Snow(severity=2)
        elif type == "rain":
            self.seq = iaa.Rain()
        elif type == "fog":
            self.seq = iaa.Fog()

    def __call__(self, img):
        width = img.width
        height = img.height

        img = np.array(img).reshape(1, width, height, 3)
        img = self.seq.augment_images(img)
        img = np.array(img).reshape(width, height, 3)
        img = Image.fromarray(img)

        return img


