import torch
from torchvision import transforms  
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import os
import fnmatch
import locale
import sys

locale.setlocale(locale.LC_ALL, '')
sys.path.insert(1, os.path.join(sys.path[0], '..'))


def collect_paths(parrent_path, pattern="*.tif"):
    """ Recoursively get all paths by pattern
    and return them as list
    """
    results = []
    for base, _, files in os.walk(parrent_path):
        goodfiles = fnmatch.filter(files, pattern)
        results.extend(os.path.join(base, f) for f in goodfiles)
    
    return results


class PlaneSet(torch.utils.data.Dataset):
    def __init__(self, dir_path, df):
        super().__init__()
        self.dir_path = dir_path
        self.df = df
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])
        
    def expand_path(self, name):
        return self.dir_path + name + ".png"
        
    def __getitem__(self, idx):
        _class, name = tuple(self.df.iloc[idx])
        
        path = self.expand_path(name)
        pil_img = Image.open(path)
        img = self.transform(pil_img)

        # shape of image must be (3, 20, 20) but
        # (wtf) there are some 4-channel photos
        # so we must manage this. Also we need to
        # cut fourth channel off if exists
        img = img[:3, :, :]

        return img, _class
    
    def __len__(self):
        return len(self.df)


def plot_grid(planes):
    plt.style.use("dark_background")
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(6, 6))
    fig.subplots_adjust(hspace=0.3, wspace=0.3)
    font = {
        "family": "serif",
        "color":  "#73081A",
        "weight": "bold",
        "size": 14,
    }
    plt.rcParams['axes.titley'] = 1.3    # y is in axes-relative coordinates.
    
    for j in range(3):
        for i,ax in enumerate(axes[j]):
            ax.imshow(planes[i+(j*3)][0], interpolation='nearest')
            ax.set_title(planes[i+(j*3)][1], fontdict=font, loc='left', pad=-50)

    fig.tight_layout()
    plt.show()


def build_datasets(csv_path, images_path):
    with open(csv_path, "r") as file:
        data = pd.read_csv(file)

    return PlaneSet(images_path, data)
