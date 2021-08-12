import torch
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
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
    def __init__(self, dir_path, df, transformation=None, augmentation=None):
        super().__init__()
        
        self.dir_path = dir_path
        self.df = df
        
        # define transform
        if transformation is None:
            self.transform = transforms.Compose([
                    transforms.ToTensor()
            ])
        else:
            self.transform = transformation
        
        # define augmentation
        if augmentation is None:
            self.augmentation = None  # do nothing
        else:
            self.augmentation = augmentation
            
        return
        
    def expand_path(self, name):
        return self.dir_path + name + ".png"
        
    def __getitem__(self, idx):
        _class, name = tuple(self.df.iloc[idx])
        
        path = self.expand_path(name)
        pil_image = Image.open(path)
        img_arr = self.transform(pil_image)
        
        # cut off 4-channel if exists
        img_arr = img_arr[:3, :, :]
        
        if self.augmentation is not None:
            img_arr = self.augmentation(img_arr)

        return img_arr, _class
    
    def __len__(self):
        return len(self.df)
    

class PlaneSet2Neurons(PlaneSet):
    def __init__(self, dir_path, df, transformation=None, augmentation=None):
        super().__init__(dir_path, df)
        
        self.dir_path = dir_path
        self.df = df
        
    def __getitem__(self, idx):
        _class, name = tuple(self.df.iloc[idx])
        
        if _class == 0:
            _class = torch.tensor([1, 0])
        elif _class == 1:
            _class = torch.tensor([0, 1])
        
        path = self.expand_path(name)
        pil_image = Image.open(path)
        img_arr = self.transform(pil_image)
        
        # cut off 4-channel if exists
        img_arr = img_arr[:3, :, :]
        
        # for inceptionv4 only!
        img_arr = transforms.Resize(299)(img_arr)
        
        if self.augmentation is not None:
            img_arr = self.augmentation(img_arr)

        return img_arr, _class

def configurate_xy_tensors(x, y):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = x.to(device=device)
    y = y.to(device=device)
    x = x.to(torch.float32)
    y = y.to(torch.float32)
    y = y.unsqueeze(1)
    
    return x, y


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


def build_dataset(df, images_path, augmentation):
    return PlaneSet(images_path, df, augmentation=augmentation)
