from __future__ import print_function
import torch.utils.data as data
from PIL import Image
import os
import os.path
import errno
import torch
import codecs
import torchvision.transforms as transforms
from torchvision.datasets import VisionDataset


class vqvae_code(VisionDataset):
    
    @property
    def train_labels(self):
        warnings.warn("train_labels has been renamed targets")
        return self.targets

    @property
    def test_labels(self):
        warnings.warn("test_labels has been renamed targets")
        return self.targets

    @property
    def train_data(self):
        warnings.warn("train_data has been renamed data")
        return self.data

    @property
    def test_data(self):
        warnings.warn("test_data has been renamed data")
        return self.data
    
    def __init__(self, root, train=True, transform=None, target_transform=None):
        super(tiny_imagenet, self).__init__(root, transform=transform,
                                    target_transform=target_transform)
        self.train = train  # training set or test set
        
        
        self.data, self.targets = torch.load(root)
        self.targets = self.targets
        
            
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        code_d, target = self.data[index], int(self.targets[index])
        code_d = Image.fromarray(code_d.numpy(), mode='L')
        if self.transform is not None:
            code_d = self.transform(code_d)    
        return code_d, target
    def extra_repr(self):
        return "Split: {}".format("Train" if self.train is True else "Test")