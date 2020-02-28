import os
import pickle
from collections import namedtuple

import torch
from torch.utils.data import Dataset
from torchvision import datasets
import lmdb
from torchvision.datasets import VisionDataset

CodeRow = namedtuple('CodeRow', ['top', 'bottom', 'filename'])

class LMDBDataset(VisionDataset) :
  
  def __init__(self, path, train=True, transform=None, target_transform=None):
    super(LMDBDataset, self).__init__(path, transform=transform, target_transform=target_transform)
    self.train = train  # training set or test set
    self.env = lmdb.open(
        path,
        max_readers=32,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
    )
    if not self.env :
      raise IOError('Cannot open lmdb dataset', path)
    
    with self.env.begin(write=False) as txn:
      self.length = int(txn.get('length'.encode('utf-8')).decode('utf-8'))
    
    self.data = torch.ones([self.length,32,32])
    self.target = torch.ones([self.length], dtype=torch.int32)         

  def get_data(self) :
    with self.env.begin(write=False) as txn:
      for i in range(self.__len__()):
        key = str(i).encode('utf-8')
        row = pickle.loads(txn.get(key))
        self.data[i], self.target[i] = torch.from_numpy(row.top), 1
    code_train_t = (self.data, self.target)
    model_save_name = 'train_c.pt'
    path = F"/content/gdrive/My Drive/test_colab/vq-vae-2-pytorch/{model_save_name}" 
    torch.save(code_train_t, path)

    model_save_name = 'test_c.pt'
    path = F"/content/gdrive/My Drive/test_colab/vq-vae-2-pytorch/{model_save_name}" 
    torch.save(code_train_t, path)
    return self.data, self.target

  def __len__(self):
    return self.length

  def __getitem__(self, index):
    target = torch.ones(self.__len__(),dtype=torch.int32)
    with self.env.begin(write=False) as txn:
        key = str(index).encode('utf-8')
        row = pickle.loads(txn.get(key))
        data_top = torch.from_numpy(row.top)
    if self.transform is not None :
      data_top = self.transform(data_top)  
    return data_top, target
