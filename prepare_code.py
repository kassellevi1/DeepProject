import os
import pickle
from collections import namedtuple

import torch
from torch.utils.data import Dataset
from torchvision import datasets
import lmdb
from torchvision.datasets import VisionDataset
import argparse



CodeRow = namedtuple('CodeRow', ['top', 'bottom', 'filename'])

parser = argparse.ArgumentParser(description='Train i-ResNet/ResNet on Cifar')
parser.add_argument('--option', default='glo', type=str, help='option')
parser.add_argument('--dataset', default='mnist', type=str, help='dataset')
parser.add_argument('--code_dir', default=None, type=str, help='directory to extract the code')

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
  
  def __len__(self):
    return self.length

  def save_vq(self) :
    with self.env.begin(write=False) as txn:
      for i in range(self.__len__()):
        key = str(i).encode('utf-8')
        row = pickle.loads(txn.get(key))
        print("Roee")
        self.data[i], self.target[i] = torch.from_numpy(row.top), 1
    code_train_t = (self.data, self.target)
    model_save_name = 'train_c.pt'
    path = F"./data/vqvae/{model_save_name}" 
    torch.save(code_train_t, path)

    model_save_name = 'test_c.pt'
    path = F"./data/vqvae/{model_save_name}" 
    torch.save(code_train_t, path)
    return self.data, self.target



  def __getitem__(self, index):
    target = torch.ones(self.__len__(),dtype=torch.int32)
    with self.env.begin(write=False) as txn:
        key = str(index).encode('utf-8')
        row = pickle.loads(txn.get(key))
        data_top = torch.from_numpy(row.top)
    if self.transform is not None :
      data_top = self.transform(data_top)  
    return data_top, target
	
def save_glo(path,dataset) :
  net = torch.load(os.path.join(path, dataset))
  net_reshape = netz['emb.weight'].reshape(60000,8,8)
  train = net_reshape[0:54000,:,:]
  target_train = torch.ones(54000, dtype=torch.int32)
  test = net_reshape[54000:60000,:,:]
  target_test = torch.ones(6000, dtype=torch.int32)

  net_train = (train, target_train)
  model_save_name = 'train_c.pt'
  path = F"/content/gdrive/My Drive/test_colab/DeepProject/data/vqvae/{model_save_name}"
  torch.save(net_train, path)

  net_test = (test, target_test)
  model_save_name = 'test_c.pt'
  path = F"/content/gdrive/My Drive/test_colab/DeepProject/data/vqvae/{model_save_name}"
  torch.save(net_test, path)
def main():
  args = parser.parse_args()

  if args.option == 'glo' :
    save_glo(args.code_dir, args.dataset)

  if args.option == 'vqvae' :
    path = os.path.join('/content/gdrive/My Drive/test_colab/DeepProject',args.code_dir)
    lmbd_set = LMDBDataset(path)
    lmbd_set.save_vq()

if __name__ == '__main__':
    main()
