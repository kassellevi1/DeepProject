# title - levi
Description - levi

## Requisite

levi - tanfiz
* Python >= 3.6
* PyTorch >= 1.1
* lmdb (for storing extracted codes)

Install dependencies:

Dependencies can be installed via pip install -r requirements.txt

Note: You need to run visdom server and set vis_server location as well as port.

## Usage

Currently supports 256px (top/bottom hierarchical prior) - levi

## 1. Stage 1: extract code from dataset via GLO/VQVAE2

1.1 option 1 train code with GLO

Edit prepare_[dataset_name].py with the correct path to the data.


Prepare a dataset:

```bash
python prepare_‫dataset_name]‬].py
```

Train GLO on a particular config:

```bash
python train_glo.py configs/[dataset_name].yaml
```

1.2 option 2 train code with vqvae2

```bash
python train_vqvae.py [DATASET PATH]
```

Extract codes from vqvae2 training

```bash
python extract_code.py --ckpt checkpoint/[VQ-VAE CHECKPOINT] --name [NAME_OF_LMBD] runs\
```

## 2. Stage 2 density estimation based on I-ResNet technique

prepare code for i-resent density estimation

```bash
python prepare_code --option [glo/vqvae]  --path runs/[DIRECTORY]  
```

Train i-ResNet density model on codes (Batch size and learning rate optimized for 4GPUs, if you have only one GPU divide batch size by 4):


Note: You need to run visdom server and set vis_server location as well as port. also specipy in dataset argument glo/vqvae


```bash
scripts/dens_est_code.sh
```


## Sample
Now, for sampling generated images use this command 
For the option 1:

```bash
python sample.py --option glo configs/mnist.yaml  
```

For the option 2:

First train the pixelSnail model for the bottom code
	
```bash
python train_pixelsnail.py --hier bottom [LMDB NAME]
```

```bash
python sample.py --option vqvae --vqvae [PATH TO VQVAE MODEL] --top [PATH TO I-ResNet MODEL] --bottom [PATH TO PixelSnail BOTTOM MODEL] [NAME_OF_FILE]
```
