# title - levi
Description - levi

## Requisite

levi - tanfiz
* Python >= 3.6
* PyTorch >= 1.1
* lmdb (for storing extracted codes)

## Usage

Currently supports 256px (top/bottom hierarchical prior) - levi

1. Stage 1 extract code from dataset GLO/vqvae2

1.1 option 1 train code with GLO

Edit prepare_mnist.py/ prepare_cifar_data.py with the correct path to the data.


Prepare a dataset:
```bash
python prepare_mnist.py
```

Train GLO on a particular config:
```bash
python train_glo.py configs/mnist.yaml

1.2 option 2 train code with vqvae2

> python train_vqvae.py [DATASET PATH]

Extract codes for stage 1.2 training

> python extract_code.py --ckpt checkpoint/[VQ-VAE CHECKPOINT] --name [LMDB NAME] [DATASET PATH]


2. Stage 2 density estimation based on IRESENT technique

prepare code for iresent density estimation

> python prepare_code --path runs/[DIRECTORY]  --name [NAME]

Train i-ResNet density model on codes (Batch size and learning rate optimized for 4GPUs):

if you have only one GPU divide batch size by 4

note - edit script to choose the desire code

```bash scripts/dens_est_code.sh

levi - if you want to add an image
## Sample

### Stage 1

Note: This is a training sample

![Sample from Stage 1 (VQ-VAE)](stage1_sample.png)
