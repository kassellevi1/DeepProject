import argparse
import os
import numpy as np
import sys
import torch
import yaml
import torch.nn as nn
import torchvision.utils as vutils
from torchvision.utils import save_image
from tqdm import tqdm

from vqvae import VQVAE
from pixelsnail import PixelSNAIL

import model
import density_estimation_main as iresnet

@torch.no_grad()
def sample_model(model, device, batch, size, temperature, condition=None):
    row = torch.zeros(batch, *size, dtype=torch.int64).to(device)
    cache = {}

    for i in tqdm(range(size[0])):
        for j in range(size[1]):
            out, cache = model(row[:, : i + 1, :], condition=condition, cache=cache)
            prob = torch.softmax(out[:, :, i, j] / temperature, 1)
            sample = torch.multinomial(prob, 1).squeeze(-1)
            row[:, i, j] = sample

    return row


def load_model(model, checkpoint, device):
    ckpt = torch.load(os.path.join('checkpoint', checkpoint))

    
    if 'args' in ckpt:
        args = ckpt['args']

    if model == 'vqvae':
        model = VQVAE()

    elif model == 'pixelsnail_top':
        model = PixelSNAIL(
            [32, 32],
            512,
            args.channel,
            5,
            4,
            args.n_res_block,
            args.n_res_channel,
            dropout=args.dropout,
            n_out_res_block=args.n_out_res_block,
        )

    elif model == 'pixelsnail_bottom':
        model = PixelSNAIL(
            [64, 64],
            512,
            args.channel,
            5,
            4,
            args.n_res_block,
            args.n_res_channel,
            attention=False,
            dropout=args.dropout,
            n_cond_res_block=args.n_cond_res_block,
            cond_res_channel=args.n_res_channel,
        )
        
    if 'model' in ckpt:
        ckpt = ckpt['model']

    model.load_state_dict(ckpt)
    model = model.to(device)
    model.eval()

    return model


if __name__ == '__main__':

    device = 'cuda'
    nBlocks = [16, 16, 16]
    nStrides = [1, 2, 2]
    nChannels = [512, 512, 512]
    numSeriesTerms = 5
    coeff = 0.9
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--option', type=str)
    parser.add_argument('--vqvae', type=str)
    parser.add_argument('--top', type=str)
    parser.add_argument('--bottom', type=str)
    parser.add_argument('--temp', type=float, default=1.0)
    parser.add_argument('filename', type=str)

    args = parser.parse_args()
    if args.option == 'vqvae':
        net_iresnet = iresnet.get_model(nBlocks, nStrides, nChannels, coeff, densityEstimation, numSeriesTerms)
        state_dict = torch.load('runs/iresnet/checkpoint.t7')
        net_iresnet.load_state_dict(state_dict)

        model_vqvae = load_model('vqvae', args.vqvae, device)

        model_bottom = load_model('pixelsnail_bottom', args.bottom, device)

        top_sample = net_iresnet(torch.randn(32, 32).cuda())
        bottom_sample = sample_model(
            model_bottom, device, args.batch, [64, 64], args.temp, condition=top_sample
            )

        decoded_sample = model_vqvae.decode_code(top_sample, bottom_sample)
        decoded_sample = decoded_sample.clamp(-1, 1)

        save_image(decoded_sample, args.filename, normalize=True, range=(-1, 1))
    if args.option == 'glo':
        parser = argparse.ArgumentParser(
            description='Train ICP.'
        )
        parser.add_argument('config', type=str, help='Path to cofig file.')

        args = parser.parse_args()

        try:
            from yaml import CLoader as Loader, CDumper as Dumper
        except ImportError:
            from yaml import Loader, Dumper

        with open(args.config, 'r') as f:
            params = yaml.load(f, Loader=Loader)

        rn = params['name']
        nc = params['nc']
        sz = params['sz']
        nz = params['glo']['nz']
        do_bn = params['glo']['do_bn']
        net_GLO = model._netG(nz, sz, nc, do_bn).cuda()
        state_dict = torch.load('runs/nets_%s/netG_nag.pth' % (rn))
        net_GLO.load_state_dict(state_dict)

        net_iresnet = iresnet.get_model(in_shape, nBlocks, nStrides, nChannels, 2, coeff, args.densityEstimation,numSeriesTerms)

        state_dict = torch.load('runs/iresnet/checkpoint.t7')
        net_iresnet.load_state_dict(state_dict)

        z = net_iresnet(torch.randn(64, 32).cuda())
        ims = net_GLO(z)
        vutils.save_image(ims,
                          'runs/ims_%s/samples.png' % (rn),
                          normalize=False)
