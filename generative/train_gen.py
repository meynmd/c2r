import argparse
import glob
import os
import sys

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from torch.autograd import Variable

from masked_convnet import MaskedConv2d
import pr_dataset


def bottleneck_block(channels_internal, mask_type, ch_in, ch_out, filter, stride, padding, bias):
    return torch.nn.Sequential(
        nn.Conv2d(ch_in, channels_internal, kernel_size=1, stride=1, bias=False),
        MaskedConv2d(mask_type, channels_internal, channels_internal, filter, stride, padding, bias=bias),
        nn.Conv2d(channels_internal, ch_out, kernel_size=1, stride=1, bias=False)
    )


def build_net(num_channels_internal, channels_int=8):
    nc = num_channels_internal
    net = torch.nn.Sequential(
        MaskedConv2d('A', 1, nc, 7, 1, 3, bias=False), nn.BatchNorm2d(nc), nn.ReLU(True),
        bottleneck_block(channels_int, 'B', nc, nc, 7, 1, 3, bias=False), nn.BatchNorm2d(nc), nn.ReLU(True),
        bottleneck_block(channels_int, 'B', nc, nc, 7, 1, 3, bias=False), nn.BatchNorm2d(nc), nn.ReLU(True),
        bottleneck_block(channels_int, 'B', nc, nc, 7, 1, 3, bias=False), nn.BatchNorm2d(nc), nn.ReLU(True),
        bottleneck_block(channels_int, 'B', nc, nc, 7, 1, 3, bias=False), nn.BatchNorm2d(nc), nn.ReLU(True),
        bottleneck_block(channels_int, 'B', nc, nc, 7, 1, 3, bias=False), nn.BatchNorm2d(nc), nn.ReLU(True),
        bottleneck_block(channels_int, 'B', nc, nc, 7, 1, 3, bias=False), nn.BatchNorm2d(nc), nn.ReLU(True),
        nn.Conv2d(nc, 1, 1)
    )
    return net

# def build_net(num_channels_internal):
#     nc = num_channels_internal
#     net = torch.nn.Sequential(
#         MaskedConv2d('A', 1, nc, 7, 1, 3, bias=False), nn.BatchNorm2d(nc), nn.ReLU(True),
#         MaskedConv2d('B', nc, nc, 7, 1, 3, bias=False), nn.BatchNorm2d(nc), nn.ReLU(True),
#         MaskedConv2d('B', nc, nc, 7, 1, 3, bias=False), nn.BatchNorm2d(nc), nn.ReLU(True),
#         MaskedConv2d('B', nc, nc, 7, 1, 3, bias=False), nn.BatchNorm2d(nc), nn.ReLU(True),
#         MaskedConv2d('B', nc, nc, 7, 1, 3, bias=False), nn.BatchNorm2d(nc), nn.ReLU(True),
#         MaskedConv2d('B', nc, nc, 7, 1, 3, bias=False), nn.BatchNorm2d(nc), nn.ReLU(True),
#         MaskedConv2d('B', nc, nc, 7, 1, 3, bias=False), nn.BatchNorm2d(nc), nn.ReLU(True),
#         MaskedConv2d('B', nc, nc, 7, 1, 3, bias=False), nn.BatchNorm2d(nc), nn.ReLU(True),
#         nn.Conv2d(nc, 1, 1)
#     )
#     return net


def train(net, dataloader, neg_pos, optim, num_epochs=50, model_dir="model", cuda_dev=None, lr_decay=0., max_w=1024):
    best_loss = float("inf")
    phases = ['train', 'val']
    for epoch in range(num_epochs):
        optim.param_groups[0]['lr'] *= (1. - lr_decay)

        print("\n" + 80*"*" + "\nEpoch {}\tlr {}\n".format(epoch + 1, optim.param_groups[0]['lr']))

        for phase in phases:
            running_loss, err = 0., 0.
            if phase == "train":
                net.train()
            else:
                net.eval()

            running_loss = 0.
            for i, data in enumerate(dataloader[phase]):
                x, _ = data
                batch_size = len(x)
                x = make_batch(x, max_w, cuda_dev)
                optim.zero_grad()
                target = x.clone()
                yh = net(x)
                w = torch.ones(target.shape) * neg_pos
                w = w.cuda(cuda_dev)
                loss = F.binary_cross_entropy_with_logits(yh, target, pos_weight=w)
                loss.backward()
                optim.step()

                if phase == 'train':
                    print('iter {:3}\ttrain loss: {:.3}'.format(i+1, loss))
                else:
                    running_loss += loss

            if phase == 'val':
                avg_loss = running_loss / (i+1)
                print('epoch {}\tval loss: {:.3}'.format(epoch+1, avg_loss))
            print(80 * '-')




def random_crop(tensor, w_new, h_new=None):
    h, w = tensor.shape[2], tensor.shape[3]
    top, left = 0, 0
    if w_new < w:
        left = np.random.randint(0, w - w_new)
    if h_new is None:
        return tensor[:, :, :, left: left + w_new]
    if h_new < h:
        top = np.random.randint(0, h - h_new)
    return tensor[top: top + h_new, left: left + w_new]


def make_batch(tensors, max_w, cuda_dev=None):
    batch_size = len(tensors)
    max_w = min(max_w, min(t.shape[1] for t in tensors))

    x_batch = []
    for tensor in tensors:
        # tensor = (tensor > 0.).type(torch.float)
        tensor = tensor.view(1, 1, tensor.shape[0], tensor.shape[1])
        if tensor.shape[3] > max_w:
            tensor = random_crop(tensor, max_w)
        elif tensor.shape[3] < max_w:
            tensor = torch.nn.functional.pad(tensor, (0, max_w - tensor.shape[3], 0, 0))
        assert(tensor.shape[3] == max_w)
        x_batch.append(tensor)

    if cuda_dev is None:
        x_batch = Variable(torch.cat(x_batch, 0))
    else:
        x_batch = Variable(torch.cat(x_batch, 0)).cuda(cuda_dev)

    return x_batch


def main(opts):
    # training script
    if opts.use_cuda is not None:
        print("using CUDA device {}".format(opts.use_cuda))
        cuda_dev = int(opts.use_cuda)
    else:
        cuda_dev = None
    torch.manual_seed(opts.seed)
    print("random seed {}".format(opts.seed))
    sys.stdout.flush()

    # initialize data loader
    datasets = { p : pr_dataset.PianoRollDataset(os.getcwd() + "/" + opts.data_dir, "labels.csv", p)
                 for p in ("train", "val") }
    dataloaders = {
        p : DataLoader(
            datasets[p],
            batch_size=opts.batch_size if p == "train" else 8,
            shuffle=True,
            num_workers=0,
            collate_fn=lambda b : list(list(l) for l in zip(*b))
        ) for p in ("train", "val")
    }

    # initialize network
    net = build_net(64)

    if opts.load:
        saved_state = torch.load(opts.load, map_location='cpu')
        net.load_state_dict(saved_state)
    if cuda_dev is not None:
        net = net.cuda(cuda_dev)

    # set up the loss function and optimizer
    # find p(x == 1) for arbitrary x
    num_ones, num_elem = 0., 0.
    for x, _ in dataloaders['train']:
        for sample in x:
            num_ones += torch.sum(sample).item()
            num_elem += torch.numel(sample)
    positive_weight = (num_elem - num_ones) / num_ones

    optim = torch.optim.SGD(net.parameters(), float(opts.init_lr), momentum=0.9)

    os.makedirs(opts.model_dir, exist_ok=True)
    sys.stdout.flush()

    # train(net, dataloader, neg_pos, optim, num_epochs=50, model_dir="model", cuda_dev=None, lr_decay=0.)
    train(net, dataloaders, positive_weight, optim, opts.max_epochs, opts.model_dir, cuda_dev, max_w=opts.max_w)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="../preprocessed")
    parser.add_argument("--max_epochs", type=int, default=12500)
    parser.add_argument("--max_w", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("-m", "--model_dir", default="model")
    parser.add_argument("-s", "--seed", type=int, default=0)
    parser.add_argument("-c", "--use_cuda", type=int, default=None)
    parser.add_argument("-l", "--init_lr", default="10e-5")
    parser.add_argument("--load", default=None)

    args = parser.parse_args()
    main(args)