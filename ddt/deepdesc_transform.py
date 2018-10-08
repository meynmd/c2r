import argparse
import glob
import os
import sys
import time
import random
import numpy as np
from collections import defaultdict

from sklearn.decomposition import PCA

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from torch.autograd import Variable
from torch import cuda

from conv_layers import ConvLayers
from fragments_dataset import FragmentsDataset

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


def make_batch(tensors, max_w):
    max_w = min(max_w, min(t.shape[1] for t in tensors))
    x_batch = []
    for tensor in tensors:
        tensor = (tensor > 0.).type(torch.float)
        tensor = tensor.view(1, 1, tensor.shape[0], tensor.shape[1])
        if tensor.shape[3] > max_w:
            tensor = random_crop(tensor, max_w)
        # elif tensor.shape[3] < max_w:
        #     num_padding = max_w - tensor.shape[3]
        #     tensor = torch.nn.functional.pad(tensor, (0, num_padding))
        x_batch.append(tensor)
    return Variable(torch.cat(x_batch, 0))


def ddt(net, dataset, phase, x, mean_vec, trans_vecs, window_size=1024):
    cuda_dev = net.use_cuda
    net.eval()
    # data_list = [(x, x_path) for (x, _), (x_path, _) in dataset.get_from_class(target_class)]
    # data_list = sorted(data_list, key=lambda p : p[1])
    # data_list = sorted(data_list, key=lambda p : len(p[1]))

    # for (x, _) in data_list:
    # batch of 1
    # x_batch = make_batch([x], window_size)
    original_dims = x.shape
    x = x.unsqueeze(0).unsqueeze(0)
    z = net(x).cpu().data
    # z = z[:, :, :, 5:-5]
    z = z.squeeze(0).transpose(0, 2).transpose(0, 1)
    z = z - mean_vec.repeat(z.shape[0], z.shape[1], 1)
    # z[:, :5, :] = torch.zeros(z[:, :5, :].shape)
    # z[:, -5:, :] = torch.zeros(z[:, -5:, :].shape)
    # z[:2, :, :] = torch.zeros(z[:2, :, :].shape)
    # z[-2:, :, :] = torch.zeros(z[-2:, :, :].shape)
    z_sum = torch.zeros(z.shape[0], z.shape[1], trans_vecs.shape[0])
    dotprods = torch.zeros(z.shape[0], z.shape[1], trans_vecs.shape[0])
    for p in range(trans_vecs.shape[0]):
        for i in range(z.shape[0]):
            for j in range(z.shape[1]):
                z_sum[i, j, p] = torch.dot(z[i, j, :] / z[i, j, :].norm(p=2), trans_vecs[p])

            # mags = torch.norm(z[i, :, :], p=2, dim=-1)
            # unit_z = z[i, :, :].transpose(0, 1).div(mags).transpose(0, 1)
            # dotprods[i, :, p] = torch.bmm(
            #     z[i, :, :].view(z.shape[1], 1, z.shape[2]),
            #     trans_vecs[p].repeat(z.shape[1], 1).unsqueeze(2)
            # ).squeeze(-1).squeeze(-1)


    # z_sum[:, :, p] = z_sum[:, :, p] - z_sum[:, :, p].view(-1).min().item()
    # z_sum[:, :, p] = z_sum[:, :, p] / z_sum[:, :, p].view(-1).max().item()

    avg_z_sum = z_sum.mean(2)
    k = 10
    kmax = torch.topk(avg_z_sum.view(-1), k)


    # mask = torch.nn.functional.interpolate(z_sum[:, :, p].unsqueeze(0).unsqueeze(0), size=original_dims)
    # mask = mask.squeeze(0).squeeze(0)
    # weighted_input = mask * x


def find_descriptors_and_mean_vector(net, x):
    cuda_dev = net.use_cuda
    net.eval()
    # data = dataset.get_from_class(classname)
    batch_size = net.batch_size
    descriptor_dim = net.out_channels

    descriptors, means = [], []
    # for j, (x, y) in enumerate(data_loader):
    x = x.unsqueeze(0).unsqueeze(0)
    # z <- [b, d, m, n]
    z = net(x)
    # z <- [b*m*n, d]
    z = z.transpose(0, 1).contiguous().view(descriptor_dim, -1).transpose(0, 1)

    means.append(z.mean(0).cpu())
    descriptors.append(z.cpu())

    return torch.stack(means).mean(0), torch.cat(descriptors, dim=0)


def find_pc(descriptors, mean_vec, num_pc):
    normalized = descriptors - torch.stack([mean_vec for i in range(descriptors.shape[0])])
    pca = PCA()
    pca.fit(normalized.cpu().data.numpy())
    pcs, var = pca.components_, pca.explained_variance_
    return torch.FloatTensor(pcs[0:num_pc, :])


def main(opts):
    if opts.use_cuda is not None:
        print("using CUDA device {}".format(opts.use_cuda), file=sys.stderr)
        cuda_dev = int(opts.use_cuda)
    else:
        cuda_dev = None
    sys.stdout.flush()

    # initialize data loader
    labels_path = os.path.join(os.getcwd(), opts.data_dir, 'labels.csv')

    dataset = FragmentsDataset(os.path.join(os.getcwd(), opts.data_dir), labels_path, 'train')
    dataloader = DataLoader(
            dataset,
            batch_size=opts.batch_size,
            shuffle=True,
            num_workers=0 #, collate_fn=lambda b : list(list(l) for l in zip(*b))
    )

    # set up the model
    net = ConvLayers(
        dataset.get_y_count(),
        opts.batch_size,
        # opts.window_size,
        use_cuda=cuda_dev,
    )
    if opts.load:
        net.load_state_dict(torch.load(opts.load), strict=False)

    for p in net.parameters():
        p.requires_grad = False

    if cuda_dev is not None:
        net = net.cuda(cuda_dev)

    # for classname in opts.classname: #idx, classname in datasets['train'].idx2name.items():
    #     mvec, dscr = find_descriptors_and_mean_vector(dataloader, 'train', net, classname)
    #     pcs = find_pc(dscr, mvec, num_pc=opts.num_pc)
    #     ddt(net, dataset, 'train', classname, mvec, pcs, opts.window_size)

    for filename in opts.classname:
        pr = np.load(filename)
        pr = torch.from_numpy(pr).type(torch.float)
        pr = (pr > 0.).type(torch.float)
        pr = Variable(pr)
        mvec, dscr = find_descriptors_and_mean_vector(net, pr)
        pcs = find_pc(dscr, mvec, num_pc=opts.num_pc)
        ddt(net, dataset, 'train', pr, mvec, pcs, opts.window_size)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="preprocessed")
    parser.add_argument("--classname", action="append")
    parser.add_argument("--window_size", type=int, default=1024)
    parser.add_argument("-c", "--use_cuda", type=int, default=None)
    parser.add_argument("--load", default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("-p","--num_pc", type=int, default=1)

    args = parser.parse_args()
    main(args)