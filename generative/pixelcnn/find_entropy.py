import argparse
import os
import sys
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import pixel_cnn


def make_batch(orig_tens, t_0, delta_t, stride, batch_size):
    # orig_tens: 1, ch, h, w
    _, channels, _, total_w = orig_tens.shape
    batch = []
    for i in range(batch_size):
        start = t_0 + i*stride
        end = start + delta_t
        if end >= total_w:
            break
        else:
            window = orig_tens[:, :, :, start : end]
            window[:, :, :, -1] = 0.
            batch.append(window)
    if batch == []:
        return None
    else:
        return torch.cat(batch, 0)


def make_batches(orig_tens, delta_t, stride, batch_size):
    batch_w = batch_size*stride
    _, channels, _, total_w = orig_tens.shape
    for i in range(0, total_w, batch_w):
        batch = make_batch(orig_tens, i, delta_t, stride, batch_size)
        if batch is not None:
            yield batch


def calc_entropy(pred_batch):
    entropies = torch.zeros(pred_batch.shape[0])
    for j in range(pred_batch.shape[0]):
        running_total = 0.
        for i in range(pred_batch.shape[2]):
            p = pred_batch[j, 0, i, -1]
            if abs(p) > 1e-8:
                running_total += -p*math.log2(p) - (1. - p)*math.log2(1. - p)
        entropies[j] = running_total / pred_batch.shape[2]
    return entropies


def find_entropies(net, pr_tensor, cuda_dev, batch_size, max_w=1024, stride=10):
    net.train(False)
    entropies = []
    for batch in make_batches(pr_tensor, max_w, stride, batch_size):
        batch = batch.cuda(cuda_dev)
        predicted = torch.sigmoid(net(batch))
        # find average binary entropy of last column
        ent = calc_entropy(predicted)
        entropies.append(ent)
    return torch.cat(entropies, 0)


def main(opts):
    # training script
    if opts.use_cuda is not None:
        print("using CUDA device {}".format(opts.use_cuda))
        cuda_dev = int(opts.use_cuda)
    else:
        cuda_dev = None
    # torch.manual_seed(opts.seed)
    # print("random seed {}".format(opts.seed))
    sys.stdout.flush()

    piano_roll_path = os.path.join(os.getcwd(), opts.target)
    piano_roll_array = np.load(piano_roll_path)
    pr_tensor = torch.from_numpy(piano_roll_array).unsqueeze(0).unsqueeze(0).type(torch.FloatTensor)

    net = pixel_cnn.PixelCNN(1, 32, 64)

    if opts.load:
        saved_state = torch.load(opts.load, map_location='cpu')
        net.load_state_dict(saved_state)
    if cuda_dev is not None:
        net = net.cuda(cuda_dev)

    with torch.no_grad():
        entropies = find_entropies(net, pr_tensor, cuda_dev, opts.batch_size, max_w=opts.max_w)
        print(','.join(['{:.4}'.format(entropies[i]) for i in range(entropies.shape[0])]))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="preprocessed")
    parser.add_argument("--max_w", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("-c", "--use_cuda", type=int, default=None)
    parser.add_argument("--load", default=None)
    parser.add_argument("--target")

    args = parser.parse_args()
    main(args)