import argparse
import glob, os, sys

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from torch.autograd import Variable

import v2.cnn as pixel_cnn


# def generate(net, start_roll, opts):
#     net.train(False)
#     start_roll = torch.from_numpy(start_roll)
#     start_roll = F.pad(start_roll, (opts.left_pad, 0), 'constant', 0.)
#     sample = torch.zeros(start_roll.shape[0], opts.max_w)
#     sample[:, opts.left_pad: opts.left_pad + opts.initial_frames] = \
#         start_roll[:, : opts.initial_frames]
#         # torch.from_numpy(start_roll[:, : opts.initial_frames])
#     sample = sample.unsqueeze(0).unsqueeze(0)
#
#     if opts.use_cuda is not None:
#         sample = sample.cuda(opts.use_cuda)
#
#     for i in range(opts.left_pad + opts.initial_frames, opts.max_w):
#         for j in range(start_roll.shape[0]):
#             z = net(Variable(sample))
#             p = torch.sigmoid(z[0, 0, j, i]).data
#             sample[0, 0, j, i] = torch.bernoulli(p)
#
#         if i % 100 == 0:
#             print('{} timeframes generated'.format(i + 1), file=sys.stderr)
#
#     return sample.squeeze(0).squeeze(0).cpu().numpy()


def generate(net, start_roll, opts):
    net.train(False)
    start_roll = torch.from_numpy(start_roll)

    p_one = torch.sum(start_roll).item() / torch.numel(start_roll)

    sample = torch.zeros(start_roll.shape[0], opts.max_w)

    ps = torch.zeros(sample.shape)

    sample[:, : opts.initial_frames] = start_roll[:, : opts.initial_frames]
    sample = sample.unsqueeze(0).unsqueeze(0)

    if opts.use_cuda is not None:
        sample = sample.cuda(opts.use_cuda)


    for i in range(opts.initial_frames, opts.max_w):
        for j in range(start_roll.shape[0]):
            z = net(Variable(sample))
            p = torch.sigmoid(z[0, 0, j, i]).data
            # p *= p_one
            sample[0, 0, j, i] = torch.bernoulli(p)
            ps[j, i] = p

        if i % 100 == 0:
            print('{}/{} timeframes generated'.format(i - opts.initial_frames + 1, opts.max_w - opts.initial_frames),
                  file=sys.stderr)

    return sample.squeeze(0).squeeze(0).cpu().numpy()


def main(opts):
    if opts.use_cuda is not None:
        print("using CUDA device {}".format(opts.use_cuda))
        cuda_dev = int(opts.use_cuda)
    else:
        cuda_dev = None

    torch.manual_seed(opts.seed)
    print("random seed {}".format(opts.seed))
    sys.stdout.flush()

    # net = pixel_cnn.PixelCNN(1, 32, 64)
    net = pixel_cnn.AutoEncoder(opts.batch_size, cuda_dev, opts.max_w)

    saved_state = torch.load(opts.load, map_location='cpu')
    net.load_state_dict(saved_state)
    if cuda_dev is not None:
        net = net.cuda(cuda_dev)

    input_path = os.path.join(os.getcwd(), opts.input)
    starter_roll = np.load(input_path)

    with torch.no_grad():
        generated = generate(net, starter_roll, opts)

        save_path = os.path.join(os.getcwd(), opts.save)
        np.save(save_path, generated)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_w", type=int, default=1024)
    parser.add_argument("-s", "--seed", type=int, default=0)
    parser.add_argument("-c", "--use_cuda", type=int, default=None)
    parser.add_argument("--load", default=None)
    parser.add_argument("--input", default=None)
    parser.add_argument("--initial_frames", "-i", type=int, default=256)
    parser.add_argument("--save", default='generated.npy')
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()
    main(args)

