import argparse
import glob, os, sys

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from torch.autograd import Variable

import pixel_cnn


def generate(net, start_roll, opts):
    net.train(False)
    sample = torch.zeros(start_roll.shape[0], opts.max_w)
    sample[:, opts.left_pad: opts.left_pad + opts.initial_frames] = \
        torch.from_numpy(start_roll[:, : opts.initial_frames])
    sample = sample.unsqueeze(0).unsqueeze(0)

    if opts.use_cuda is not None:
        sample = sample.cuda(opts.use_cuda)

    for i in range(opts.left_pad + opts.initial_frames, opts.max_w):
        z = net(Variable(sample))
        p = torch.sigmoid(z[0, 0, :, i]).data
        sample[0, 0, :, i] = torch.bernoulli(p)
        if i % 100 == 0:
            print('{} samples generated'.format(i + 1), file=sys.stderr)
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

    net = pixel_cnn.PixelCNN(1, 32, 64)

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
    parser.add_argument("--left_pad", type=int, default=512)
    parser.add_argument("--input", default=None)
    parser.add_argument("--initial_frames", "-i", type=int, default=100)
    parser.add_argument("--save", default='generated.npy')
    args = parser.parse_args()
    main(args)

