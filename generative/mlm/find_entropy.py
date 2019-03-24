import argparse
import os
import sys
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import v5.cnn as cnn


def make_batches(orig_tens, delta_t, batch_size):
    _, channels, _, total_w = orig_tens.shape
    start_t = 0
    while start_t + delta_t < total_w:
        batch = []
        for _ in range(batch_size):
            if start_t + delta_t >= total_w:
                break
            else:
                batch += [orig_tens[:, :, :, start_t : start_t + delta_t]]
                start_t += 1

        batch = torch.cat(batch, 0)
        print('batch shape: {}'.format(batch.shape), file=sys.stderr)
        yield batch


def calc_entropy(pred_batch):
    entropies = torch.zeros(pred_batch.shape[0])
    for j in range(pred_batch.shape[0]):
        running_total = 0.
        for i in range(pred_batch.shape[2]):
            p = pred_batch[j, 0, i, -1]
            if abs(p) > 1e-5 and abs(p) < 1.:
                try:
                    running_total += -p*math.log2(p) - (1. - p)*math.log2(1. - p)
                except ValueError as err:
                    print('exception occurred: {}\np={}\n'.format(err, p), file=sys.stderr)
        entropies[j] = running_total / pred_batch.shape[2]
    return entropies


def find_entropies(net, pr_tensor, cuda_dev, batch_size, max_w=1024, stride=10):
    net.train(False)
    entropies, generated, num_generated = [], torch.zeros(pr_tensor.shape), 0
    generated_list = []
    for batch in make_batches(pr_tensor, max_w, batch_size):
        batch = batch.cuda(cuda_dev)
        predicted = torch.sigmoid(net(batch))

        # use predicted activations from last column
        num_samples, _, h, w = predicted.shape
        for i in range(num_samples):
            output = predicted[i, 0, :, w - 1]
            # generated[0, 0, :, num_generated + i] = (output > 0.5).type(torch.LongTensor)
            # print(output, file=sys.stderr)
            generated_list.append((output > 0.5).type(torch.LongTensor).unsqueeze(1))
        num_generated += num_samples

        # find average binary entropy of last column
        ent = calc_entropy(predicted)
        entropies.append(ent)

    if num_generated == pr_tensor.shape[-1]:
        print('number of generated timeframes matches original', file=sys.stderr)
    else:
        print('generated timeframes: {}\noriginal timeframes: {}'.format(num_generated, pr_tensor.shape[-1]),
              file=sys.stderr)

    generated = torch.cat(generated_list, dim=1)
    print('generated dimensions: {}'.format(generated.shape), file=sys.stderr)

    return torch.cat(entropies, 0), generated


def main(opts):
    # set cuda device
    if opts.use_cuda is not None:
        print("generating from previous timeframes\nusing CUDA device {}".format(opts.use_cuda), file=sys.stderr)
        cuda_dev = int(opts.use_cuda)
    else:
        cuda_dev = None
    sys.stdout.flush()

    # load and prepare piano roll matrix
    piano_roll_path = os.path.join(os.getcwd(), opts.target)
    piano_roll_array = np.load(piano_roll_path)
    pr_tensor = torch.from_numpy(piano_roll_array).type(torch.FloatTensor)
    pr_tensor = F.pad(pr_tensor, (opts.left_pad, 0), 'constant', 0.)
    pr_tensor = pr_tensor.unsqueeze(0).unsqueeze(0)

    # initialize network
    net = cnn.LanguageModeler(1, 32, 64)

    if opts.load:
        saved_state = torch.load(opts.load, map_location='cpu')
        net.load_state_dict(saved_state)
    if cuda_dev is not None:
        net = net.cuda(cuda_dev)

    # find entropy sequence
    with torch.no_grad():
        entropies, generated = find_entropies(net, pr_tensor, cuda_dev, opts.batch_size, max_w=opts.max_w, stride=opts.stride)
        print(','.join(['{:.4}'.format(entropies[i]) for i in range(entropies.shape[0])]))
        np.save('generated', generated)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="preprocessed")
    parser.add_argument("--max_w", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("-c", "--use_cuda", type=int, default=None)
    parser.add_argument("--load", default=None)
    parser.add_argument("--target")
    parser.add_argument("--left_pad", type=int, default=1024)
    parser.add_argument("--stride", type=int, default=10)

    args = parser.parse_args()
    main(args)