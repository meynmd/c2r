import argparse
import glob
import os
import sys
import time
import random
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from torch.autograd import Variable
from torch import cuda

import convnet
from copy import deepcopy

# make batch_size overlapping windows
def make_cropped_batch(tensor, num_windows, window_size):
    h, w = tensor.shape[0], tensor.shape[1]
    windows = torch.zeros(num_windows, 1, tensor.shape[0], window_size)
    stride = (w - window_size) // (num_windows - 1)
    for i in range(num_windows):
        windows[i, :, :, :] = tensor[:, i*stride: i*stride + window_size]
    return windows, stride


# piano_roll : numpy array
def make_windows(piano_roll, width, stride):
    h, w = piano_roll.shape
    windows, start = [], 0
    while start + width < w:
        windows.append(np.expand_dims(piano_roll[:, start : start + width], axis=0))
        start += stride
    windows.append(np.expand_dims(piano_roll[:, -width :], axis=0))
    return np.stack(windows)


def pad_piano_roll(piano_roll, width_final):
    h, w = piano_roll.shape
    pr_new = np.zeros((h, width_final))
    pr_new[h, w] = piano_roll[:, :]
    return pr_new


# def make_batches(datafiles, target_file, window_size, stride=500):
#     xs, ys = [], []
#     for file in datafiles:
#         label, piano_roll = file.split('/')[-1].split('.')[0], np.load(file)
#         target_file = target_file.split('/')[-1].split('.')[0]
#         piano_roll = piano_roll > 0
#         piano_roll = piano_roll.astype(float)
#         x = piano_roll
#         if x.shape[1] == window_size:
#             # x = torch.from_numpy( x ).unsqueeze(0).unsqueeze(0)
#             x = np.expand_dims(np.expand_dims(x, axis=0), axis=0)
#         elif x.shape[1] > window_size:
#             #x = make_cropped_batch(x, batch_size, window_size)
#             x = make_windows(x, window_size, stride)
#         else:
#             x = pad_piano_roll(x)
#         xs.append(x)
#         if label == target_file:
#             label = 1.
#         else:
#             label = 0.
#         ys += [label for _ in range(x.shape[0])]
#
#     real_xs = np.concatenate(xs)
#
#     x = np.load('data/' + target_file + '.mid.npy')
#     x = x > 0
#     x = x.astype(float)
#     if x.shape[1] == window_size:
#         x = np.expand_dims(np.expand_dims(x, axis=0), axis=0)
#     elif x.shape[1] > window_size:
#         x = make_windows(x, window_size, stride)
#     else:
#         x = pad_piano_roll(x)
#     num_extra = real_xs.shape[0] // x.shape[0]
#     for _ in range(num_extra):
#         xs.append(deepcopy(x))
#         ys += [1. for _ in range(x.shape[0])]
#
#     return np.concatenate(xs), ys

def make_batches(datafiles, label2idx, window_size, stride=500):
    xs, ys = [], []
    for file in datafiles:
        label, piano_roll = file.split('/')[-1].split('.')[0], np.load(file)
        # target_file = target_file.split('/')[-1].split('.')[0]
        piano_roll = piano_roll > 0
        piano_roll = piano_roll.astype(float)
        x = piano_roll
        if x.shape[1] == window_size:
            x = np.expand_dims(np.expand_dims(x, axis=0), axis=0)
        elif x.shape[1] > window_size:
            x = make_windows(x, window_size, stride)
        else:
            x = pad_piano_roll(x)
        xs.append(x)
        label = label2idx[label]
        ys += [label for _ in range(x.shape[0])]

    # real_xs = np.concatenate(xs)

    # x = np.load('data/' + target_file + '.mid.npy')
    # x = x > 0
    # x = x.astype(float)
    # if x.shape[1] == window_size:
    #     x = np.expand_dims(np.expand_dims(x, axis=0), axis=0)
    # elif x.shape[1] > window_size:
    #     x = make_windows(x, window_size, stride)
    # else:
    #     x = pad_piano_roll(x)
    # num_extra = real_xs.shape[0] // x.shape[0]
    # for _ in range(num_extra):
    #     xs.append(deepcopy(x))
    #     ys += [1. for _ in range(x.shape[0])]

    return np.concatenate(xs), ys

def copy_batch(xs, ys, idxs):
    x_batch = torch.FloatTensor(len(idxs), xs.shape[1], xs.shape[2], xs.shape[3])
    y_batch = torch.LongTensor(len(idxs))
    for i, idx in enumerate(idxs):
        x_batch[i, :, :, :] = xs[idx, :, :, :]
        y_batch[i] = ys[idx]
    return x_batch, y_batch


def train(model, xs, ys, loss_fn, optim, batch_size=16, num_epochs=50, model_dir="model", cuda_dev=None, lr_init=1e-5, lr_decay=None, window_size=1000):
    best_loss, patience = float("inf"), 20
    model.train()

    for epoch in range(num_epochs):
        idxs = random.sample(range(xs.shape[0]), xs.shape[0])
        num_batches = xs.shape[0] // batch_size
        running_loss, running_err = 0., 0.
        print("\nEpoch {}\n".format(epoch + 1) + 80*"*")

        for batch_num in range(num_batches):
            x_batch, y_batch = copy_batch(xs, ys, idxs[batch_num*batch_size : (batch_num + 1)*batch_size])
            y_batch = Variable( y_batch )
            if cuda_dev is not None:
                x_batch = x_batch.cuda(cuda_dev)
                y_batch = y_batch.cuda(cuda_dev)

            optim.zero_grad()

            # run and calc loss
            z = model(x_batch)
            loss = loss_fn(z, y_batch)
            running_loss += loss.data[0]

            loss.backward()
            optim.step()

            # make best prediction and find err
            _, y_hat = torch.max(z, 1)
            running_err += torch.sum(y_hat.data != y_batch.data)

        # print progress
        avg_loss = running_loss / float(num_batches)
        print("err: {:.0%}\t loss: {:.5}\n".format(running_err / float(batch_size) / float(num_batches), avg_loss))

        save_name = "model_epoch_{}_loss{:.3}_".format(epoch + 1, avg_loss)
        save_name += "-".join(time.asctime().split(" ")[:-1]).replace(":", ".")
        if avg_loss < best_loss:
            save_name = "model_best.pt"
        save_path = "{}/{}".format(model_dir, save_name)
        if epoch % 100 == 0 or epoch == num_epochs - 1 or avg_loss < best_loss:
            torch.save(model.state_dict(), save_path)
            print("Model saved to {}".format(save_path))


def load_model(dest_dict, source_dict, exclude):
    for param in dest_dict:
        if param in exclude:
            print('excluding parameter {}'.format(param), file=sys.stderr)
        if param in source_dict:
            dest_dict[param] = source_dict[param]
        else:
            raise KeyError('cannot find parameter {} in the checkpoint'.format(param))


def main(opts):
    # basic parameters
    exclude_params = ['fc1.weight', 'fc2.weight', 'fc3.weight']
    if opts.use_cuda is not None:
        print("using CUDA device {}".format(opts.use_cuda), file=sys.stderr)
        cuda_dev = int(opts.use_cuda)
    else:
        cuda_dev = None
    torch.manual_seed(opts.seed)
    print("random seed {}\n".format(opts.seed))
    sys.stdout.flush()

    # load input files
    with open('data/labels.csv') as fp:
        idx2y = {int(x) : y for (x, y) in [line.strip().split(',') for line in fp.readlines()]}
        y2idx = {y : x for (x, y) in idx2y.items()}

    input_files = glob.glob(opts.data_dir + '/' + '*.mid.npy')
    xs, ys = make_batches(input_files, y2idx, opts.window, opts.window // 2)
    xs = torch.from_numpy(xs)
    # ys = [y2idx[name] for name in ys]

    # set up the model
    model = convnet.ConvNet(len(input_files), opts.batch_size, use_cuda=opts.use_cuda, window_size=opts.window)
    if opts.load:
        print('loading parameters from {}'.format(opts.load), file=sys.stderr)
        load_model(model.state_dict(), torch.load(opts.load), exclude=exclude_params)
    if cuda_dev is not None:
        model = model.cuda(cuda_dev)

    # set up the loss function and optimizer
    # frac = sum(ys) / (len(ys) - sum(ys))
    # print('pos/neg: {}'.format(frac))
    # w = torch.FloatTensor([frac, 1.])
    # if opts.use_cuda:
    #     w = w.cuda(opts.use_cuda)

    lf = nn.CrossEntropyLoss()
    # lf = nn.BCEWithLogitsLoss()
    # lf = nn.BCEWithLogitsLoss(weight=torch.FloatTensor([1., frac]))

    optim = torch.optim.SGD(model.parameters(), lr=10**(-opts.init_lr), momentum=0.9)

    train(model, xs, ys, lf, optim, opts.batch_size, opts.max_epochs, model_dir=opts.out_model, cuda_dev=cuda_dev)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--target")
    parser.add_argument("--window", type=int, default=1000)
    parser.add_argument("--max_epochs", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("-m", "--out_model", default="out_model")
    parser.add_argument("-s", "--seed", type=int, default=0)
    parser.add_argument("-c", "--use_cuda", type=int, default=None)
    parser.add_argument("-l", "--init_lr", type=int, default=5)
    parser.add_argument("--load", default="input_model/model_loss_0_4.pt")

    args = parser.parse_args()
    main(args)