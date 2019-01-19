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
import scipy.ndimage as ndimage

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


def make_batches(file, window_size, stride=500):
    label, piano_roll = file.split('/')[-1].split('.')[0], np.load(file)
    piano_roll = piano_roll > 0
    piano_roll = piano_roll.astype(float)
    x = piano_roll
    if x.shape[1] == window_size:
        x = np.expand_dims(np.expand_dims(x, axis=0), axis=0)
    elif x.shape[1] > window_size:
        x = make_windows(x, window_size, stride)
    else:
        x = pad_piano_roll(x)

    label = 1
    y = [label for _ in range(x.shape[0])]

    return x, y


def copy_batch(xs, ys, idxs):
    x_batch = torch.FloatTensor(len(idxs), xs.shape[1], xs.shape[2], xs.shape[3])
    y_batch = torch.LongTensor(len(idxs))
    for i, idx in enumerate(idxs):
        x_batch[i, :, :, :] = xs[idx, :, :, :]
        y_batch[i] = ys[idx]
    return x_batch, y_batch


def train(
        model, xs, ys, loss_fn, batch_size=16, num_epochs=50, cuda_dev=None, lr_init=1e-5,
        lr_decay=None, window_size=1000
    ):
    model.eval()
    num_batches = xs.shape[0] // batch_size + 1
    predictions = np.zeros(model.num_cat)

    pr = np.zeros((xs.shape[-2], xs.shape[0] * xs.shape[-1]))

    for batch_num in range(num_batches):
        start, end = batch_num*batch_size, min(xs.shape[0], (batch_num + 1)*batch_size)
        x_batch = xs[start : end, :, :, :]
        y_batch = Variable( torch.LongTensor(ys[start : end]) )
        if cuda_dev is not None:
            x_batch = x_batch.cuda(cuda_dev)
            y_batch = y_batch.cuda(cuda_dev)


        z = model(x_batch)
        s = torch.nn.functional.softmax(z, dim=1)
        prediction = torch.sum(torch.max(s, dim=1)[1]).data[0]

        print('predicted: {}'.format(prediction))

        # find activations
        activations = model.conv_activations(x_batch)
        activations = activations.cpu().data

        time_factor = x_batch.shape[-1] / activations.shape[-1]
        pitch_factor = x_batch.shape[-2] / activations.shape[-2]

        for i in range(activations.shape[0]):
            batch = x_batch.cpu()[i, :, :, :]
            activation = activations[i, :, :, :]
            flattened_act = torch.max(activation, dim=0)[0].squeeze(0).numpy()
            flattened_act = flattened_act - flattened_act.mean()
            flattened_act = flattened_act - np.min(flattened_act)
            zoomed_act = ndimage.zoom(flattened_act, (pitch_factor, time_factor))
            heatmap = zoomed_act * batch.squeeze(0).squeeze(0).numpy()
            left, right = i * heatmap.shape[1], (i + 1) * heatmap.shape[1]
            pr[:, batch_num*batch_size + left : batch_num*batch_size + right] = np.maximum(pr[:, batch_num*batch_size + left : batch_num*batch_size + right], heatmap[:, :])
    np.save('heatmap_piano_roll', pr)







def load_model(dest_dict, source_dict, exclude):
    for param in dest_dict:
        if param in exclude:
            print('excluding parameter {}'.format(param), file=sys.stderr)
        if param in source_dict:
            dest_dict[param] = source_dict[param]
        else:
            raise KeyError('cannot find parameter {} in the checkpoint'.format(param))


def main(opts):
    exclude_params = []
    # cuda device and random seed
    if opts.use_cuda is not None:
        print("using CUDA device {}".format(opts.use_cuda), file=sys.stderr)
        cuda_dev = int(opts.use_cuda)
    else:
        cuda_dev = None
    torch.manual_seed(opts.seed)
    print("random seed {}\n".format(opts.seed))
    sys.stdout.flush()

    # all_input_files = glob.glob(opts.data_dir + '/' + '*.mid.npy')
    # y2idx, idx2y = {}, {}
    # for name in all_input_files:
    #     name = name.split('/')[-1].split('.')[0]
    #     if name not in y2idx:
    #         idx = len(y2idx)
    #         y2idx[name] = idx
    #         idx2y[idx] = name
    # y_idxs = idx2y.keys()

    with open('data/labels.csv') as fp:
        idx2y = {int(x) : y for (x, y) in [line.strip().split(',') for line in fp.readlines()]}
        y2idx = {y : x for (x, y) in idx2y.items()}

    # load input file
    xs, ys = make_batches(opts.input, opts.window, opts.window)
    # ys = [y2idx[n] for n in ys]
    xs = torch.from_numpy(xs).type(torch.FloatTensor)

    # set up the model
    model = convnet.ConvNet(len(y2idx), opts.batch_size, use_cuda=opts.use_cuda, window_size=opts.window)
    if opts.load:
        print('loading parameters from {}'.format(opts.load), file=sys.stderr)
        load_model(model.state_dict(), torch.load(opts.load), exclude=exclude_params)
    if cuda_dev is not None:
        model = model.cuda(cuda_dev)

    # set up the loss function
    lf = nn.CrossEntropyLoss(reduce=False)

    train(model, xs, ys, lf, opts.batch_size, opts.max_epochs, cuda_dev=cuda_dev)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input")
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--window", type=int, default=1000)
    parser.add_argument("--max_epochs", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--load", default="recog_model/model_best.pt")
    parser.add_argument("-s", "--seed", type=int, default=0)
    parser.add_argument("-c", "--use_cuda", type=int, default=None)
    parser.add_argument("-l", "--init_lr", type=int, default=5)

    args = parser.parse_args()
    main(args)