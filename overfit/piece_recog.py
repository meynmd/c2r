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
    best_loss = 10e9
    best_minibatches, best_minibatch_idxs = [], []
    num_batches = xs.shape[0] // batch_size + 1
    predictions = np.zeros(model.num_cat)

    for batch_num in range(num_batches):
        start, end = batch_num*batch_size, min(xs.shape[0], (batch_num + 1)*batch_size)
        x_batch = xs[start : end, :, :, :]
        y_batch = Variable( torch.LongTensor(ys[start : end]) )
        if cuda_dev is not None:
            x_batch = x_batch.cuda(cuda_dev)
            y_batch = y_batch.cuda(cuda_dev)

        # run and calc loss
        z = model(x_batch)
        loss = loss_fn(z, y_batch)

        this_best_loss, _ = torch.min(loss, dim=0)
        this_best_loss = this_best_loss.data[0]

        # choose the best windows
        if this_best_loss < best_loss + 10e-4:
            best_idxs = torch.LongTensor([j for j in range(loss.data.shape[0]) if loss.data[j] - this_best_loss < 0.0001])
            this_best_minibatches = torch.index_select(x_batch.cpu(), 0, best_idxs)
            best_loss = this_best_loss
            if this_best_loss < best_loss:
                best_minibatches = [this_best_minibatches]
                best_minibatch_idxs = [best_idxs]
            else:
                best_minibatches += [this_best_minibatches]
                best_minibatch_idxs += [best_idxs]

        # make prediction
        _, y_hat = torch.max(z, 1)
        y_hat = y_hat.data[0]
        predictions[y_hat] += 1;

    best_minibatches = torch.cat(best_minibatches)
    best_minibatch_idxs = torch.cat(best_minibatch_idxs)
    if best_minibatches.shape[0] > batch_size:
        best_minibatches = best_minibatches[0 : batch_size, :, :, :]    # maybe do something better later

    # find activations on best windows
    activations = model.conv_activations(best_minibatches)
    activations = activations.cpu().data

    time_factor = best_minibatches.shape[-1] / activations.shape[-1]
    pitch_factor = best_minibatches.shape[-2] / activations.shape[-2]

    act_max_idxs = np.unravel_index(np.argmax(activations.numpy(), axis=None), activations.numpy().shape)
    left_bound, right_bound = round(time_factor*(act_max_idxs[-1] - 1)), round(time_factor*(1 + act_max_idxs[-1]))
    lower_bound, upper_bound = round(pitch_factor*(act_max_idxs[-2] - 1)), round(pitch_factor*(1 + act_max_idxs[-2]))

    x_batch = best_minibatches[act_max_idxs[0], :, :, :]
    x_batch = x_batch.squeeze(0).squeeze(0)
    print("l:u:l:r\n{}:{}:{}:{}".format(lower_bound, upper_bound, left_bound+1000*best_minibatch_idxs[act_max_idxs[0]], right_bound+1000*best_minibatch_idxs[act_max_idxs[0]]))
    print('{}'.format(x_batch.shape))
    pr_activation = x_batch[int(lower_bound) : int(upper_bound), int(left_bound) : int(right_bound)]
    np.save('highest_act', pr_activation.numpy())
    print('saved highest_act.npy')

    pr = np.zeros((xs.shape[-2], xs.shape[0] * xs.shape[-1]))
    heatmaps = []
    for i, idx in enumerate(best_minibatch_idxs):
        batch = best_minibatches[i, :, :, :]
        activation = activations[i, :, :, :]
        flattened_act = torch.max(activation, dim=0)[0].squeeze(0).numpy()
        flattened_act = flattened_act - flattened_act.mean()
        flattened_act = flattened_act - np.min(flattened_act)
        # flattened_act = torch.max(torch.zeros(flattened_act.shape), flattened_act).numpy()
        zoomed_act = ndimage.zoom(flattened_act, (pitch_factor, time_factor))
        heatmap = zoomed_act * batch.squeeze(0).squeeze(0).numpy()
        left, right = idx * heatmap.shape[1], (idx + 1) * heatmap.shape[1]
        pr[:, left : right] = np.maximum(pr[:, left : right], heatmap[:, :])
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
    xs, ys = make_batches(opts.input, opts.window, opts.window // 2)
    ys = [y2idx[n] for n in ys]
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