import argparse
import glob
import os
import sys, math
import random
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from torch.autograd import Variable
from torch import cuda

import encoder
import pr_dataset

# make batch_size overlapping windows
def make_cropped_batch(tensor, batch_size, window_size, stride):
    h, w = tensor.shape[0], tensor.shape[1]
    windows, start, stop = [], 0, window_size
    while stop < w:
        windows.append(tensor[:, start : stop].unsqueeze(0))
        start += stride
        stop += stride
    batches = []
    if len(windows) > batch_size:
        num_batches = int(math.ceil(len(windows) / batch_size))
        for i in range(num_batches):
            batch = windows[i : min((i+1)*batch_size, len(windows))]
            batches.append(torch.stack(batch))
    else:
        batches = [torch.stack(windows)]

    return batches



    windows = torch.zeros(num_windows, 1, tensor.shape[0], window_size)
    stride = (w - window_size) // (num_windows - 1)
    for i in range(num_windows):
        windows[i, :, :, :] = tensor[:, i*stride: i*stride + window_size]
    return windows, stride


def test(model, data, num_per_class, cuda_dev, max_w, batch_size=1, out_dir="bestfrag", stride=50):
    loss_fn = torch.nn.CrossEntropyLoss(reduce=False)
    correct_total, xy_count = 0, 0
    labels = data.idx2name.items()
    num_classes = len(data.get_all_labels())
    correct = np.zeros(num_classes)
    confusion = np.zeros((num_classes, num_classes))
    predict_confusion = np.zeros((num_classes, num_classes))
    for i, label in labels:
        # get (x, y)'s from dataset object, select num_per_class of them
        class_datapoints = list(data.get_from_class(label))
        # random.shuffle(class_datapoints)
        num_d = min(num_per_class, len(class_datapoints))
        probs = np.zeros(num_classes) # [0. for j in range(len(data.get_all_labels()))]
        preds = np.zeros(num_classes)
        for (x, y_idx), (x_name, y_name) in class_datapoints[:num_d]:
            if x.shape[1] <= model.max_w:
                x_batches = [x.unsqueeze(0).unsqueeze(0)]
                continue
            else:
                # tensor, batch_size, window_size, max_overlap
                x_batches = make_cropped_batch(x, batch_size, max_w, stride)

            model.eval()
            best_ce = float('inf')
            best_window = None
            for k, batch in enumerate(x_batches):
                z = model(batch)
                print(batch.shape[0])
                y = Variable(torch.LongTensor([y_idx for _ in range(batch.shape[0])]))
                if cuda_dev:
                    y = y.cuda(cuda_dev)

                # need to update this code
                crossentropy = loss_fn(z, y).data.cpu().numpy()
                batch_best_window = stride*(np.argmin(crossentropy))
                batch_best_ce = np.min(crossentropy)
                if batch_best_ce < best_ce:
                    best_ce = batch_best_ce
                    best_window = k*stride*batch_size + batch_best_window

            w_left, w_right = best_window, best_window+ max_w
            print("{}, {}: ({}, {})".format(y_name, x_name.split("/")[-1].split(".")[0], w_left, w_right))
            # out_sdir = out_dir + "/" + y_name
            # if not os.path.exists(out_sdir):
            #     os.makedirs(out_sdir)
            # window = x_batch.cpu()[best_window, :, :, :].squeeze(0).squeeze(0).numpy()
            # np.save(out_sdir + "/" + x_name.split("/")[-1].split(".")[0], window)



def main(opts):
    if opts.use_cuda is not None:
        print("using CUDA device {}".format(opts.use_cuda), file=sys.stderr)
        cuda_dev = int(opts.use_cuda)
    else:
        cuda_dev = None

    # load the data
    dataset = pr_dataset.PianoRollDataset(os.getcwd() + "/" + opts.data_dir, "labels.csv", "val")

    # load the model
    enc = encoder.Encoder(
        dataset.get_y_count(),
        opts.batch_size,
        rnn_size=opts.rnn_size,
        num_rnn_layers=opts.rnn_layers,
        use_cuda=cuda_dev,
        max_w=opts.max_w
    )

    if opts.load:
        saved_state = torch.load(opts.load, map_location='cpu')
        enc.load_state_dict(saved_state)
    if cuda_dev is not None:
        enc = enc.cuda(cuda_dev)

    test(enc, dataset, 100, opts.use_cuda, opts.max_w, opts.batch_size)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rnn_size", type=int, default=128)
    parser.add_argument("--rnn_layers", type=int, default=3)
    parser.add_argument("--data_dir", default="preprocessed")
    parser.add_argument("--max_epochs", type=int, default=1000)
    parser.add_argument("--max_w", type=int, default=5000)
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("-m", "--model_dir", default="model")
    parser.add_argument("-e", "--num_epochs", type=int, default=5)
    parser.add_argument("--num_batch_valid", type=int, default=1)
    parser.add_argument("--beam_size", type=int, default=5)
    parser.add_argument("-s", "--seed", type=int, default=0)
    parser.add_argument("-c", "--use_cuda", type=int, default=None)
    parser.add_argument("-l", "--init_lr", type=int, default=5)
    parser.add_argument("--load", default=None)

    args = parser.parse_args()
    main(args)