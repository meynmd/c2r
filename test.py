import argparse
import glob
import os
import sys
import random
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from torch.autograd import Variable
from torch import cuda

import encoder
import pr_dataset


def test(model, data, num_per_class, cuda_dev):
    correct_total, xy_count = 0, 0
    labels = data.idx2name.items()
    num_classes = len(data.get_all_labels())
    correct = np.zeros(num_classes)
    confusion = np.zeros((num_classes, num_classes))
    predict_confusion = np.zeros((num_classes, num_classes))
    for i, label in labels:
        # get (x, y)'s from dataset object, select num_per_class of them
        class_datapoints = list(data.get_from_class(label))
        random.shuffle(class_datapoints)
        num_d = min(num_per_class, len(class_datapoints))
        probs = np.zeros(num_classes) # [0. for j in range(len(data.get_all_labels()))]
        preds = np.zeros(num_classes)
        for x, y in class_datapoints[:num_d]:
            model.eval()
            z = model([x])
            # record softmax "probabilities" for each label
            p = torch.nn.functional.softmax(z).cpu().data.numpy()
            probs = probs + p
            # make prediction
            _, y_hat = torch.max(z, 1)
            preds[y_hat.data[0]] += 1
            c = int(y_hat.data[0] == y)
            correct[i] += c
            correct_total += c
            xy_count += 1
        # column i shows probabilities that composer i is classified as...
        confusion[:, i] = probs / num_d
        correct[i] /= num_d
        predict_confusion[:, i] = preds / num_d

    print("*"*10, "Probabilities", "*"*10, "\n")
    print(" " * 11, end="")
    short_label = []
    for _, label in labels:
        l = min(8, len(label))
        short_label.append("{:>8}".format(label[:l]))
    for sl in short_label:
        print("{}".format(sl), end="   ")
    print()
    for j, sl in enumerate(short_label):
        print(sl, end="   ")
        for i in range(confusion.shape[1]):
            print("{:8.1%}".format(confusion[j][i]), end="   ")
        print()

    print("\n" + "*"*10, "Predictions", "*"*10, "\n")
    print(" " * 11, end="")
    short_label = []
    for _, label in labels:
        l = min(8, len(label))
        short_label.append("{:>8}".format(label[:l]))
    for sl in short_label:
        print("{}".format(sl), end="   ")
    print()
    for j, sl in enumerate(short_label):
        print(sl, end="   ")
        for i in range(predict_confusion.shape[1]):
            print("{:8.1%}".format(predict_confusion[j][i]), end="   ")
        print()
    print()
    for sl in short_label:
        print(sl, end="   ")
    print()
    for i in range(correct.shape[0]):
        print("{:8.2%}".format(correct[i]), end="   ")

    print("\n\nAverage accuracy: {:.3%}\n".format(float(correct_total) / xy_count))


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
        enc.load_state_dict(torch.load(opts.load))
    if cuda_dev is not None:
        enc = enc.cuda(cuda_dev)

    test(enc, dataset, 100, opts.use_cuda)

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