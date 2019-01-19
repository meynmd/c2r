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

import regen_encoder as encoder
import pr_dataset


def load_data(path):
    data = []
    for d in os.listdir(path):
        filenames = glob.glob(path + "/" + d + "/*.npy")
        data += [(np.load(f), d) for f in filenames]
    return data


def main(opts):
    # training script
    if opts.use_cuda is not None:
        print("using CUDA device {}".format(opts.use_cuda), file=sys.stderr)
        cuda_dev = int(opts.use_cuda)
    else:
        cuda_dev = None
    torch.manual_seed(opts.seed)
    print("random seed {}".format(opts.seed))
    sys.stdout.flush()

    # initialize data loader
    datasets = { p : pr_dataset.PianoRollDataset(os.getcwd() + "/" + opts.data_dir, "labels.csv", p)
                 for p in ("train", "val") }
    dataloaders = {
        p : DataLoader(
            datasets[p],
            batch_size=opts.batch_size if p == "train" else 1,
            shuffle=True,
            num_workers=2,
            collate_fn=lambda b : list(list(l) for l in zip(*b))
        ) for p in ("train", "val")
    }

    # set up the model
    enc = encoder.Encoder(
        datasets["train"].get_y_count(),
        opts.batch_size,
        rnn_size=opts.rnn_size,
        num_rnn_layers=opts.rnn_layers,
        use_cuda=cuda_dev,
        max_w=opts.max_w
    )

    if opts.load:
        enc.load_state_dict(torch.load(opts.load))
    else:
        print("error: need --load <file> option", file=sys.stderr)
        exit(1)

    if cuda_dev is not None:
        enc = enc.cuda(cuda_dev)

    enc.eval()

    labels = datasets["train"].idx2name.items()
    # opt_img = Variable(torch.rand(1, 1, 128, 5000), requires_grad=True)
    opt_img = np.random.normal(0, 0.01, (128, 5000))
    opt_img = Variable(torch.max(torch.FloatTensor(1, 1, 128, 5000)*1e-20, torch.zeros(1, 1, 128, 5000)), requires_grad=True)
    if cuda_dev is not None:
        opt_img = opt_img.cuda((cuda_dev))
    loss_fn = torch.nn.CrossEntropyLoss()

    optim = torch.optim.SGD([opt_img], lr=10**(-opts.init_lr))

    target_class = opts.targetclass
    target_name = datasets["train"].idx2name[target_class]
    print("target class: {}".format(target_name))

    best_loss, max_patience, patience, lastsave = float("inf"), 5, 5, 0
    for i in range(100000):
        optim.zero_grad()
        output = enc(opt_img)

        y = torch.LongTensor(1)
        y[0] = target_class
        y = Variable(y)
        if opts.use_cuda:
            y = y.cuda(opts.use_cuda)
        class_loss = loss_fn(output, y)
        if opts.use_cuda:
            l2_loss = 1e-3 * (opt_img.cuda(opts.use_cuda)**2).mean()
        else:
            l2_loss = 1e-3 * (opt_img ** 2).mean()
        loss = class_loss + l2_loss
        if i % 100 == 0:
            print("class loss: {:.04f}\treg loss: {:.04f}\ttotal loss:{:.04f}".format(class_loss.data[0], l2_loss.data[0], loss.data[0]))
        loss.backward()
        optim.step()
        if loss.data[0] < best_loss:
            best_loss = loss.data[0]
            patience = max_patience
            if i - lastsave > 10:
                np.save("generated_{}".format(target_name), opt_img.cpu().data.numpy())
                lastsave = i
        if patience == 0:
            break

    np.save("generated_{}".format(target_name), opt_img.cpu().data.numpy())

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
    parser.add_argument("--num_batch_valid", type=int, default=1)
    parser.add_argument("--beam_size", type=int, default=5)
    parser.add_argument("-s", "--seed", type=int, default=0)
    parser.add_argument("-c", "--use_cuda", type=int, default=None)
    parser.add_argument("-l", "--init_lr", type=int, default=2)
    parser.add_argument("--load", default=None)
    parser.add_argument("--targetclass", type=int, default=0)

    args = parser.parse_args()
    main(args)