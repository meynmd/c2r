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

    target_class = opts.targetclass
    target_name = datasets["train"].idx2name[target_class]
    class_datapoints = list(datasets["train"].get_from_class(target_name))
    x, label = class_datapoints[0]


    x_batch = torch.FloatTensor(1, 1, x.shape[0], x.shape[1])
    x_batch[0,0,:,:] = x
    x_batch = Variable(x_batch, requires_grad=True)
    if opts.use_cuda is not None:
        x_batch = x_batch.cuda(opts.use_cuda)

    optim = torch.optim.SGD([x_batch], lr=10**(-opts.init_lr))

    loss_fn = torch.nn.CrossEntropyLoss()

    print("target class: {}".format(target_name))
    enc.zero_grad()

    z = enc(x_batch)
    y = torch.LongTensor(1)
    y[0] = label
    y = Variable(y)
    loss = loss_fn(z, y)
    target_vec = torch.zeros(z.data.shape[-1])
    loss.backward()

    np.save("x_{}".format(target_name), x_batch.cpu().data.view(128, -1).numpy())
    np.save("grad_{}".format(target_name), x_batch.cpu().grad.data.view(128, -1).numpy())

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