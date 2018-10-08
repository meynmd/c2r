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

import pr_dataset
import vae

def random_crop(tensor, w_new, h_new=None):
    h, w = tensor.shape[2], tensor.shape[3]
    top, left = 0, 0
    if w_new < w:
        left = np.random.randint(0, w - w_new)
    if h_new is None:
        return tensor[:, :, :, left: left + w_new]
    if h_new < h:
        top = np.random.randint(0, h - h_new)
    return tensor[top: top + h_new, left: left + w_new]


def make_batch(tensors, max_w, batch_size):
    # batch_size = len(tensors)
    # max_w = min(max_w, min(t.shape[1] for t in tensors))

    assert(len(tensors) > 0)
    col_height = tensors[0].shape[0]
    dims = (col_height, max_w)
    x_batch = torch.zeros(batch_size, 1, *dims)
    for i, tensor in enumerate(tensors):
        tensor = (tensor > 0.).type(torch.FloatTensor)
        tensor = tensor.view(1, 1, tensor.shape[0], tensor.shape[1])
        if tensor.shape[3] > max_w:
            tensor = random_crop(tensor, max_w)
        elif tensor.shape[3] < max_w:
            padded = torch.zeros(*tensor.shape[:-1], max_w)
            padded[:, :, :, :tensor.shape[3]] = tensor[:, :, :, :]
            tensor = padded
        x_batch[i, :, :, :] = tensor

    return Variable(x_batch)


def train(model, phase, dataloader, batch_size, loss_function, optim, num_epochs=50, num_batches_val=1,
              model_dir="model_narrow", beam_size=10, cuda_dev=None, learning_rate_init=1e-5, lr_decay=None,
              start_decay_at=None):

    best_loss = float("inf")
    phases = [phase]
    if phase == "val":
        num_epochs = 1
    else:
        phases.append("val")

    for epoch in range(num_epochs):
        print("\nEpoch {}\n".format(epoch + 1) + 80*"*")

        # phases ["train", "val"], or ["val"]
        for phase in phases:
            running_loss, err = 0., 0.
            if phase == "train":
                model.train()
            else:
                model.eval()

            # dataloader should provide whatever batch size was specified when instantiated
            acc_z, z_min, z_max = 0., float('inf'), -float('inf')
            for i, data in enumerate(dataloader[phase]):
                x, y = data
                # y = Variable(torch.LongTensor(y))
                # if cuda_dev is not None:
                #     y = y.cuda(cuda_dev)

                x = make_batch(x, model.max_w, batch_size)
                if cuda_dev:
                    x = x.cuda(cuda_dev)

                optim.zero_grad()

                # run and calc loss
                z, mu, logvar = model(x)
                if cuda_dev is not None:
                    x = x.cuda(cuda_dev)
                    z = z.cuda(cuda_dev)

                acc_z += z.data.mean().item()
                z_min, z_max = min(z.data.min().item(), z_min), max(z.data.max().item(), z_max)

                loss, bce, kld = loss_fn(z, x, mu, logvar)
                print('bce: {}\tkld: {}'.format(bce, kld))
                running_loss += loss.data.item()

                # update model
                if phase == "train":
                    loss.backward()
                    optim.step()

            avg_z = acc_z / float(len(dataloader[phase]))

            print('avg z: {}\tmin z: {}\tmax z: {}'.format(avg_z, z_min, z_max))

            # print progress
            avg_loss = running_loss / float(i + 1)
            print("{} loss: {:.5}".format(
                phase, avg_loss)
            )

            # save model if best so far, or every 100 epochs
            if phase == "val":
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    if epoch > 99:
                        save_name = "model_epoch{}_loss{:.3}_".format(epoch + 1, avg_loss)
                        save_name += "-".join(time.asctime().split(" ")[:-1]).replace(":", ".")
                    else:
                        save_name = "model_best".format(epoch + 1, avg_loss)
                    save_name += "_rnn-size{}.pt".format(model.rnn_size)
                    save_path = "{}/{}".format(model_dir, save_name)
                    torch.save(model.state_dict(), save_path)
                    print("Model saved to {}".format(save_path))
                sys.stdout.flush()
    print()


def load_data(path):
    data = []
    for d in os.listdir(path):
        filenames = glob.glob(path + "/" + d + "/*.npy")
        data += [(np.load(f), d) for f in filenames]
    return data


def loss_fn(z, x, mu, logvar):
    bce = torch.nn.functional.binary_cross_entropy(z, x, size_average=True)
    kld = -0.5*torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    kld = 0.
    return bce + kld, bce, kld


def main(opts):
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
    enc = vae.VariationalAutoencoder(
        opts.batch_size,
        rnn_size=opts.rnn_size,
        num_rnn_layers=opts.rnn_layers,
        use_cuda=cuda_dev,
        max_w=opts.max_w,
        h_dim=opts.latent_dim
    )
    if opts.load:
        enc.load_state_dict(torch.load(opts.load))
    if cuda_dev is not None:
        enc = enc.cuda(cuda_dev)

    optim = torch.optim.SGD(enc.parameters(), lr=10**(-opts.init_lr), momentum=0.9)

    os.makedirs(os.path.join(os.getcwd(), opts.model_dir), exist_ok=True)

    train(enc, "train", dataloaders, opts.batch_size, loss_fn, optim, opts.max_epochs, cuda_dev=cuda_dev, model_dir=opts.model_dir)


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
    parser.add_argument("-l", "--init_lr", type=int, default=5)
    parser.add_argument("--load", default=None)
    parser.add_argument("--latent_dim", type=int, default=512)

    args = parser.parse_args()
    main(args)