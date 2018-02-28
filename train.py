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


def train(model, phase, dataloader, batch_size, loss_fn, optim, num_epochs=50, num_batches_val=1,
          model_dir="model", beam_size=10, cuda_dev=None, learning_rate_init=1e-5, lr_decay=None,
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
            running_loss, err = 0., 0

            # dataloader should provide whatever batch size was specified when instantiated
            for i, data in enumerate(dataloader[phase]):
                x, y = data
                y = Variable(torch.LongTensor(y))
                if cuda_dev is not None:
                    y = y.cuda(cuda_dev)
                optim.zero_grad()

                # run and calc loss
                z = model(x)  # .view(1, -1)
                loss = loss_fn(z, y)
                running_loss += loss.data[0]

                # update model
                if phase == "train":
                    loss.backward()
                    optim.step()

                # make best prediction and find err
                _, y_hat = torch.max(z, 1)
                err += (y_hat.data[0] != y.data[0])

            # print progress
            avg_loss = running_loss / float(i + 1)
            print("{} err: {:.0%}\t{} loss: {:.3}".format(
                phase, err / float(i + 1), phase, avg_loss)
            )

            # save model if best so far, or every 100 epochs
            if phase == "val":
                if avg_loss < best_loss or epoch % 99 == 0:
                    if avg_loss < best_loss:
                        best_loss = avg_loss
                        save_name = "model_best".format(epoch + 1, avg_loss)
                    else:
                        save_name = "model_epoch_{}_loss_{}".format(epoch + 1, avg_loss)
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
    enc = encoder.Encoder(datasets["train"].get_y_count(), opts.batch_size, rnn_size=128, use_cuda=cuda_dev, max_w=opts.max_w)
    if opts.load:
        enc.load_state_dict(torch.load(opts.load))
    if cuda_dev is not None:
        enc = enc.cuda(cuda_dev)

    # set up the loss function and optimizer
    class_probs = torch.FloatTensor(max(datasets["train"].x_counts.keys()) + 1)
    for idx, count in datasets["train"].x_counts.items():
        class_probs[idx] = count
    class_probs /= sum(class_probs)
    lf = nn.CrossEntropyLoss(weight=torch.FloatTensor(
        torch.FloatTensor([1. for x in class_probs]) - class_probs).cuda(cuda_dev)
    )
    optim = torch.optim.SGD(enc.parameters(), lr=10**(-opts.init_lr), momentum=0.9)

    # train(model, phase, dataloader, batch_size, loss_fn, optim, num_epochs=50)
    train(enc, "train", dataloaders, 1, lf, optim, opts.max_epochs, cuda_dev=cuda_dev)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
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