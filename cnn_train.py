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

import cnn
import pr_dataset


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


def make_batch(tensors, max_w, cuda_dev=None):
    batch_size = len(tensors)
    max_w = min(max_w, min(t.shape[1] for t in tensors))

    if max_w % 16 != 0:
        max_w = 16*(max_w // 16 + 1)

    x_batch = []
    for tensor in tensors:
        # tensor = (tensor > 0.).type(torch.float)
        tensor = tensor.view(1, 1, tensor.shape[0], tensor.shape[1])
        if tensor.shape[3] > max_w:
            tensor = random_crop(tensor, max_w)
        elif tensor.shape[3] < max_w:
            tensor = torch.nn.functional.pad(tensor, (0, max_w - tensor.shape[3], 0, 0))
        assert(tensor.shape[3] == max_w)
        x_batch.append(tensor)

    if cuda_dev is None:
        x_batch = Variable(torch.cat(x_batch, 0))
    else:
        x_batch = Variable(torch.cat(x_batch, 0)).cuda(cuda_dev)

    return x_batch


def run_loss(net, dataloader, loss_fn, cuda_dev=None, max_w=1024):
    net.eval()
    total_loss = 0.

    for i, data in enumerate(dataloader):
        x, y = data
        y = Variable(torch.LongTensor(y))
        if cuda_dev is not None:
            y = y.cuda(cuda_dev)

        x = make_batch(x, max_w, cuda_dev)

        # run and calc loss
        z = net(x)
        z_mean = z.mean(dim=3).mean(dim=2)
        loss = loss_fn(z_mean, y)
        total_loss += loss.cpu().item()

    avg_loss = total_loss / len(dataloader)
    return avg_loss


def train_epoch(net, dataloader, optim, loss_fn, cuda_dev=None, max_w=1024):
    net.train()
    running_loss = 0.
    for i, data in enumerate(dataloader):
        x, y = data
        y = Variable(torch.LongTensor(y))
        if cuda_dev is not None:
            y = y.cuda(cuda_dev)

        x = make_batch(x, max_w, cuda_dev)

        optim.zero_grad()

        # run and calc loss
        z = net(x)
        z_mean = z.mean(dim=3).mean(dim=2)
        loss = loss_fn(z_mean, y)
        loss.backward()
        optim.step()
        running_loss += loss.data.item()

        if i > 0 and i % 200 == 0:
            print('iter {:3}\trunning loss: {}'.format(i + 1, running_loss / (i + 1)))
            sys.stdout.flush()

    print('iter {:3}\trunning loss: {}'.format(i + 1, running_loss / (i + 1)))


def train(
        net, dataloaders, optim, loss_fn,
        num_epochs=50, model_dir="model", cuda_dev=None, lr_decay=0., max_w=1024
):
    best_loss = float('inf')
    for epoch in range(num_epochs):
        optim.param_groups[0]['lr'] *= (1. - lr_decay)
        print("\n" + 80*"*" + "\nEpoch {}\tlr {}\n".format(epoch + 1, optim.param_groups[0]['lr']))

        train_epoch(net, dataloaders['train'], optim, loss_fn, cuda_dev, max_w)

        with torch.no_grad():
            val_loss = run_loss(net, dataloaders['val'], loss_fn, cuda_dev, max_w)
            print('val loss: {:.3}'.format(val_loss))

            if epoch % 100 == 0 and epoch > 0:
                save_name = "checkpoint-epoch{}-loss{:.3}.pt".format(epoch + 1, val_loss)
                save_path = "{}/{}".format(model_dir, save_name)
                torch.save(net.state_dict(), save_path)
                print("Model checkpoint saved to {}".format(save_path))

            if val_loss < best_loss:
                best_loss = val_loss
                save_name = "weights-epoch{}-loss{:.3}.pt".format(epoch + 1, val_loss)
                save_path = os.path.join(os.getcwd(), model_dir, save_name)
                torch.save(net.state_dict(), save_path)
                print("Model saved to {}".format(save_path))

        print(80 * '-')
        sys.stdout.flush()


def load_data(path):
    data = []
    for d in os.listdir(path):
        filenames = glob.glob(path + "/" + d + "/*.npy")
        data += [(np.load(f), d) for f in filenames]
    return data


def main(opts):
    # training script
    if opts.use_cuda is not None:
        print("using CUDA device {}".format(opts.use_cuda))
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
            batch_size=opts.batch_size if p == "train" else 8,
            shuffle=True,
            num_workers=0,
            collate_fn=lambda b : list(list(l) for l in zip(*b))
        ) for p in ("train", "val")
    }

    # set up the model
    net = cnn.FCNN(datasets["train"].get_y_count(), opts.batch_size, cuda_dev, opts.max_w)

    if opts.load:
        saved_state = torch.load(opts.load, map_location='cpu')
        net.load_state_dict(saved_state)
    if cuda_dev is not None:
        net = net.cuda(cuda_dev)

    # set up the loss function and optimizer
    class_probs = torch.FloatTensor(max(datasets["train"].x_counts.keys()) + 1)
    for idx, count in datasets["train"].x_counts.items():
        class_probs[idx] = count
    class_probs /= sum(class_probs)
    lf = nn.CrossEntropyLoss(weight=torch.FloatTensor(
        torch.FloatTensor([1. for x in class_probs]) - class_probs).cuda(cuda_dev)
    )
    optim = torch.optim.SGD(net.parameters(), float(opts.init_lr), momentum=0.9)

    os.makedirs(opts.model_dir, exist_ok=True)

    sys.stdout.flush()
    # net, dataloaders, optim, loss_fn, num_epochs=50, model_dir="model", cuda_dev=None, lr_decay=0., max_w=1024
    train(net, dataloaders, optim, lf, opts.max_epochs, model_dir=opts.model_dir, cuda_dev=cuda_dev, lr_decay=0.00025, max_w=opts.max_w)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="preprocessed")
    parser.add_argument("--max_epochs", type=int, default=12500)
    parser.add_argument("--max_w", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("-m", "--model_dir", default="junk_weights")
    parser.add_argument("-s", "--seed", type=int, default=0)
    parser.add_argument("-c", "--use_cuda", type=int, default=None)
    parser.add_argument("-l", "--init_lr", default="10e-5")
    parser.add_argument("--load", default=None)

    args = parser.parse_args()
    main(args)