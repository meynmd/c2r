import argparse
import glob
import os
import sys

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from torch.autograd import Variable

import composer_dataset
import v2.cnn as autoencoder
import focal_loss

# train a model for one category

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


def make_batch(tensors, max_w, cuda_dev, left_pad=512):
    batch_size = len(tensors)
    # max_w = min(max_w, min(t.shape[1] for t in tensors))

    x_batch = []
    for tensor in tensors:
        h, w = tensor.shape
        padding = left_pad if w < max_w else max_w - w
        tensor = (tensor > 0.).type(torch.float)                # make sure input is binary but float type
        tensor = F.pad(tensor, (padding, 0), 'constant', 0.)    # zero padding for initial time frames

        h, w = tensor.shape
        tensor = tensor.view(1, 1, h, w)
        if tensor.shape[3] > max_w:
            tensor = random_crop(tensor, max_w)
        # elif tensor.shape[3] < max_w:
        #     tensor = torch.nn.functional.pad(tensor, (0, max_w - tensor.shape[3], 0, 0))
        assert(tensor.shape[3] == max_w)
        x_batch.append(tensor)

    x_batch = Variable(torch.cat(x_batch, 0))

    if cuda_dev is not None:
        x_batch = x_batch.cuda(cuda_dev)

    return x_batch


def train(
        net, dataloaders, optim, loss_fn,
        num_epochs=50, model_dir="model", cuda_dev=None, lr_decay=0., max_w=1024, left_pad=0
):
    for epoch in range(num_epochs):
        optim.param_groups[0]['lr'] *= (1. - lr_decay)
        print("\n" + 80*"-" + "\nEpoch {}\tlr {}\n".format(epoch + 1, optim.param_groups[0]['lr']))

        running_loss = train_epoch(net, dataloaders['train'], optim, loss_fn, cuda_dev, max_w, left_pad)
        with torch.no_grad():
            val_loss, bce_loss = run_loss(net, dataloaders['val'], loss_fn, cuda_dev, max_w)

            print('run loss: {:.3}\nval loss: {:.3}, {:.3}'.format(running_loss, val_loss, bce_loss))

        if (epoch + 1) % 50 == 0 and epoch > 0:
            save_name = "checkpoint-epoch{}-loss{:.3}.pt".format(epoch + 1, bce_loss)
            save_path = "{}/{}".format(model_dir, save_name)
            torch.save(net.state_dict(), save_path)
            print("Model checkpoint saved to {}".format(save_path))

        # elif avg_loss < best_loss:
        #     best_loss = avg_loss
        #     save_name = "model-loss{:.3}-epoch{}.pt".format(avg_loss, epoch + 1)
        #     save_path = os.path.join(os.getcwd(), model_dir, save_name)
        #     torch.save(net.state_dict(), save_path)
        #     print("Model saved to {}".format(save_path))

        print(80 * '-')
        sys.stdout.flush()


def run_loss(net, dataloader, loss_fn, cuda_dev=None, max_w=1024):
    net.eval()
    total_loss = 0.
    bce_loss = 0.
    for i, data in enumerate(dataloader):
        x, _ = data
        batch_size = len(x)
        x = make_batch(x, max_w, cuda_dev, left_pad=max_w - 1)

        # num_ones = torch.sum(x, -1)
        # num_ones = torch.sum(num_ones, 0).squeeze(0)
        # num_ones = num_ones + torch.ones(num_ones.shape).cuda(cuda_dev) * 1e-3
        # positive_weight = (torch.ones(num_ones.shape).cuda(cuda_dev) * (x.shape[0] * x.shape[-1]) - num_ones) / num_ones

        target = x.clone()
        yh = net(x)

        # w = torch.ones(target.shape).cuda(cuda_dev)
        # for row in range(w.shape[-2]):
        #     w[:, :, row, :] *= positive_weight[row]

        # loss = F.binary_cross_entropy_with_logits(yh, target, pos_weight=w) #, reduction='none')
        bce_loss += F.binary_cross_entropy_with_logits(yh, target)
        loss = loss_fn(yh, target)
        total_loss += loss.cpu().item()

    avg_loss = total_loss / len(dataloader)

    return avg_loss, bce_loss / len(dataloader)


def train_epoch(net, dataloader, optim, loss_fn, cuda_dev=None, max_w=1024, left_pad=0):
        net.train()
        running_loss, err = 0., 0.
        for i, data in enumerate(dataloader):
            x, _ = data
            batch_size = len(x)
            x = make_batch(x, max_w, cuda_dev, left_pad=max_w - 1)

            # num_ones = torch.sum(x, -1)
            # num_ones = torch.sum(num_ones, 0).squeeze(0)
            # num_ones = num_ones + torch.ones(num_ones.shape).cuda(cuda_dev) * 1e-3

            # positive_weight = (torch.ones(num_ones.shape).cuda(cuda_dev)*(x.shape[0] * x.shape[-1]) - num_ones) / num_ones

            target = x.clone()
            yh = net(x)

            # w = None
            # w = torch.ones(target.shape).cuda(cuda_dev)
            # for row in range(w.shape[-2]):
            #     w[:, :, row, :] *= positive_weight[row]

            optim.zero_grad()
            # loss = F.binary_cross_entropy_with_logits(yh, target, pos_weight=w) #, reduction='none')
            loss = loss_fn(yh, target)
            running_loss += loss.item()
            loss.backward()
            optim.step()

            # if i % 200 == 0:
            #     print('iter {:3}\ttrain loss: {:.3}'.format(i+1, loss))
            #     sys.stdout.flush()

        return running_loss / len(dataloader)


def main(opts):
    # training script
    if opts.use_cuda is not None:
        cuda_dev = int(opts.use_cuda)
    else:
        cuda_dev = None
    torch.manual_seed(opts.seed)
    lr_decay = float(opts.lr_decay)

    # initialize data loader
    datasets = { p : composer_dataset.ComposerDataset(
                 os.getcwd() + "/" + opts.data_dir, "labels.csv", p, classname=opts.classname)
                 for p in ("train", "val") }

    classname = opts.classname
    # train_datapoints, val_datapoints = (list(datasets[phase].get_from_class(classname)) for phase in ["train", "val"])

    dataloaders = {
        p : DataLoader(
            datasets[p],
            batch_size=opts.batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=lambda b : list(list(l) for l in zip(*b))
        ) for p in ("train", "val")
    }

    # initialize network
    # in_channels, h_channels, discrete_channels
    # net = pixel_cnn.PixelCNN(1, 32, 64)
    net = autoencoder.AutoEncoder(opts.batch_size, cuda_dev, opts.max_w)

    if opts.load:
        saved_state = torch.load(opts.load, map_location='cpu')
        net.load_state_dict(saved_state)
    if cuda_dev is not None:
        net = net.cuda(cuda_dev)

    optim = torch.optim.SGD(net.parameters(), float(opts.init_lr), momentum=0.9)

    os.makedirs(opts.model_dir, exist_ok=True)
    sys.stdout.flush()

    if opts.left_pad is None:
        opts.left_pad = opts.max_w - 1

    # use neg/pos as alpha for focal loss
    num_ones, num_elem = 0., 0.
    for x, _ in dataloaders['train']:
        for sample in x:
            num_ones += torch.sum(sample).item()
            num_elem += torch.numel(sample)
    zero_one = (num_elem - num_ones) / num_ones
    frac_ones = num_ones / num_elem

    loss_function = focal_loss.FocalLoss(gamma=2.)

    train(
        net, dataloaders, optim, loss_function,
        opts.max_epochs, opts.model_dir, cuda_dev,
        max_w=opts.max_w, left_pad=opts.left_pad,
        lr_decay=lr_decay
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="../../clean_preproc")
    parser.add_argument("--max_epochs", type=int, default=1000)
    parser.add_argument("--max_w", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("-m", "--model_dir", default="model")
    parser.add_argument("-s", "--seed", type=int, default=0)
    parser.add_argument("-c", "--use_cuda", type=int, default=None)
    parser.add_argument("-l", "--init_lr", default="5e-6")
    parser.add_argument("--load", default=None)
    parser.add_argument("--left_pad", default=None)
    parser.add_argument("--classname", required=True)
    parser.add_argument("--lr_decay", default="0.025")
    args = parser.parse_args()

    print("training on category {}".format(args.classname))
    print(args.__dict__)
    sys.stdout.flush()

    main(args)

