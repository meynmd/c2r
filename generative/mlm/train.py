'''
training script for generative model
trains generative model on whole dataset
'''
import argparse
import os
import sys
from itertools import takewhile

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from torch.autograd import Variable

import pr_dataset

# # import baseline.rnn_baseline as model
# # import v5.crnn_v5_1 as model
# import v5.cnn as model

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
        assert(tensor.shape[3] == max_w)
        x_batch.append(tensor)

    x_batch = Variable(torch.cat(x_batch, 0))
    if cuda_dev is not None:
        x_batch = x_batch.cuda(cuda_dev)

    return x_batch


def train(
        net, dataloaders, optim, loss_fn,
        start_epoch=0, num_epochs=50, model_dir="model", cuda_dev=None, lr_decay=0., max_w=1024, left_pad=0
):
    if start_epoch is None:
        start_epoch = 0
    else:
        print('resuming from epoch {}'.format(start_epoch), file=sys.stderr)
        sys.stderr.flush()

    best_loss = float('inf')
    for epoch in range(start_epoch, num_epochs):
        optim.param_groups[0]['lr'] *= (1. - lr_decay)
        print("\n" + 80*"-" + "\nEpoch {}\tlr {}\n".format(epoch + 1, optim.param_groups[0]['lr']))

        running_loss = train_epoch(net, dataloaders['train'], optim, loss_fn, cuda_dev, max_w, left_pad)
        with torch.no_grad():
            val_loss, bce_loss = find_loss(net, dataloaders['val'], loss_fn, cuda_dev, max_w)
            print('run loss: {:.3}\nval loss: {:.3}\nbce val loss: {:.3}'.format(running_loss, val_loss, bce_loss))

            if (epoch + 1) % 50 == 0 and epoch > 0:
                save_name = "checkpoint-epoch{:04}-loss{:.3}.pt".format(epoch + 1, bce_loss)
                save_path = "{}/{}".format(model_dir, save_name)
                torch.save(net.state_dict(), save_path)
                print("Model checkpoint saved to {}".format(save_path))

            if bce_loss < best_loss:
                best_loss = bce_loss
                if epoch > 50:
                    save_name = "best-epoch{:04}-loss{:.3}.pt".format(epoch + 1, bce_loss)
                    save_path = os.path.join(os.getcwd(), model_dir, save_name)
                    torch.save(net.state_dict(), save_path)
                    print("Model saved to {}".format(save_path))

        print(80 * '-')
        sys.stdout.flush()


def find_loss(net, dataloader, loss_fn, cuda_dev=None, max_w=1024):
    net.eval()
    total_loss, total_bce_loss = 0., 0.
    for i, data in enumerate(dataloader):
        x, _ = data
        x = make_batch(x, max_w, cuda_dev, left_pad=max_w - 1)
        target = x.clone()
        yh = net(x)

        bce_loss = F.binary_cross_entropy_with_logits(yh.squeeze(1).transpose(1, 2), target.squeeze(1).transpose(1, 2))
        total_bce_loss += bce_loss.cpu().item()

        loss = loss_fn(yh.squeeze(1).transpose(1, 2), target.squeeze(1).transpose(1, 2))
        total_loss += loss.cpu().item()

    avg_loss, avg_bce_loss = total_loss / len(dataloader), total_bce_loss / len(dataloader)
    return avg_loss, avg_bce_loss


def train_epoch(net, dataloader, optim, loss_fn, cuda_dev=None, max_w=1024, left_pad=0):
        net.train()
        running_loss, err = 0., 0.

        for i, data in enumerate(dataloader):
            x, _ = data
            x = make_batch(x, max_w, cuda_dev, left_pad=max_w - 1)
            target = x.clone()
            z = net(x)

            optim.zero_grad()
            # reshape output and target for BCE loss fn
            loss = loss_fn(z.squeeze(1).transpose(1, 2), target.squeeze(1).transpose(1, 2))
            running_loss += loss.item()
            loss.backward()
            optim.step()

        return running_loss / len(dataloader)


def extract_first_number(s):
    if s == '':
        return None
    pos = 0
    while not s[pos].isdigit():
        pos += 1
    chars = takewhile(lambda x : x.isdigit(), s[pos:])
    return int(''.join(list(chars)))


def find_neg_pos_ratio(dataloader):
    # batch is a list of 2d tensors
    num_ones, num_elem = 0, 0
    for (_, (batch, _)) in enumerate(dataloader):
        num_ones += sum([torch.sum(x) for x in batch])
        num_elem += sum([torch.numel(x) for x in batch])
    return (num_elem - num_ones)/num_elem


def main(opts):
    # training script
    if opts.use_cuda is not None:
        cuda_dev = int(opts.use_cuda)
    else:
        cuda_dev = None
    torch.manual_seed(opts.seed)
    lr_decay = float(opts.lr_decay)
    start_epoch = None

    # initialize data loaders
    datasets = { p : pr_dataset.PianoRollDataset(os.getcwd() + "/" + opts.data_dir, "labels.csv", p)
                 for p in ("train", "val") }
    dataloaders = {
        p : DataLoader(
            datasets[p],
            batch_size=opts.batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=lambda b : list(list(l) for l in zip(*b))
        ) for p in ("train", "val")
    }

    print('\ndataset sizes:\t{}\t{}\n'.format(*[(p, len(d)) for (p, d) in dataloaders.items()]))

    # # rnn
    # # net = model.LanguageModeler(rnn_size=opts.rnn_size, rnn_layers=1, batch_size=opts.batch_size)
    #
    # # crnn
    # # net = model.LanguageModeler(batch_size=opts.batch_size, rnn_size=opts.rnn_size, rnn_layers=1, use_cuda=opts.use_cuda, max_w=opts.max_w)
    #
    # # cnn
    # net = model.LanguageModeler(batch_size=opts.batch_size, use_cuda=opts.use_cuda, max_w=opts.max_w)

    if opts.arch == 'crnn':
        import v5.crnn_v5_1 as crnn
        net = crnn.LanguageModeler(batch_size=opts.batch_size, rnn_size=opts.rnn_size, rnn_layers=1, use_cuda=opts.use_cuda, max_w=opts.max_w)

    elif opts.arch == 'baseline':
        import baseline.rnn_baseline as base
        net = base.LanguageModeler(rnn_size=opts.rnn_size, rnn_layers=1, batch_size=opts.batch_size, use_cuda=opts.use_cuda)

    elif opts.arch == 'cnn':
        import v5.cnn as cnn
        net = cnn.LanguageModeler(batch_size=opts.batch_size, use_cuda=opts.use_cuda, max_w=opts.max_w)

    if opts.load:
        saved_state = torch.load(opts.load, map_location='cpu')
        net.load_state_dict(saved_state)
        epoch_string = opts.load.split('epoch')[-1]
        start_epoch = extract_first_number(epoch_string)

    if cuda_dev is not None:
        net = net.cuda(cuda_dev)

    os.makedirs(opts.weights_dir, exist_ok=True)
    sys.stdout.flush()

    optim = torch.optim.SGD(net.parameters(), float(opts.init_lr), momentum=0.9)

    left_pad = opts.left_pad
    fl_gamma = opts.focal_gamma

    if opts.pos_w is None:
        print('Focal loss, gamma={}'.format(fl_gamma), file=sys.stderr)
        loss_function = focal_loss.FocalLoss(gamma=fl_gamma)
    else:
        try:
            npr = float(opts.pos_w)
        except:
            try:
                with open(opts.pos_w, 'r') as fp:
                    for line in fp.readlines():
                        line = line.strip().split(',')
                        if line[0] == 'all':
                            npr = float(line[1])
            except:
                print('error: cannot interpret --pos_w value: {}'.format(opts.pos_w), file=sys.stderr)
                exit(1)

        if npr == 0.:
            pw = None
        else:
            pw = torch.ones(88)*npr

        print('BCE loss, positive weight:', file=sys.stderr)
        print(pw, file=sys.stderr)

        if cuda_dev is not None and pw is not None:
            pw = pw.cuda(cuda_dev)

        loss_function = nn.BCEWithLogitsLoss(pos_weight=pw)

    sys.stderr.flush()

    train(
        net, dataloaders, optim, loss_function,
        start_epoch=start_epoch, num_epochs=opts.max_epochs, model_dir=opts.weights_dir, cuda_dev=cuda_dev,
        max_w=opts.max_w, left_pad=left_pad, lr_decay=lr_decay
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/fugue_preproc")
    parser.add_argument("--max_epochs", type=int, default=5000)
    parser.add_argument("--max_w", type=int, default=4096)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("-w", "--weights_dir", default="junk")
    parser.add_argument("-s", "--seed", type=int, default=0)
    parser.add_argument("-c", "--use_cuda", type=int, default=None)
    parser.add_argument("-l", "--init_lr", default="5e-6")
    parser.add_argument("--load", default=None)
    parser.add_argument("--left_pad", type=int, default=512)
    parser.add_argument("--lr_decay", default="0.0075")
    parser.add_argument("--ch_down", type=int, default=64)
    parser.add_argument("--ch_up", type=int, default=64)
    parser.add_argument("--rnn_size", type=int, default=2048)
    parser.add_argument("--pos_w", default=None)
    parser.add_argument("--focal_gamma", type=float, default=2.)
    parser.add_argument("--arch", default="baseline")

    args = parser.parse_args()

    print(args.__dict__)
    sys.stdout.flush()

    main(args)

