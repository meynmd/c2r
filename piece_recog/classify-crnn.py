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

import crnn_classifier
from fragments_dataset import FragmentsDataset


def train(model, phase, dataloader, batch_size, loss_fn, optim, num_epochs=50, model_dir="model", cuda_dev=None,):
    best_loss = float("inf")
    phases = [phase]
    if phase == "val":
        num_epochs = 1
    else:
        phases.append("val")

    for epoch in range(num_epochs):

        for j in range(len(optim.param_groups)):
            optim.param_groups[j]['lr'] += (optim.param_groups[0]['lr'] - optim.param_groups[j]['lr']) / 100.

        print('\n' + 80*"*" + "\nEpoch {}".format(epoch + 1))
        for j in range(len(optim.param_groups)):
            print("parameter group {}\tlearning rate: {}".format(j, optim.param_groups[j]['lr']))

        # phases ["train", "val"], or ["val"]
        for phase in phases:
            running_loss, err = 0., 0.
            if phase == "train":
                model.train()
            else:
                model.eval()

            for i, data in enumerate(dataloader[phase]):
                x, y = data
                batch_size = x.shape[0]
                x = x.unsqueeze(1)
                y = Variable(torch.LongTensor(y))
                if cuda_dev is not None:
                    x = x.cuda(cuda_dev)
                    y = y.cuda(cuda_dev)

                optim.zero_grad()

                # run and calc loss
                z = model(x)
                loss = loss_fn(z, y)
                running_loss += loss.data.item()

                # update model
                if phase == "train":
                    loss.backward()
                    optim.step()
                else:
                    # make best prediction and find err
                    _, y_hat = torch.max(z, 1)
                    err += (y_hat.data != y.data).sum().item() / float(x.shape[0])

            # print progress
            avg_loss = running_loss / float(i + 1)
            print("{} loss: {:.5}".format(phase, avg_loss))
            if phase != 'train':
                print("{} err: {:.0%}".format(phase, float(err) / float(i + 1)))
            sys.stdout.flush()

            # save model if best so far, or every 100 epochs
            if phase == "val":
                if avg_loss < best_loss:
                    if epoch > 99:
                        save_name = "model-loss{:.3}-epoch{}".format(avg_loss, epoch + 1)
                        # save_name += "-".join(time.asctime().split(" ")[:-1]).replace(":", ".")
                    else:
                        save_name = "model-best"
                    # save_name += ".pt"
                    save_path = "{}/{}".format(model_dir, save_name)
                    torch.save(model.state_dict(), save_path)
                    print("Model saved to {}".format(save_path))

                elif (epoch + 1) % 100 == 0 and epoch > 0:
                    save_name = "model-epoch{}-loss{:.3}".format(epoch + 1, avg_loss)
                    save_path = "{}/{}".format(model_dir, save_name)
                    torch.save(model.state_dict(), save_path)
                    print("Model saved to {}".format(save_path))

                sys.stdout.flush()
                best_loss = min(best_loss, avg_loss)

        print(80*"*")
    print()


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
    labels_path = os.path.join(os.getcwd(), opts.data_dir, 'labels.csv')
    # data_root, label_dict_file, phase="train"
    datasets = { p :  FragmentsDataset(os.path.join(os.getcwd(), opts.data_dir), labels_path, p)
                 for p in ("train", "val") }
    dataloaders = {
        p : DataLoader(
            datasets[p],
            batch_size=opts.batch_size if p == "train" else 8,
            shuffle=True,
            num_workers=0 #, collate_fn=lambda b : list(list(l) for l in zip(*b))
        ) for p in ("train", "val")
    }

    # set up the model
    # num_categories, batch_size, rnn_size=2048, num_rnn_layers=3, use_cuda=None, max_w=None
    net = crnn_classifier.Encoder(
        datasets["train"].get_y_count(),
        opts.batch_size,
        512,
        1,
        use_cuda=cuda_dev,
        max_w=opts.window_size
    )
    if opts.load:
        saved_state = torch.load(opts.load, map_location='cpu')
        net.load_state_dict(saved_state, strict=False)
    if cuda_dev is not None:
        net = net.cuda(cuda_dev)

    # set up the loss function and optimizer
    class_probs = torch.FloatTensor(max(datasets["train"].x_counts.keys()) + 1)
    for idx, count in datasets["train"].x_counts.items():
        class_probs[idx] = count
    class_probs /= sum(class_probs)
    class_weights = torch.FloatTensor(torch.FloatTensor([1. for x in class_probs]) - class_probs)
    lf = nn.CrossEntropyLoss(weight=class_weights.cuda(cuda_dev))

    learn_params, freeze_params = net.parameters(), []
    if opts.freeze:
        if not opts.load:
            print("error: option --freeze only makes sense with option --load")
            exit(1)
        else:
            freeze_modules = [net.conv1, net.conv2, net.conv3, net.conv4, net.conv5, net.conv6, net.batchnorm256,
                              net.batchnorm256_2, net.batchnorm128, net.batchnorm64]
            for m in freeze_modules:
                freeze_params += list(m.parameters())
            learn_params = list(set(learn_params) - set(freeze_params))

            half_freeze_params = set(net.rnn.parameters())
            learn_params = list(set(learn_params) - half_freeze_params)
            half_freeze_params = list(half_freeze_params)

    base_lr = 10**(-opts.init_lr)
    freeze_lr = opts.freeze_factor * base_lr
    half_freeze_lr = 0.1 * base_lr

    optim = torch.optim.SGD(
        [   {'params' : learn_params, 'lr' : base_lr},
            {'params' : freeze_params, 'lr' : freeze_lr},
            {'params' : half_freeze_params, 'lr' : half_freeze_lr}],
        lr=base_lr,
        momentum=0.95
    )

    # (model, phase, dataloader, batch_size, loss_fn, optim, num_epochs=50)
    train(net, "train", dataloaders, opts.batch_size, lf, optim, opts.max_epochs, cuda_dev=cuda_dev, model_dir=opts.model_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="pprocessed")
    parser.add_argument("--max_epochs", type=int, default=1000)
    parser.add_argument("--window_size", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("-m", "--model_dir", default="model_piece_classifier")
    parser.add_argument("-s", "--seed", type=int, default=0)
    parser.add_argument("-c", "--use_cuda", type=int, default=None)
    parser.add_argument("-l", "--init_lr", type=int, default=5)
    parser.add_argument("--load", default=None)
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--freeze_factor", type=float, default=0.01)

    args = parser.parse_args()
    main(args)