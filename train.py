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

# import onmt
# import onmt.io
# import onmt.Models
# import onmt.ModelConstructor
# import onmt.modules
# from onmt.Utils import use_gpu

import encoder
import pr_dataset

def train(model, phase, dataloader, batch_size, loss_fn, optim, num_epochs=5, num_batches_val=1,
          model_dir="model", beam_size=10, use_cuda=False, learning_rate_init=0.05, lr_decay=None,
          start_decay_at=None):

    loss, num_seen, num_samples, num_nonzero, accuracy = 0, 0, 0, 0, 0
    learning_rate = learning_rate_init
    model.learning_rate = learning_rate
    prev_loss = None
    val_losses = []
    if phase == "train":
        forward_only = False
    elif phase == "test":
        forward_only = True
        num_epochs = 1
        model.global_step = 0
    else:
        raise NameError("phase must be either train or test")

    # num_batches = len(train_data) // batch_size
    # print("learning rate: {}".format(learning_rate), file=sys.stderr)

    for epoch in range(0, num_epochs):
        for phase in ("train", "val"):
            running_loss = 0.
            for i, data in enumerate(dataloader[phase]):
                x, y = data
                # y is a list because it would be a minibatch of labels
                y = Variable(y[0])
                optim.zero_grad()
                z = model(x)
                _, predicted = torch.max(z.data, 1)
                loss = loss_fn(z, y)
                if phase == "train":
                    loss.backward()
                    optim.step()
                print("Loss : {}".format(loss.cpu().data), end="\r")
                running_loss += loss.data[0] * x.size(0)


            # train_idxs = [i for i in range(len(train_data))]
            # random.shuffle(train_idxs)
            # for batch in range(num_batches):
            #     # x_list, y_list = [list(t) for t in zip(*[
            #     #     xy for xy in train_data[batch*batch_size : (batch + 1)*batch_size]
            #     # ])]
            #     h = train_data[0][0].shape[0]
            #     w_max = max(x.shape[1] for x, y in train_data)
            #     x_batch = torch.FloatTensor(batch_size, 1, h, w_max)
            #     for i in range(batch_size):
            #         x, y = train_data[train_idxs[batch*batch_size + i]]
            #         x = np.pad(x, ((0, 0), (0, w_max - x.shape[1])), "constant", constant_values=(0, 0))
            #         x = torch.FloatTensor(x)
            #         x_batch[i, 0, :, :] = x
            #         del x
            #     x_batch = Variable(x_batch).cuda()
            #     z = model(x_batch)
            #     del x_batch


            # for idx in train_idxs:
            #     x, y = train_data[idx]
            #     x = Variable(torch.from_numpy(x)).cuda()
            #     z = model(x)

            print("Epoch {}\ttrain err: {}".format(epoch + 1, running_loss))


def load_data(path):
    data = []
    for d in os.listdir(path):
        filenames = glob.glob(path + "/" + d + "/*.npy")
        data += [(np.load(f), d) for f in filenames]
    return data





def main(opts):
    if opts.use_cuda:
        print("using CUDA", file=sys.stderr)
    # data = load_data("./" + opts.data_dir)
    datasets = { p : pr_dataset.PianoRollDataset(os.getcwd() + "/" + opts.data_dir, "labels.csv", p)
                 for p in ("train", "val") }
    dataloaders = {
        p : DataLoader(
            datasets[p],
            batch_size=opts.batch_size,
            shuffle=True,
            num_workers=2,
            collate_fn=lambda b : list(list(l) for l in zip(*b))
        ) for p in ("train", "val")
    }
    enc = encoder.Encoder(2, rnn_size=128, use_cuda=True)
    enc = enc.cuda()
    lf = nn.CrossEntropyLoss()
    optim = torch.optim.SGD(enc.parameters(), lr=0.01, momentum=0.9)
    train(enc, "train", dataloaders, 1, lf, optim, 5, use_cuda=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="preprocessed")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--batch_size", default=1)
    parser.add_argument("-m", "--model_dir", default="model")
    parser.add_argument("-e", "--num_epochs", default=5)
    parser.add_argument("--num_batch_valid", default=1)
    parser.add_argument("--beam_size", default=5)
    parser.add_argument("-s", "--seed", default=0)
    parser.add_argument("-c", "--use_cuda", action="store_true")

    args = parser.parse_args()
    main(args)