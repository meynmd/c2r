import argparse
import glob
import os
import sys
import random
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import cuda

# import onmt
# import onmt.io
# import onmt.Models
# import onmt.ModelConstructor
# import onmt.modules
# from onmt.Utils import use_gpu

import encoder

def train(model, phase, train_data, val_data, batch_size, num_epochs, model_dir="model",
          num_batches_val=1, beam_size=10, output_dir=None, use_cuda=False,
          trie=False, learning_rate_init=0.05, lr_decay=None, start_decay_at=None):

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

    num_batches = len(train_data) // batch_size
    print("learning rate: {}".format(learning_rate), file=sys.stderr)

    for epoch in range(0, num_epochs):
        train_idxs = [i for i in range(len(train_data))]
        random.shuffle(train_idxs)
        for batch in range(num_batches):
            # x_list, y_list = [list(t) for t in zip(*[
            #     xy for xy in train_data[batch*batch_size : (batch + 1)*batch_size]
            # ])]
            h = train_data[0][0].shape[0]
            w_max = max(x.shape[1] for x, y in train_data)
            x_batch = torch.FloatTensor(batch_size, 1, h, w_max)
            for i in range(batch_size):
                x, y = train_data[train_idxs[batch*batch_size + i]]
                x = np.pad(x, ((0, 0), (0, w_max - x.shape[1])), "constant", constant_values=(0, 0))
                x = torch.FloatTensor(x)
                x_batch[i, 0, :, :] = x
                del x
            x_batch = Variable(x_batch).cuda()
            z = model(x_batch)
            del x_batch


        # for idx in train_idxs:
        #     x, y = train_data[idx]
        #     x = Variable(torch.from_numpy(x)).cuda()
        #     z = model(x)
        print("Epoch {}\ttrain err: {}".format(epoch + 1, 0))


def load_data(path):
    data = []
    for d in os.listdir(path):
        filenames = glob.glob(path + "/" + d + "/*.npy")
        data += [(np.load(f), d) for f in filenames]
    return data


def main(opts):
    if opts.use_cuda:
        print("using CUDA", file=sys.stderr)
    print("building model...", file=sys.stderr)
    data = load_data("./" + opts.data_dir)
    enc = encoder.Encoder(rnn_size=128, use_cuda=True)
    train(enc, "train", data, None, 1, 5, use_cuda=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="preprocessed")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--batch_size", default=64)
    parser.add_argument("-m", "--model_dir", default="model")
    parser.add_argument("-e", "--num_epochs", default=5)
    parser.add_argument("--num_batch_valid", default=1)
    parser.add_argument("--beam_size", default=5)
    parser.add_argument("-s", "--seed", default=0)
    parser.add_argument("-c", "--use_cuda", action="store_true")

    args = parser.parse_args()
    main(args)