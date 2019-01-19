import argparse
import math
import sys
import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data.dataloader import DataLoader
import pr_dataset

class ConvNet(nn.Module):
    def __init__(self, num_categories, batch_size, use_cuda=None, max_w=None):
        super(ConvNet, self).__init__()
        self.max_w = max_w
        self.batch_size = batch_size
        self.conv1 = nn.Conv2d(1, 16, (3, 9), padding=(1, 4), stride=(1, 1))
        self.maxpool_square = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.conv2 = nn.Conv2d(16, 32, (3, 3), padding=(1, 1), stride=(1, 1))
        self.batchnorm64 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(32, 64, (3, 3), padding=(1, 1), stride=(1, 1))
        self.batchnorm128 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(64, 128, (3, 3), padding=(1, 1), stride=(1, 1))
        self.maxpool_wide = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))
        self.conv5 = nn.Conv2d(128, 256, (3, 3), padding=(1, 1), stride=(1, 1))
        self.batchnorm256 = nn.BatchNorm2d(256)

    def forward(self, input):
        batch_size = len(input)
        x_batch = []
        for tensor in input:
            tensor = tensor.view(1, 1, tensor.shape[0], tensor.shape[1])
            if tensor.shape[3] > self.max_w:
                tensor = tensor[:, :self.max_w]
            if tensor.shape[3] < self.max_w:
                tmp = torch.zeros(tensor.shape[0], tensor.shape[1], tensor.shape[2], self.max_w)
                tmp[:, :, :, tensor.shape[3]] = tensor
                tensor = tmp
            x_batch.append(tensor)

        x_batch = Variable(torch.cat(x_batch, 0))
        if self.use_cuda is not None:
            x_batch = x_batch.cuda(self.use_cuda)

        features = F.relu(self.conv1(x_batch))
        features = self.maxpool_square(features)
        features = self.maxpool_square(F.relu(self.conv2(features)))
        features = self.maxpool_square(F.relu(self.batchnorm64(self.conv3(features))))
        features = self.maxpool_wide(F.relu(self.batchnorm128(self.conv4(features))))
        features = F.relu(self.batchnorm256(self.conv5(features)))

        return features


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
            if phase == "train":
                model.train()
            else:
                model.eval()

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
            print("{} err: {:.0%}\t{} loss: {:.5}".format(
                phase, err / float(i + 1), phase, avg_loss)
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


def run_batch(model, phase, dataloader, cuda_dev=None):
    pass


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
    cnn = ConvNet(
        datasets["train"].get_y_count(),
        opts.batch_size,
        use_cuda=cuda_dev,
        max_w=opts.max_w
    )

    if opts.load:
        pretrained = torch.load(opts.load)
        statedict = cnn.state_dict()
        for param in statedict:
            statedict[param] = pretrained[param]
    if cuda_dev is not None:
        cnn = cnn.cuda(cuda_dev)

    cnn.eval()
    target_class = opts.targetclass
    target_name = datasets["train"].idx2name[target_class]
    for phase in ["train", "val", "test"]:
        class_datapoints = list(datasets[phase].get_from_class(target_name))






if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rnn_size", type=int, default=128)
    parser.add_argument("--rnn_layers", type=int, default=3)
    parser.add_argument("--data_dir", default="preprocessed")
    parser.add_argument("--max_epochs", type=int, default=1000)
    parser.add_argument("--max_w", type=int, default=5000)
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("-m", "--model_dir", default="baseline_model")
    parser.add_argument("--num_batch_valid", type=int, default=1)
    parser.add_argument("--beam_size", type=int, default=5)
    parser.add_argument("-s", "--seed", type=int, default=0)
    parser.add_argument("-c", "--use_cuda", type=int, default=None)
    parser.add_argument("-l", "--init_lr", type=int, default=5)
    parser.add_argument("--load", default=None)

    args = parser.parse_args()
    main(args)
