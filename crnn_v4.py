import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.transforms import RandomCrop
from torch import cuda

def conv1x1(ch_in, ch_out, stride=1):
    return nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=stride, bias=False)


def convnx1(ch, k, d):
    assert (type(k) is int and type(d) is int)
    if k == 3:
        return nn.Conv2d(ch, ch, (3, 1), 1, (d, 0), (d, 1))

    num_3x1s = k // 2
    padding = d

    layers = []
    for _ in range(num_3x1s):
        layers += [
            nn.Conv2d(ch, ch, (3, 1), 1, (padding, 0), (d, 1)),
            nn.BatchNorm2d(ch),
            nn.ReLU()
        ]

    return nn.Sequential(*layers)


def conv1xn(ch, k, d):
    assert (type(k) is int and type(d) is int)
    if k == 3:
        return nn.Conv2d(ch, ch, (1, 3), 1, (0, d), (1, d))

    num_1x3s = k // 2
    padding = d

    layers = []
    for _ in range(num_1x3s):
        layers += [
            nn.Conv2d(ch, ch, (1, 3), 1, (0, padding), (1, d)),
            nn.BatchNorm2d(ch),
            nn.ReLU()
        ]

    return nn.Sequential(*layers)


class ConvBlock(nn.Module):
    def __init__(
            self, ch_in, ch_h, ch_out,
            kernel_size=(7, 7), stride=1, padding=(3, 3), dilation=1
    ):
        super(ConvBlock, self).__init__()

        kernel_size, stride, padding, dilation = ((var, var) if type(var) is int else var
                                                  for var in (kernel_size, stride, padding, dilation))

        self.layers = [
            convnx1(ch_h, kernel_size[0], dilation[0]),
            nn.BatchNorm2d(ch_h),
            nn.ReLU(),

            conv1xn(ch_h, kernel_size[1], dilation[1]),
            nn.BatchNorm2d(ch_h),
        ]

        if ch_in != ch_h:
            self.layers = [nn.Conv2d(ch_in, ch_h, 1), nn.BatchNorm2d(ch_h), nn.ReLU()] + self.layers
        if ch_h != ch_out:
            self.layers += [nn.ReLU(), nn.Conv2d(ch_h, ch_out, 1), nn.BatchNorm2d(ch_out)]

        self.net = nn.Sequential(*self.layers)


    def forward(self, x):
        z = self.net(x)
        return z


class Encoder(nn.Module):
    def __init__(self, num_categories, batch_size, rnn_size=2048, num_rnn_layers=3, use_cuda=None, max_w=None):
        super(Encoder, self).__init__()
        self.name = 'v4'
        self.rnn_size = rnn_size
        self.rnn_layers = num_rnn_layers
        self.max_w = max_w
        self.batch_size = batch_size
        if use_cuda is not None:
            self.cuda_dev = use_cuda
        else:
            self.cuda_dev = None

        self.conv_layers = nn.Sequential(
            ConvBlock(1, 16, 16, (3, 9), padding=(1, 2), stride=(1, 1)),        # conv1
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),

            ConvBlock(16, 16, 32, (3, 7), stride=(1, 1), padding=(1, 2)),       # conv2
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),

            ConvBlock(32, 32, 64, (3, 5), padding=(1, 2), stride=(1, 1)),       # conv3
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),

            ConvBlock(64, 64, 128, (3, 3), padding=(1, 1), stride=(1, 1)),      # conv4
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),

            ConvBlock(128, 128, 256, (3, 3), padding=(1, 1), stride=(1, 1)),    # conv5
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),

            ConvBlock(256, 256, 512, (3, 3), padding=(1, 1), stride=(1, 1)),    # conv6
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),

            ConvBlock(512, 512, 1024, (3, 3), padding=(1, 1), stride=(1, 1)),   # conv7
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),

            nn.Conv2d(1024, 2048, kernel_size=(8, 1), padding=(0, 0), stride=(1, 1)),
            nn.BatchNorm2d(2048),
            nn.ReLU()
        )

        # self.rnn = nn.LSTM(int(1024*128/2**4), rnn_size, num_layers=self.rnn_layers, bidirectional=False)
        self.rnn = nn.LSTM(2048, rnn_size, num_layers=self.rnn_layers, bidirectional=False)


        self.classifier = nn.Sequential(
            nn.Linear(rnn_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, num_categories)
        )

    def forward(self, x_batch):
        batch_size = x_batch.shape[0]

        if self.cuda_dev is not None:
            x_batch = x_batch.cuda(self.cuda_dev)

        features = self.conv_layers(x_batch)

        rnn_in = features.view(features.data.shape[0],
                               features.data.shape[1]*features.data.shape[2],
                               features.data.shape[3]).transpose(0, 2).transpose(1, 2)
        output, (hidden, _) = self.rnn(rnn_in)

        return self.classifier(hidden.view(batch_size, self.rnn_size))

