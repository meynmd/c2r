import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch import cuda

from masked_conv import MaskedConv2d


def conv1x1(ch_in, ch_out, stride=1):
    return nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=stride, bias=False)


def convnx1(mask_type, ch, k, d):
    assert (type(k) is int and type(d) is int)
    if k == 3:
        return MaskedConv2d(mask_type, ch, ch, (3, 1), 1, (d, 0), (d, 1))

    num_3x1s = k // 2
    padding = d

    layers = []
    for _ in range(num_3x1s):
        layers += [
            MaskedConv2d(mask_type, ch, ch, (3, 1), 1, (padding, 0), (d, 1)),
            nn.BatchNorm2d(ch),
            nn.ReLU()
        ]
    return nn.Sequential(*layers)


def conv1xn(mask_type, ch, k, d):
    assert (type(k) is int and type(d) is int)
    if k == 3:
        return MaskedConv2d(mask_type, ch, ch, (1, 3), 1, (0, d), (1, d))

    num_1x3s = k // 2
    padding = d

    layers = []
    for _ in range(num_1x3s):
        layers += [
            MaskedConv2d(mask_type, ch, ch, (1, 3), 1, (0, padding), (1, d)),
            nn.BatchNorm2d(ch),
            nn.ReLU()
        ]

    return nn.Sequential(*layers)


class ConvBlock(nn.Module):
    def __init__(
            self, mask_type, ch_in, ch_h, ch_out,
            kernel_size=(7, 7), stride=1, dilation=1
    ):
        super(ConvBlock, self).__init__()

        kernel_size, stride, dilation = ((var, var) if type(var) is int else var
                                                  for var in (kernel_size, stride, dilation))
        self.layers = [
            convnx1(mask_type, ch_h, kernel_size[0], dilation[0]),
            conv1xn(mask_type, ch_h, kernel_size[1], dilation[1]),
        ]
        if ch_in != ch_h:
            self.layers = [nn.Conv2d(ch_in, ch_h, 1), nn.BatchNorm2d(ch_h), nn.ReLU()] + self.layers
        if ch_h != ch_out:
            self.layers += [nn.Conv2d(ch_h, ch_out, 1), nn.BatchNorm2d(ch_out), nn.ReLU()]

        self.net = nn.Sequential(*self.layers)


    def forward(self, x):
        z = self.net(x)
        return z


class Downsample(nn.Module):
    def __init__(self, batch_size, use_cuda=None, max_w=None):
        super(Downsample, self).__init__()
        self.name = 'encoder cnn v2'
        self.max_w = max_w
        self.batch_size = batch_size
        self.cuda_dev = use_cuda

        self.conv_layers = nn.Sequential(
            ConvBlock('A', 1, 8, 8, (9, 3), stride=(1, 1), dilation=(1, 1)),        # conv1
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),

            ConvBlock('B', 8, 16, 16, (7, 3), stride=(1, 1), dilation=(1, 1)),      # conv2
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),

            ConvBlock('B', 16, 32, 32, (5, 3), stride=(1, 1), dilation=(1, 1)),     # conv3
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),

            ConvBlock('B', 32, 64, 64, (3, 3), stride=(1, 1), dilation=(1, 1)),  # conv3b
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),

            ConvBlock('B', 64, 128, 128, (3, 3), stride=(1, 1), dilation=(2, 2)),     # conv4

            ConvBlock('B', 128, 128, 256, (3, 3), stride=(1, 1), dilation=(4, 4)),    # conv5

            ConvBlock('B', 256, 256, 256, (3, 3), stride=(1, 1), dilation=(8, 8)),  # conv6

            # ConvBlock('B', 512, 512, 512, (3, 3), stride=(1, 1), dilation=(16, 16)) # conv7

        )

    def forward(self, x_batch):
        if (self.cuda_dev is not None) and (not x_batch.is_cuda):
            x_batch = x_batch.cuda(self.cuda_dev)

        z = self.conv_layers(x_batch)
        return z


class Upsample(nn.Module):
    def __init__(self, batch_size, use_cuda=None, max_w=None):
        super(Upsample, self).__init__()
        self.name = 'decoder cnn v2'
        self.max_w = max_w
        self.batch_size = batch_size
        self.cuda_dev = use_cuda

        self.conv_layers = nn.Sequential(
            nn.ConvTranspose2d(256, 128, (1, 3), stride=(1, 2), padding=(0, 1), output_padding=(0, 1)),
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, (1, 3), stride=(1, 2), padding=(0, 1), output_padding=(0, 1)),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, (1, 3), stride=(1, 2), padding=(0, 1), output_padding=(0, 1)),
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.ConvTranspose2d(32, 32, (1, 3), stride=(1, 2), padding=(0, 1), output_padding=(0, 1)),
            nn.BatchNorm2d(32), nn.ReLU(),
        )

    def forward(self, x):
        return self.conv_layers(x)


class AutoEncoder(nn.Module):
    def __init__(self, *args, **kwargs):
        super(AutoEncoder, self).__init__()
        self.name = 'cnn autoencoder'

        self.downsample = Downsample(*args, **kwargs)

        self.upsample = Upsample(*args, **kwargs)

        self.rnn_size = 1024
        self.rnn = nn.LSTM(32*128, self.rnn_size, num_layers=1, bidirectional=False)

        self.projection = nn.Conv2d(1, 128, (self.rnn_size, 1), stride=1, padding=0)

    def forward(self, x):
        z = self.downsample(x)
        z = self.upsample(z)

        z = z.view(
                    z.data.shape[0],
                    z.data.shape[1]*z.data.shape[2],
                    z.data.shape[3]
        ).transpose(0, 2).transpose(1, 2)

        z, (x, y) = self.rnn(z)
        z = z.transpose(0, 1).transpose(1, 2).unsqueeze(1)  # n, b, h ==> b, 1, h, n
        z = self.projection(z).transpose(1, 2)

        return z