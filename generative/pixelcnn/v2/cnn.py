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


class Encoder(nn.Module):
    def __init__(self, batch_size, use_cuda=None, max_w=None):
        super(Encoder, self).__init__()
        self.name = 'encoder cnn v2'
        self.max_w = max_w
        self.batch_size = batch_size
        self.cuda_dev = use_cuda

        self.conv_layers = nn.Sequential(
            ConvBlock('A', 1, 16, 16, (3, 7), stride=(1, 1), dilation=(1, 2)),        # conv1
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),

            ConvBlock('B', 16, 16, 32, (3, 5), stride=(1, 1), dilation=(1, 4)),       # conv2
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),

            ConvBlock('B', 32, 32, 64, (3, 3), stride=(1, 1), dilation=(2, 8)),       # conv3
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),

            ConvBlock('B', 64, 64, 128, (3, 3), stride=(1, 1), dilation=(4, 16)),      # conv4
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),

            ConvBlock('B', 128, 128, 256, (3, 3), stride=(1, 1), dilation=(8, 32)),    # conv5
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),

            ConvBlock('B', 256, 256, 512, (3, 3), stride=(1, 1), dilation=(16, 64)),    # conv6
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),

            ConvBlock('B', 512, 512, 1024, (3, 3), stride=(1, 1), dilation=(32, 128)),   # conv7
        )

    def forward(self, x_batch):
        if (self.cuda_dev is not None) and (not x_batch.is_cuda):
            x_batch = x_batch.cuda(self.cuda_dev)

        z = self.conv_layers(x_batch)
        return z


class Decoder(nn.Module):
    def __init__(self, batch_size, use_cuda=None, max_w=None):
        super(Decoder, self).__init__()
        self.name = 'decoder cnn v2'
        self.max_w = max_w
        self.batch_size = batch_size
        self.cuda_dev = use_cuda

        self.conv_layers = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, 3, stride=(1, 2), padding=1, output_padding=(0, 1)),
            nn.BatchNorm2d(512), nn.ReLU(),
            nn.ConvTranspose2d(512, 256, 3, stride=(1, 2), padding=1, output_padding=(0, 1)),
            nn.BatchNorm2d(256), nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 3, stride=(1, 2), padding=1, output_padding=(0, 1)),
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 3, stride=(1, 2), padding=1, output_padding=(0, 1)),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=(1, 2), padding=1, output_padding=(0, 1)),
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 3, stride=(1, 2), padding=1, output_padding=(0, 1)),
        )

    def forward(self, x):
        return self.conv_layers(x)


class AutoEncoder(nn.Module):
    def __init__(self, *args, **kwargs):
        super(AutoEncoder, self).__init__()
        self.name = 'cnn autoencoder'
        self.encoder = Encoder(*args, **kwargs)
        self.decoder = Decoder(*args, **kwargs)

    def forward(self, x):
        z = self.encoder(x)
        z = self.decoder(z)
        return z
