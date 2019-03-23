import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch import cuda

from masked_conv import MaskedConv2d

def convnxn(mask_type, ch, k, d):
    print('\t\t{} dilated conv {}x{}:\t{} channels'.format(d, k, k, ch))

    assert (type(k) is int and type(d) is int)
    if k == 3:
        return MaskedConv2d(mask_type, ch, ch, (3, 3), 1, (d, d), (d, d))

    num_3x3s = k // 2
    layers = []
    for _ in range(num_3x3s):
        layers += [
            MaskedConv2d(mask_type, ch, ch, (3, 3), 1, (d, d), (d, d)),
            nn.BatchNorm2d(ch),
            nn.ReLU()
        ]
    return nn.Sequential(*layers)


def conv1x1(ch_in, ch_out, stride=1):
    return nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=stride, bias=False)


def convnx1(mask_type, ch, k, d):
    print('\t\t{} dilated conv {}x1:\t{} channels'.format(d, k, ch))
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
    print('\t\t{} dilated conv 1x{}:\t{} channels'.format(d, k, ch))

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
        print('\tconv block {}x{}'.format(kernel_size[0], kernel_size[1]), end=' ')
        if ch_in != ch_h or ch_h != ch_out:
            print('bottleneck {} => {} => {}'.format(ch_in, ch_h, ch_out))
        else:
            print('{} => {}'.format(ch_in, ch_out))

        super(ConvBlock, self).__init__()

        kernel_size, stride, dilation = ((var, var) if type(var) is int else var
                                                  for var in (kernel_size, stride, dilation))
        if kernel_size[0] == kernel_size[1]:
            self.layers = [convnxn(mask_type, ch_h, kernel_size[0], dilation[0])]
        else:
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
    def __init__(self, batch_size, ch_out=64, use_cuda=None, max_w=None):
        print('Downsample, out channels: {}'.format(ch_out))

        super(Downsample, self).__init__()
        self.channels_out = ch_out
        self.max_w = max_w
        self.batch_size = batch_size
        self.cuda_dev = use_cuda

        self.conv_layers = nn.Sequential(
            ConvBlock('A', 1, 16, 16, (3, 3), stride=(1, 1), dilation=(2, 1)),          # conv1
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),

            ConvBlock('B', 16, 32, 32, (3, 3), stride=(1, 1), dilation=(4, 1)),         # conv2
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),

            ConvBlock('B', 32, 64, 64, (3, 3), stride=(1, 1), dilation=(8, 1)),         # conv3
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),

            ConvBlock('B', 64, 128, 128, (3, 3), stride=(1, 1), dilation=(16, 1)),        # conv4
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),

            ConvBlock('B', 128, 256, 256, (3, 3), stride=(1, 1), dilation=(16, 2)),

            ConvBlock('B', 256, 256, 256, (3, 3), stride=(1, 1), dilation=(16, 4)),

            ConvBlock('B', 256, 256, 256, (3, 3), stride=(1, 1), dilation=(16, 8)),

            ConvBlock('B', 256, 256, 256, (3, 3), stride=(1, 1), dilation=(16, 16)),

            nn.Conv2d(256, self.channels_out, 1),
            nn.BatchNorm2d(self.channels_out),
            nn.ReLU()
        )

    def forward(self, x_batch):
        if (self.cuda_dev is not None) and (not x_batch.is_cuda):
            x_batch = x_batch.cuda(self.cuda_dev)

        z = self.conv_layers(x_batch)
        return z


class Upsample(nn.Module):
    def __init__(self, batch_size, ch_in=256, ch_out=64, use_cuda=None, max_w=None):
        print('Upsample: in-channels {}'.format(ch_in))
        super(Upsample, self).__init__()
        self.channels_in = ch_in
        self.max_w = max_w
        self.batch_size = batch_size
        self.cuda_dev = use_cuda

        self.conv_layers = nn.Sequential(
            nn.ConvTranspose2d(ch_in, 128, (1, 3), stride=(1, 2), padding=(0, 1), output_padding=(0, 1)),
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.ConvTranspose2d(128, 128, (1, 3), stride=(1, 2), padding=(0, 1), output_padding=(0, 1)),
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.ConvTranspose2d(128, 128, (1, 3), stride=(1, 2), padding=(0, 1), output_padding=(0, 1)),
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.ConvTranspose2d(128, ch_out, (1, 3), stride=(1, 2), padding=(0, 1), output_padding=(0, 1)),
        )

    def forward(self, x):
        return self.conv_layers(x)


class AutoEncoder(nn.Module):
    def __init__(self, batch_size, rnn_size=1024, ch_downsample=256, ch_upsample=64, rnn_layers=1, use_cuda=None, max_w=None):
        super(AutoEncoder, self).__init__()

        self.name = 'cnn-rnn autoencoder v7\n'
        print(self.name)

        self.channels_downsample = ch_downsample
        self.channels_upsample = ch_upsample
        self.rnn_size = rnn_size

        self.downsample = Downsample(batch_size, ch_out=ch_downsample, use_cuda=use_cuda, max_w=max_w)
        self.upsample = Upsample(batch_size, ch_in=ch_downsample, ch_out=ch_upsample, use_cuda=use_cuda, max_w=max_w)

        print('LSTM: hidden dim {}'.format(rnn_size))
        self.rnn = nn.LSTM(ch_upsample*88, rnn_size, num_layers=1, bidirectional=False)

        self.projection = nn.Conv2d(1, 88, (self.rnn_size, 1), stride=1, padding=0)



    def forward(self, x):
        z = self.downsample(x)
        z = self.upsample(z)

        z = z.view(
            z.data.shape[0],
            z.data.shape[1] * z.data.shape[2],
            z.data.shape[3]
        ).transpose(0, 2).transpose(1, 2)

        z, _ = self.rnn(z)      # n, b, h
        z = z.transpose(0, 1).transpose(1, 2).unsqueeze(1)  # n, b, h ==> b, 1, h, n
        z = self.projection(z).transpose(1, 2)

        return z
