import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.transforms import RandomCrop
from torch import cuda
import sys

# def conv1x1(ch_in, ch_out, stride=1):
#     return nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=stride, bias=False)
#
#
# def convnx1(ch_in, ch_h, ch_out, k, d):
#     assert (type(k) is int and type(d) is int)
#
#     if k == 3:
#         return nn.Conv2d(ch_in, ch_out, (3, 1), 1, (d, 0), (d, 1))
#
#     num_3x1s = k // 2
#     padding = d
#
#     # layers = [nn.Conv2d(ch_h, ch_h, (3, 1), 1, (padding, 0), (d, 1)) for _ in range(num_3x1s)]
#     layers = []
#     for _ in range(num_3x1s):
#         layers += [
#             nn.Conv2d(ch_h, ch_h, (3, 1), 1, (padding, 0), (d, 1)),
#             nn.BatchNorm2d(ch_h),
#             nn.ReLU()
#         ]
#
#     if ch_h == ch_out:
#         assert (ch_h == ch_in)
#     else:
#         layers = [nn.Conv2d(ch_in, ch_h, 1), nn.BatchNorm2d(ch_h), nn.ReLU()] + layers + \
#                  [nn.Conv2d(ch_h, ch_out, 1), nn.BatchNorm2d(ch_out), nn.ReLU()]
#
#     return nn.Sequential(*layers)
#
#
# def conv1xn(ch_in, ch_h, ch_out, k, d):
#     assert(type(k) is int and type(d) is int)
#
#     if k == 3:
#         return nn.Conv2d(ch_in, ch_out, (1, 3), 1, (0, d), (1, d))
#
#     num_1x3s = k // 2
#     padding = d
#
#     layers = [nn.Conv2d(ch_h, ch_h, (1, 3), 1, (0, padding), (1, d)) for _ in range(num_1x3s)]
#     layers = []
#     for _ in range(num_1x3s):
#         layers += [
#             nn.Conv2d(ch_h, ch_h, (1, 3), 1, (0, padding), (1, d)),
#             nn.BatchNorm2d(ch_h),
#             nn.ReLU()
#         ]
#
#     if ch_h == ch_out:
#         assert(ch_in == ch_h)
#     else:
#         layers = [nn.Conv2d(ch_in, ch_h, 1), nn.BatchNorm2d(ch_h), nn.ReLU()] + layers + \
#                  [nn.Conv2d(ch_h, ch_out, 1), nn.BatchNorm2d(ch_out), nn.ReLU()]
#
#     return nn.Sequential(*layers)


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


class ConvResidualBlock(nn.Module):
    def __init__(
            self, ch_in, ch_h, ch_out,
            kernel_size=(7, 7), stride=1, padding=(3, 3), dilation=1
    ):
        super(ConvResidualBlock, self).__init__()

        self.relu = nn.ReLU(inplace=True)
        self.resample = None
        if ch_in != ch_out:
            self.resample = nn.Sequential(
                conv1x1(ch_in, ch_out),
                nn.BatchNorm2d(ch_out)
            )

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
        identity = x
        z = self.net(x)
        if self.resample is not None:
            identity = self.resample(x)
        return self.relu(z + identity)


class FCNN(nn.Module):
    def __init__(self, num_categories, batch_size=16, use_cuda=None, timeframes=None):
        super(FCNN, self).__init__()
        self.timeframes = timeframes
        self.batch_size = batch_size
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, (3, 7), padding=(1, 3)), nn.BatchNorm2d(32), nn.ReLU(),

            # ConvResidualBlock(32, 32, 64, (3, 7), padding=(2, 6), stride=1, dilation=2), nn.ReLU(),
            # ConvResidualBlock(64, 32, 128, (3, 7), padding=(2, 4), stride=1, dilation=4), nn.ReLU(),
            # ConvResidualBlock(128, 32, 128, (3, 7), padding=2, stride=1, dilation=8), nn.ReLU(),
            # ConvResidualBlock(128, 32, 128, (3, 7), padding=2, stride=1, dilation=16), nn.ReLU(),
            # ConvResidualBlock(128, 32, 128, (3, 7), padding=2, stride=1, dilation=32), nn.ReLU(),

            nn.Conv2d(32, 64, (3, 7), padding=(2, 6), stride=1, dilation=2), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, (3, 7), padding=(4, 12), stride=1, dilation=4), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 256, (3, 7), padding=(8, 24), stride=1, dilation=8), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, (3, 7), padding=(16, 48), stride=1, dilation=16), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, (3, 7), padding=(32, 96), stride=1, dilation=32), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, num_categories, (128, 3), padding=0, stride=1)
        )

        print(self.net, file=sys.stderr)

        if use_cuda is not None:
            self.use_cuda = use_cuda
        else:
            self.use_cuda = None


    def forward(self, input):
        batch_size = input.shape[0]

        x_batch = input
        if self.use_cuda is not None:
            x_batch = x_batch.cuda(self.use_cuda)

        return self.net(x_batch)
