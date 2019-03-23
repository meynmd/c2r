import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.transforms import RandomCrop
from torch import cuda

def conv1x1(ch_in, ch_out, stride=1):
    return nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=stride, bias=False)


def convnx1(ch_in, ch_h, ch_out, k, d):
    assert (type(k) is int and type(d) is int)

    if k == 3:
        return nn.Conv2d(ch_in, ch_out, (3, 1), 1, (d, 0), (d, 1))

    num_3x1s = k // 2
    padding = d

    # layers = [nn.Conv2d(ch_h, ch_h, (3, 1), 1, (padding, 0), (d, 1)) for _ in range(num_3x1s)]
    layers = []
    for _ in range(num_3x1s):
        layers += [
            nn.Conv2d(ch_h, ch_h, (3, 1), 1, (padding, 0), (d, 1)),
            nn.BatchNorm2d(ch_h),
            nn.ReLU()
        ]

    if ch_h == ch_out:
        assert (ch_h == ch_in)
    else:
        layers = [nn.Conv2d(ch_in, ch_h, 1), nn.BatchNorm2d(ch_h), nn.ReLU()] + layers + \
                 [nn.Conv2d(ch_h, ch_out, 1), nn.BatchNorm2d(ch_out), nn.ReLU()]

    return nn.Sequential(*layers)


def conv1xn(ch_in, ch_h, ch_out, k, d):
    assert(type(k) is int and type(d) is int)

    if k == 3:
        return nn.Conv2d(ch_in, ch_out, (1, 3), 1, (0, d), (1, d))

    num_1x3s = k // 2
    padding = d

    layers = [nn.Conv2d(ch_h, ch_h, (1, 3), 1, (0, padding), (1, d)) for _ in range(num_1x3s)]
    layers = []
    for _ in range(num_1x3s):
        layers += [
            nn.Conv2d(ch_h, ch_h, (1, 3), 1, (0, padding), (1, d)),
            nn.BatchNorm2d(ch_h),
            nn.ReLU()
        ]

    if ch_h == ch_out:
        assert(ch_in == ch_h)
    else:
        layers = [nn.Conv2d(ch_in, ch_h, 1), nn.BatchNorm2d(ch_h), nn.ReLU()] + layers + \
                 [nn.Conv2d(ch_h, ch_out, 1), nn.BatchNorm2d(ch_out), nn.ReLU()]

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
        self.net = nn.Sequential(
            nn.ReLU(),

            nn.Conv2d(ch_in, ch_h, 1),
            nn.BatchNorm2d(ch_h),
            nn.ReLU(),

            # nn.Conv2d(ch_h, ch_h, (kernel_size[0], 1), (stride[0], 1), (padding[0], 0), (dilation[0], 1)),
            convnx1(ch_h, ch_h, ch_h, kernel_size[0], dilation[0]),
            nn.BatchNorm2d(ch_h),
            nn.ReLU(),

            # nn.Conv2d(ch_h, ch_h, (1, kernel_size[1]), (1, stride[1]), (0, padding[1]), (1, dilation[1])),
            conv1xn(ch_h, ch_h, ch_h, kernel_size[1], dilation[1]),
            nn.BatchNorm2d(ch_h),
            nn.ReLU(),

            nn.Conv2d(ch_h, ch_out, 1),
            nn.BatchNorm2d(ch_out)
        )

    def forward(self, x):
        z = self.net(x)
        return z


class Encoder(nn.Module):
    def __init__(self, num_categories, batch_size, rnn_size=2048, num_rnn_layers=3, use_cuda=None, max_w=None):
        super(Encoder, self).__init__()
        self.rnn_size = rnn_size
        self.rnn_layers = num_rnn_layers
        self.max_w = max_w
        self.batch_size = batch_size
        if use_cuda is not None:
            self.cuda_dev = use_cuda
        else:
            self.cuda_dev = None

        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 16, (3, 5), padding=(1, 2), stride=(1, 1)),    # conv1
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),

            nn.Conv2d(16, 32, (3, 5), padding=(1, 2), stride=(1, 1)),   # conv2
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),

            nn.Conv2d(32, 64, (3, 5), padding=(1, 2), stride=(1, 1)),   # conv3
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),

            nn.Conv2d(64, 128, (3, 3), padding=(1, 1), stride=(1, 1)),  # conv4
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),

            nn.Conv2d(128, 256, (3, 3), padding=(1, 1), stride=(1, 1)), # conv5
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),

            nn.Conv2d(256, 512, (3, 3), padding=(1, 1), stride=(1, 1)),  # conv6
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        )

        self.rnn = nn.LSTM(int(512*128/2**3), rnn_size, num_layers=self.rnn_layers, bidirectional=False)

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


    # def random_crop(self, tensor, w_new, h_new=None):
    #     h, w = tensor.shape[2], tensor.shape[3]
    #     top, left = 0, 0
    #     if w_new < w:
    #         left = np.random.randint(0, w - w_new)
    #     if h_new is None:
    #         return tensor[:, :, :, left : left + w_new]
    #     if h_new < h:
    #         top = np.random.randint(0, h - h_new)
    #     return tensor[top : top + h_new, left : left + w_new]
