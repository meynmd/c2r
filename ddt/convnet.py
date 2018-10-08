import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.transforms import RandomCrop
from torch import cuda

class ConvNet(nn.Module):
    def __init__(self, num_categories, batch_size, use_cuda=None, max_w=None):
        super(ConvNet, self).__init__()

        self.max_w = max_w
        self.batch_size = batch_size
        self.conv1 = nn.Conv2d(1, 16, (3, 5), padding=(1, 2), stride=(1, 1))
        self.maxpool_narrow = nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))
        self.maxpool_square = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.conv2 = nn.Conv2d(16, 32, (3, 3), padding=(1, 1), stride=(1, 1))
        self.batchnorm64 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(32, 64, (3, 3), padding=(1, 1), stride=(1, 1))
        self.batchnorm128 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(64, 128, (3, 3), padding=(1, 1), stride=(1, 1))
        self.maxpool_wide = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))
        self.conv5 = nn.Conv2d(128, 256, (3, 3), padding=(1, 1), stride=(1, 1))
        self.batchnorm256 = nn.BatchNorm2d(256)
        self.batchnorm1d = nn.BatchNorm1d(1024)

        if use_cuda is not None:
            self.use_cuda = use_cuda
        else:
            self.use_cuda = None

    def forward(self, input):
        x_batch = input
        if self.use_cuda is not None:
            x_batch = x_batch.cuda(self.use_cuda)

        features = F.relu(self.conv1(x_batch))
        features = self.maxpool_narrow(features)
        features = self.maxpool_square(F.relu(self.conv2(features)))
        features = self.maxpool_square(F.relu(self.batchnorm64(self.conv3(features))))
        features = self.maxpool_wide(F.relu(self.batchnorm128(self.conv4(features))))
        features = self.conv5(features)
        return features
