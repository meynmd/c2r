import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.transforms import RandomCrop
from torch import cuda

class ConvLayers(nn.Module):
    def __init__(self, num_categories, batch_size, use_cuda=None):
        super(ConvLayers, self).__init__()
        # self.window_size = window_size
        self.batch_size = batch_size

        self.maxpool_wide = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))
        self.maxpool_square = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.conv1 = nn.Conv2d(1, 16, (3, 5), padding=(1, 2), stride=(1, 1))
        self.conv2 = nn.Conv2d(16, 32, (3, 5), padding=(1, 2), stride=(1, 1))
        self.batchnorm64 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(32, 64, (3, 5), padding=(1, 2), stride=(1, 1))
        self.batchnorm128 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(64, 128, (3, 3), padding=(1, 1), stride=(1, 1))

        self.conv5_new = nn.Conv2d(128, 128, (3, 3), padding=(1, 1), stride=(1, 1))
        self.batchnorm5 = nn.BatchNorm2d(128)
        self.conv6_new = nn.Conv2d(128, 256, (3, 3), padding=(1, 1), stride=(1, 1))
        self.batchnorm6 = nn.BatchNorm2d(256)
        self.conv7_new = nn.Conv2d(256, 256, (3, 3), padding=(1, 1), stride=(1, 1))
        self.batchnorm7 = nn.BatchNorm2d(256)

        self.out_channels = 256

        if use_cuda is not None:
            self.use_cuda = use_cuda
        else:
            self.use_cuda = None


    def forward(self, input):
        x_batch = input

        features = self.maxpool_wide(F.relu(self.conv1(x_batch)))
        features = self.maxpool_wide(F.relu(self.conv2(features)))
        features = self.maxpool_wide(F.relu(self.batchnorm64(self.conv3(features))))
        features = self.maxpool_square(F.relu(self.batchnorm128(self.conv4(features))))

        features = self.maxpool_square(F.relu(self.batchnorm5(self.conv5_new(features))))
        features = self.maxpool_wide(F.relu(self.batchnorm6(self.conv6_new(features))))
        features = F.relu(self.batchnorm7(self.conv7_new(features)))

        return features


    def random_crop(self, tensor, w_new, h_new=None):
        h, w = tensor.shape[2], tensor.shape[3]
        top, left = 0, 0
        if w_new < w:
            left = np.random.randint(0, w - w_new)
        if h_new is None:
            return tensor[:, :, :, left : left + w_new]
        if h_new < h:
            top = np.random.randint(0, h - h_new)
        return tensor[top : top + h_new, left : left + w_new]
