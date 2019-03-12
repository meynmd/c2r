import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.transforms import RandomCrop
from torch import cuda

class Encoder(nn.Module):
    def __init__(self, num_categories, batch_size, rnn_size=2048, num_rnn_layers=3, use_cuda=None, max_w=None):
        super(Encoder, self).__init__()
        self.name = 'v0'
        self.rnn_size = rnn_size
        self.rnn_layers = num_rnn_layers
        self.max_w = max_w
        self.batch_size = batch_size

        self.maxpool_narrow = nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))
        self.maxpool_square = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.maxpool_wide = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))

        self.conv1 = nn.Conv2d(1, 16, (3, 5), padding=(1, 2), stride=(1, 1))
        self.bn1 = nn.BatchNorm2d(16)

        self.conv2 = nn.Conv2d(16, 32, (3, 5), padding=(1, 2), stride=(1, 1))
        self.bn2 = nn.BatchNorm2d(32)

        self.batchnorm64 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(32, 64, (3, 5), padding=(1, 2), stride=(1, 1))
        self.bn3 = nn.BatchNorm2d(64)

        self.batchnorm128 = nn.BatchNorm2d(128)

        self.conv4 = nn.Conv2d(64, 128, (3, 3), padding=(1, 1), stride=(1, 1))
        self.bn4 = nn.BatchNorm2d(128)

        self.conv5 = nn.Conv2d(128, 256, (3, 3), padding=(1, 1), stride=(1, 1))
        self.bn5 = nn.BatchNorm2d(256)

        self.batchnorm256 = nn.BatchNorm2d(256)

        self.conv6 = nn.Conv2d(256, 512, (3, 3), padding=(1, 1), stride=(1, 1))
        self.bn6 = nn.BatchNorm2d(512)

        self.batchnorm512 = nn.BatchNorm2d(512)


        self.rnn = nn.LSTM(int(512*128/2**2), rnn_size, num_layers=self.rnn_layers, bidirectional=False) #, dropout=0.5)

        self.batchnorm_fc = nn.BatchNorm1d(512)
        self.fc1 = nn.Linear(rnn_size, 512)
        # self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(512, num_categories)

        if use_cuda is not None:
            self.use_cuda = use_cuda
        else:
            self.use_cuda = None

    def forward(self, input):
        batch_size = input.shape[0]

        x_batch = input
        if self.use_cuda is not None:
            x_batch = x_batch.cuda(self.use_cuda)

        # features = self.maxpool_wide(F.relu(self.conv1(x_batch)))
        # features = self.maxpool_wide(F.relu(self.conv2(features)))
        # features = self.maxpool_wide(F.relu(self.batchnorm64(self.conv3(features))))
        # features = self.maxpool_square(F.relu(self.batchnorm128(self.conv4(features))))
        # features = self.maxpool_square(F.relu(self.batchnorm256(self.conv5(features))))
        # features = self.batchnorm512(self.conv6(features))

        features = self.maxpool_wide(F.relu(self.bn1(self.conv1(x_batch))))
        features = self.maxpool_wide(F.relu(self.bn2(self.conv2(features))))
        features = self.maxpool_wide(F.relu(self.bn3(self.conv3(features))))
        features = self.maxpool_square(F.relu(self.bn4(self.conv4(features))))
        features = self.maxpool_square(F.relu(self.bn5(self.conv5(features))))
        features = F.relu(self.bn6(self.conv6(features)))

        rnn_in = features.view(features.data.shape[0],
                               features.data.shape[1]*features.data.shape[2],
                               features.data.shape[3]).transpose(0, 2).transpose(1, 2)
        output, (hidden, _) = self.rnn(rnn_in)

        hidden = hidden.view(batch_size, self.rnn_size)
        out = F.relu(self.batchnorm_fc(self.fc1(hidden)))
        # out = F.relu(self.fc2(out))

        return self.fc3(out)


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
