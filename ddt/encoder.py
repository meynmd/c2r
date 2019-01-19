import sys
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
        self.rnn_size = rnn_size
        self.rnn_layers = num_rnn_layers
        self.max_w = max_w
        self.batch_size = batch_size
        self.conv1 = nn.Conv2d(1, 16, (3, 5), padding=(1, 2), stride=(1, 1))
        self.maxpool_narrow = nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))
        self.maxpool_square = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.conv2 = nn.Conv2d(16, 32, (3, 5), padding=(1, 2), stride=(1, 1))
        self.batchnorm64 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(32, 64, (3, 5), padding=(1, 2), stride=(1, 1))
        self.batchnorm128 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(64, 128, (3, 3), padding=(1, 1), stride=(1, 1))
        self.maxpool_wide = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))
        self.conv5 = nn.Conv2d(128, 256, (3, 3), padding=(1, 1), stride=(1, 1))
        self.batchnorm256 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 256, (3, 3), padding=(1, 1), stride=(1, 1))
        self.batchnorm256_2 = nn.BatchNorm2d(256)

        self.rnn = nn.LSTM(8192, rnn_size, num_layers=self.rnn_layers, bidirectional=False)

        # self.batchnorm1d = nn.BatchNorm1d(1024)
        self.fc1 = nn.Linear(rnn_size, 128)
        # self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_categories)

        if use_cuda is not None:
            self.use_cuda = use_cuda
        else:
            self.use_cuda = None

    def forward(self, input):
        batch_size = input.shape[0]

        x_batch = input
        if self.use_cuda is not None:
            x_batch = x_batch.cuda(self.use_cuda)

        features = self.maxpool_wide(F.relu(self.conv1(x_batch)))
        features = self.maxpool_wide(F.relu(self.conv2(features)))
        features = self.maxpool_wide(F.relu(self.batchnorm64(self.conv3(features))))
        features = self.maxpool_square(F.relu(self.batchnorm128(self.conv4(features))))
        # features = self.batchnorm256(self.conv5(features))

        features = self.maxpool_square(F.relu(self.batchnorm256(self.conv5(features))))
        features = self.batchnorm256_2(self.conv6(features))

        # print('features shape: {}'.format(features.data.shape))

        rnn_in = features.view(features.data.shape[0],
                               features.data.shape[1]*features.data.shape[2],
                               features.data.shape[3]).transpose(0, 2).transpose(1, 2)
        output, (hidden, _) = self.rnn(rnn_in)

        return output


    def locate_transitions(self, x, avg_window=10, points_max=10):
        z = self.forward(x).squeeze(1)
        if z.shape[0] <= avg_window:
            print('the output sequence is way too short!', file=sys.stderr)
            return None

        dot_prods = torch.zeros(z.shape[0])
        dot_prods[:avg_window] = 1.

        running_avg =torch.mean(z[:avg_window, :], 0)
        # running_avg = running_avg / torch.norm(running_avg)

        # last_vec = z[0, :] / torch.norm(z[0, :])

        for i in range(avg_window, z.shape[0]):
            cur_vec = z[i, :] / torch.norm(z[i, :])

            # mean_vec = torch.mean(z[max(0, i-avg_window) : i], 0)
            # mean_vec /= torch.norm(mean_vec)

            dot_prods[i] = torch.dot(running_avg/torch.norm(running_avg), cur_vec)
            running_avg = running_avg*(avg_window - 1.)/avg_window + z[i, :]*1./avg_window

            # dot_prods[i] = torch.dot(last_vec, cur_vec)
            # last_vec = cur_vec

        vals, idxs = torch.topk(dot_prods, points_max, 0, False)
        return z, dot_prods, vals, idxs


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
