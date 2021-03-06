import math
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
        self.conv1 = nn.Conv2d(1, 16, (3, 9), padding=(1, 4), stride=(1, 1))
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
        self.fc1 = nn.Linear(253952, self.max_w)
        self.fc2 = nn.Linear(self.max_w, self.max_w // 16)
        self.fc3 = nn.Linear(self.max_w // 16, num_categories)

        if use_cuda is not None:
            self.use_cuda = use_cuda
        else:
            self.use_cuda = None

    def forward(self, input):
        batch_size = len(input)

        # max_w = min(self.max_w, min(t.shape[1] for t in input))
        t_min = min(t.shape[1] for t in input)
        if t_min < self.max_w:
            max_w = t_min
        else:
            max_w = self.max_w

        x_batch = []
        for tensor in input:
            tensor = tensor > 0.
            tensor = tensor.type(torch.FloatTensor)
            tensor = tensor.view(1, 1, tensor.shape[0], tensor.shape[1])
            if tensor.shape[3] > max_w:
                tensor = self.random_crop(tensor, max_w)
            if max_w < self.max_w:
                pl = (self.max_w - max_w) // 2
                pr = math.ceil((self.max_w - max_w) / 2)
                padding = ((0, 0), (0, 0), (0, 0), (pl, pr))
                tensor = tensor.numpy()
                tensor = np.pad(tensor, padding, "constant")
                tensor = torch.from_numpy(tensor)
            x_batch.append(tensor)

        x_batch = Variable(torch.cat(x_batch, 0))
        if self.use_cuda is not None:
            x_batch = x_batch.cuda(self.use_cuda)

        features = F.relu(self.conv1(x_batch))
        features = self.maxpool_square(features)
        features = self.maxpool_square(F.relu(self.conv2(features)))
        features = self.maxpool_square(F.relu(self.batchnorm64(self.conv3(features))))
        features = self.maxpool_wide(F.relu(self.batchnorm128(self.conv4(features))))
        features = F.relu(self.batchnorm256(self.conv5(features)))

        flat = features.view(batch_size, -1)
        out = F.relu(self.fc1(flat))
        out = F.relu(self.fc2(out))
        return self.fc3(out)

        #
        # out = F.relu(self.batchnorm1d(self.fc1(hidden)))
        # out = F.relu(self.fc2(out))
        # return self.fc3(out)

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