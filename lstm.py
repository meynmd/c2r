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
        self.rnn = nn.LSTM(128, rnn_size, num_layers=self.rnn_layers, bidirectional=False)

        self.batchnorm1d = nn.BatchNorm1d(1024)
        self.fc1 = nn.Linear(rnn_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_categories)

        if use_cuda is not None:
            self.use_cuda = use_cuda
        else:
            self.use_cuda = None

    def forward(self, input):
        batch_size = len(input)
        max_w = min(self.max_w, min(t.shape[1] for t in input))
        x_batch = []
        for tensor in input:
            tensor = tensor.view(1, 1, tensor.shape[0], tensor.shape[1])
            if tensor.shape[3] > max_w:
                tensor = self.random_crop(tensor, max_w)
            x_batch.append(tensor)

        x_batch = Variable(torch.cat(x_batch, 0))
        if self.use_cuda is not None:
            x_batch = x_batch.cuda(self.use_cuda)

        output, (hidden, _) = self.rnn(x_batch)

        hidden = hidden.view(batch_size, self.rnn_size)
        out = F.relu(self.fc1(hidden))
        out = F.relu(self.fc2(out))
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