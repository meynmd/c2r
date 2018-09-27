import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.transforms import RandomCrop
from torch import cuda

class Generator(nn.Module):
    def __init__(self, batch_size=1, rnn_size=128, num_rnn_layers=1, pr_height=128, use_cuda=None, max_w=None):
        super(Generator, self).__init__()

        self.rnn_dim, self.num_layers, self.batch_size = rnn_size, num_rnn_layers, batch_size
        self.output_size, self.max_w = pr_height, max_w
        self.cuda_dev = use_cuda

        self.rnn = nn.LSTM(self.output_size, rnn_size, num_layers=num_rnn_layers, bidirectional=False)
        self.fc = nn.Linear(self.rnn_dim, self.output_size + 1)
        self.sig = nn.Sigmoid()

    # input: (1, batch_size, rnn_size)
    def forward(self, *input):
        h, c = input
        h, c = Variable(h), Variable(c)

        out_seq = []
        out = Variable(torch.zeros(self.output_size + 1).unsqueeze(0).unsqueeze(0).type(torch.long))

        if self.cuda_dev is not None:
            h, c, out = h.cuda(self.cuda_dev), c.cuda(self.cuda_dev), out.cuda(self.cuda_dev)

        while out[0, 0, -1] != 1:
            if self.max_w and len(out_seq) >= self.max_w:
                break
            input = out[:, :, :self.output_size].type(torch.float)
            z, (h, c) = self.rnn(input, (h, c))
            out = self.sig(self.fc(z)).type(torch.long)
            out_seq.append(out)
            if len(out_seq) % 10 == 0:
                print('seq. length {}'.format(len(out_seq)))

        out_batch = torch.stack(out_seq, dim=3)
        out_batch = out_batch.unsqueeze(0)




