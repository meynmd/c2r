import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch import cuda
import torch.legacy.nn as legacy

class Encoder(nn.Module):
    def __init__(self, num_categories, rnn_size=128, use_cuda=False):
        super(Encoder, self).__init__()
        self.learning_rate = 0.05
        input_size = 512
        self.conv1 = nn.Conv2d(1, 64, (3, 3), padding=(1, 1), stride=(1, 1))
        self.maxpool_square = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.conv2 = nn.Conv2d(64, 128, (3, 3), padding=(1, 1), stride=(1, 1))
        self.conv3 = nn.Conv2d(128, 256, (3, 3), padding=(1, 1), stride=(1, 1))
        self.batchnorm256 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 256, (3, 3), padding=(1, 1), stride=(1, 1))
        self.maxpool_wide = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))
        self.conv5 = nn.Conv2d(256, 512, (3, 3), padding=(1, 1), stride=(1, 1))
        self.batchnorm512 = nn.BatchNorm2d(512)
        self.conv6 = nn.Conv2d(512, 512, (3, 3), padding=(1, 1), stride=(1, 1))

        self.rnn = nn.LSTM(8192, rnn_size, num_layers=1, dropout=0, bidirectional=False)
        self.pos_lut = nn.Embedding(1000, input_size)

        self.fc1 = nn.Linear(rnn_size, 100)
        self.fc2 = nn.Linear(100, num_categories)

        if use_cuda:
            self.use_cuda = True
        else:
            self.use_cuda = False

    def forward(self, input):
        batch_size = len(input)
        print("batch size: {}".format(batch_size))
        max_w = max(t.shape[1] for t in input)
        x_batch = []
        for tensor in input:
            tensor = tensor.view(1, 1, tensor.shape[0], tensor.shape[1])
            if tensor.shape[3] < max_w:
                x_batch.append(
                    torch.cat((tensor, torch.FloatTensor(1, 1, tensor.shape[2], max_w - tensor.shape[3]).fill_(0.)), 3)
                )
            else:
                x_batch.append(tensor)
        del input
        x_batch = Variable(torch.cat(x_batch, 0))
        if self.use_cuda:
            x_batch = x_batch.cuda()

        features = F.relu(self.conv1(x_batch))
        features = self.maxpool_square(features)
        features = self.maxpool_square(F.relu(self.conv2(features)))
        features = F.relu(self.batchnorm256(self.conv3(features)))
        features = self.maxpool_wide(F.relu(self.conv4(features)))
        features = self.maxpool_wide(F.relu(self.batchnorm512(self.conv5(features))))
        features = F.relu(self.batchnorm512(self.conv6(features)))

        all_channels = features.view(512*16, 1, -1).transpose(0, 2)
        del features
        del x_batch
        output, (hidden, _) = self.rnn(all_channels)
        del all_channels
        hidden = hidden.view(-1)
        prediction = F.relu(self.fc1(hidden))
        return self.fc2(prediction)


