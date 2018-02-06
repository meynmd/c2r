import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import cuda
import torch.legacy.nn as legacy

class Encoder(nn.Module):
    def __init__(self, rnn_size=128, use_cuda=False):
        super(Encoder, self).__init__()
        self.learning_rate = 0.05
        input_size = 512
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, (3, 3), padding=(1, 1), stride=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, (3, 3), padding=(1, 1), stride=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, (3, 3), padding=(1, 1), stride=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 256, (3, 3), padding=(1, 1), stride=(1, 1)),
            nn.ReLU(),
            # not sure what this means:
            # model:add(cudnn.SpatialConvolution(256, 256, 3, 3, 1, 1, 1, 1)) -- (batch_size, 256, imgH/2/2, imgW/2/2)
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 512, (3, 3), padding=(1, 1), stride=(1, 1)),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            # model:add(cudnn.SpatialMaxPooling(2, 1, 2, 1, 0, 0))
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(512, 512, (3, 3), padding=(1, 1), stride=(1, 1)),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )

        self.cnn = nn.Sequential(self.conv1, self.conv2, self.conv3, self.conv4, self.conv5, self.conv6)
        self.rnn = nn.LSTM(8192, rnn_size, num_layers=3, dropout=0, bidirectional=False)
        self.pos_lut = nn.Embedding(1000, input_size)

        if use_cuda:
            self.cnn = self.cnn.cuda()
            self.rnn = self.rnn.cuda()
            self.pos_lut = self.pos_lut.cuda()

    def forward(self, input):
        batch_size = input.size()[0]
        conv_features = self.cnn(input)
        del input
        all_channels = conv_features.view(512*16, 1, -1).transpose(0, 2)
        del conv_features
        output, hidden = self.rnn(all_channels)
        del all_channels

        # all_out = []
        # for col in range(input.size(3)):
        #     inp = input[:, :, :, col]
        #     col_vec = torch.Tensor(batch_size).type_as(inp.data).long().fill_(col)
        #     pos_emb = self.pos_lut(Variable(row_vec))
        #     with_pos = torch.cat(
        #         (pos_emb.view(1, pos_emb.size(0), pos_emb.size(1)), inp), 0
        #     )
        #     outputs, hidden_t = self.rnn(with_pos)
        #     all_out.append(outputs)

        # out = torch.cat(all_out, 0)

        return hidden, output


