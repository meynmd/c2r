import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class UnFlatten(nn.Module):
    def __init__(self, h_dim):
        super(UnFlatten, self).__init__()
        self.h_dim = h_dim

    def forward(self, input):
        return input.view(input.size(0), self.h_dim, 1, 1)


class VariationalAutoencoder(nn.Module):
    def __init__(self, batch_size, rnn_size=2048, num_rnn_layers=3, use_cuda=None, max_w=None, h_dim=128):
        super(VariationalAutoencoder, self).__init__()

        self.rnn_size = rnn_size
        self.rnn_layers = num_rnn_layers
        self.max_w = max_w
        self.batch_size = batch_size
        self.h_dim = h_dim

        self.conv_sizes = [None for _ in range(5)]

        self.conv1 = nn.Conv2d(1, 16, (3, 9), padding=(1, 4), stride=(2, 2))
        self.conv2 = nn.Conv2d(16, 32, (3, 3), padding=(1, 1), stride=(2, 2))
        self.conv3 = nn.Conv2d(32, 64, (3, 3), padding=(1, 1), stride=(2, 2))
        self.conv4 = nn.Conv2d(64, 128, (3, 3), padding=(1, 1), stride=(1, 2))
        self.conv5 = nn.Conv2d(128, 256, (3, 3), padding=(1, 1), stride=(1, 1))
        self.maxpool_square = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.maxpool_wide = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))
        self.batchnorm64 = nn.BatchNorm2d(64)
        self.batchnorm128 = nn.BatchNorm2d(128)
        self.batchnorm256 = nn.BatchNorm2d(256)
        self.batchnorm1d = nn.BatchNorm1d(1024)

        self.enc_rnn = nn.LSTM(input_size=4096, hidden_size=rnn_size, num_layers=self.rnn_layers, bidirectional=False)
        self.dec_rnn = nn.LSTM(input_size=rnn_size, hidden_size=rnn_size, num_layers=self.rnn_layers, bidirectional=False) # output size?

        #
        # self.deconv1 = nn.ConvTranspose2d(h_dim, 128, (3, 3), padding=(1, 1), stride=(1, 1))
        # self.deconv2 = nn.ConvTranspose2d(128, 64, (3, 3), padding=(1, 1), stride=(1, 1))
        # self.deconv3 = nn.ConvTranspose2d(64, 32, (3, 3), padding=(1, 1), stride=(1, 1))
        # self.deconv4 = nn.ConvTranspose2d(32, 16, (3, 3), padding=(1, 1), stride=(1, 1))
        # self.deconv5 = nn.ConvTranspose2d(16, 1, (3, 9), padding=(1, 4), stride=(1, 1))
        # self.maxpool_square = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        # self.maxpool_wide = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))
        # self.batchnorm64 = nn.BatchNorm2d(64)
        # self.batchnorm128 = nn.BatchNorm2d(128)
        # self.batchnorm256 = nn.BatchNorm2d(256)
        # self.batchnorm1d = nn.BatchNorm1d(1024)

        self.fc1 = nn.Linear(rnn_size, h_dim)
        self.fc2 = nn.Linear(rnn_size, h_dim)
        self.fc_h = nn.Linear(h_dim, h_dim)
        self.fc_c = nn.Linear(h_dim, h_dim)
        self.fc_out = nn.Linear(h_dim, 16*256)

        self.fc_init = nn.Linear(h_dim, 4096*64)

        # self.encoder = nn.Sequential(
        #     nn.Conv2d(1, 16, kernel_size=8, stride=2),
        #     nn.ReLU(),
        #     nn.Conv2d(16, 32, kernel_size=3, stride=2),
        #     nn.ReLU(),
        #     nn.Conv2d(32, 64, kernel_size=3, stride=2),
        #     nn.ReLU(),
        #     nn.Conv2d(64, 128, kernel_size=3, stride=2),
        #     nn.ReLU(),
        #     nn.Conv2d(128, 256, kernel_size=3, stride=2),
        #     nn.ReLU(),
        #     Flatten()
        # )


        # self.decoder = nn.Sequential(
        #     # UnFlatten(h_dim),
        #     nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2),
        #     nn.ReLU(),
        #     nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2),
        #     nn.ReLU(),
        #     nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2),
        #     nn.ReLU(),
        #     nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2),
        #     nn.ReLU(),
        #     nn.ConvTranspose2d(16, 1, kernel_size=8, stride=2),
        #     nn.ReLU()
        # )


        self.deconv1 = nn.ConvTranspose2d(256, 128, kernel_size=(3, 3), stride=1, padding=(1, 1))
        self.deconv2 =    nn.ConvTranspose2d(128, 64, kernel_size=3, stride=(1,2), padding=(1, 1))
        self.deconv3 =    nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1)
        self.deconv4 =    nn.ConvTranspose2d(32, 16, kernel_size=(3, 3), stride=2, padding=(1, 1))
        self.deconv5 =    nn.ConvTranspose2d(16, 1, kernel_size=(3, 9), stride=2, padding=(1, 4))

        self.sig = nn.Sigmoid()

        if use_cuda is not None:
            self.use_cuda = use_cuda
        else:
            self.use_cuda = None

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        esp = Variable(torch.randn(*mu.size()))
        if self.use_cuda is not None:
            esp = esp.cuda(self.use_cuda)
            std = std.cuda(self.use_cuda)
            mu = mu.cuda(self.use_cuda)
        z = mu + std*esp
        return z

    def encode(self, x):
        # batch_size = len(x)
        batch_size = x.shape[0]
        x_batch = x
        if self.use_cuda is not None:
            x_batch = x_batch.cuda(self.use_cuda)

        # self.conv_sizes[4] = x.shape[0], 1, x.shape[2], x.shape[3]
        self.conv_sizes[4] = x.shape
        features = F.relu(self.conv1(x_batch))
        # features = self.maxpool_square(features)
        self.conv_sizes[3] = features.shape[0], 16, features.shape[2], features.shape[3]
        features = F.relu(self.conv2(features))
        self.conv_sizes[2] = features.shape[0], 32, features.shape[2], features.shape[3]
        features = F.relu(self.batchnorm64(self.conv3(features)))
        self.conv_sizes[1] = features.shape[0], 64, features.shape[2], features.shape[3]
        features = F.relu(self.batchnorm128(self.conv4(features)))
        features = F.relu(self.batchnorm256(self.conv5(features)))
        self.conv_sizes[0] = features.shape[0], 128, features.shape[2], features.shape[3]   # why 128?


        rnn_in = features.view(batch_size, 256 * features.data.shape[2], -1).transpose(0, 2).transpose(1, 2)
        output, (hidden, _) = self.enc_rnn(rnn_in)
        hidden = hidden.squeeze(0)
        mu, logvar = self.fc1(hidden), self.fc2(hidden)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar, features

    def decode(self, z, tsize):
        # channels_target, height_target, width_target = tsize
        # h, c = self.fc_h(z), self.fc_c(z)
        # h, c = h.unsqueeze(0), c.unsqueeze(0)
        # out = Variable(torch.zeros(1, z.shape[0], self.rnn_size))
        # seq_out = Variable(torch.zeros(z.shape[0], channels_target, height_target, width_target))
        # if self.use_cuda is not None:
        #     seq_out = seq_out.cuda(self.use_cuda)
        #     out = out.cuda(self.use_cuda)

        # for i in range(width_target):
        #     out, (h, c) = self.dec_rnn(out, (h, c))
        #     col = self.fc_out(out)
        #     # col = col > col.mean()
        #     # col = col.type(torch.FloatTensor)
        #     col = col.squeeze(0).unsqueeze(1)
        #     col = col.view(col.shape[0], 256, 16, -1)
        #
        #     if self.use_cuda is not None:
        #         out = out.cuda(self.use_cuda)
        #         col = col.cuda(self.use_cuda)
        #     # dec_in = out
        #     # out = out.squeeze(0).unsqueeze(1)
        #     seq_out[:, :, :, i] = col.squeeze(-1)
        # if self.use_cuda is not None:
        #     seq_out = seq_out.cuda(self.use_cuda)

        fmap = self.fc_init(z)
        fmap = fmap.view(self.batch_size, 256, 16, 64)

        # return self.decoder(seq_out)
        # b_size, c_size, height, width = seq_out.shape[0], seq_out.shape[1], seq_out.shape[2]*2, seq_out.shape[3]*2
        out = F.relu(self.deconv1(fmap, output_size=self.conv_sizes[0]))
        out = F.relu(self.deconv2(out, output_size=self.conv_sizes[1]))
        out = F.relu(self.deconv3(out, output_size=self.conv_sizes[2]))
        out = F.relu(self.deconv4(out, output_size=self.conv_sizes[3]))
        out = self.deconv5(out, output_size=self.conv_sizes[4])
        means = out.view(out.shape[0], -1).mean(1)
        for i in range(means.shape[0]):
            out[i, :, :, :] -= means[i].item()
        stds = out.view(out.shape[0], -1).std(1)
        for i in range(stds.shape[0]):
            out[i, :, :, :] /= stds[i].item()

        out = self.sig(out)

        # out = out.round()
        # out = out.type(torch.long)
        return out


    def forward(self, x):
        z, mu, logvar, features = self.encode(x)
        z = self.decode(z, (features.shape[1], features.shape[2], features.shape[3]))
        # z = (z > 0.).type(torch.FloatTensor)
        return z, mu, logvar

