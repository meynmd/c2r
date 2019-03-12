# this version seems to learn poorly

import torch.nn as nn
from convblock import ConvBlock

class Encoder(nn.Module):
    def __init__(self, batch_size, use_cuda=None, max_w=None):
        super(Encoder, self).__init__()
        self.name = 'encoder cnn v2'
        self.max_w = max_w
        self.batch_size = batch_size
        self.cuda_dev = use_cuda

        self.conv_layers = nn.Sequential(
            ConvBlock('A', 1, 8, 8, (9, 3), stride=(1, 1), dilation=(1, 1)),        # conv1
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),

            ConvBlock('B', 8, 16, 16, (7, 3), stride=(1, 1), dilation=(1, 1)),      # conv2
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),

            ConvBlock('B', 16, 32, 32, (5, 3), stride=(1, 1), dilation=(1, 1)),     # conv3
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),

            ConvBlock('B', 32, 64, 64, (3, 3), stride=(1, 1), dilation=(1, 1)),  # conv3b
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),

            ConvBlock('B', 64, 128, 128, (3, 3), stride=(1, 1), dilation=(2, 2)),     # conv4

            ConvBlock('B', 128, 128, 256, (3, 3), stride=(1, 1), dilation=(4, 4)),    # conv5

            ConvBlock('B', 256, 256, 512, (3, 3), stride=(1, 1), dilation=(8, 8)),  # conv6

            ConvBlock('B', 512, 512, 512, (3, 3), stride=(1, 1), dilation=(16, 16)) # conv7

        )

    def forward(self, x_batch):
        if (self.cuda_dev is not None) and (not x_batch.is_cuda):
            x_batch = x_batch.cuda(self.cuda_dev)

        z = self.conv_layers(x_batch)
        return z


class Decoder(nn.Module):
    def __init__(self, batch_size, use_cuda=None, max_w=None):
        super(Decoder, self).__init__()
        self.name = 'decoder cnn v2'
        self.max_w = max_w
        self.batch_size = batch_size
        self.cuda_dev = use_cuda

        self.conv_layers = nn.Sequential(
            nn.ConvTranspose2d(512, 256, (1, 3), stride=(1, 2), padding=(0, 1), output_padding=(0, 1)),
            nn.BatchNorm2d(256), nn.ReLU(),
            nn.ConvTranspose2d(256, 128, (1, 3), stride=(1, 2), padding=(0, 1), output_padding=(0, 1)),
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, (1, 3), stride=(1, 2), padding=(0, 1), output_padding=(0, 1)),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, (1, 3), stride=(1, 2), padding=(0, 1), output_padding=(0, 1)),
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 1, 1)
        )

    def forward(self, x):
        return self.conv_layers(x)


class AutoEncoder(nn.Module):
    def __init__(self, *args, **kwargs):
        super(AutoEncoder, self).__init__()
        self.name = 'cnn autoencoder'
        self.encoder = Encoder(*args, **kwargs)
        self.decoder = Decoder(*args, **kwargs)

    def forward(self, x):
        z = self.encoder(x)
        z = self.decoder(z)
        return z
