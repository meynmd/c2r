import torch
import torch.nn as nn
from torch import cuda
import torch.legacy.nn as legacy

class ConvNetModel(nn.Module):
    def __init__(self):
        super(ConvNetModel, self).__init__()
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
        self.last_layer = nn.Sequential(
            legacy.Transpose((2, 3), (3, 4)),
            legacy.SplitTable(1)
        )
