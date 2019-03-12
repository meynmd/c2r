import torch.nn as nn
from conv_blocks import MaskBConvResidualBlock, MaskAConv # mask_a_conv

class PixelCNN(nn.Module):
    def __init__(self, in_channels=1, h_channels=64, out_h_channels=1024):
        super(PixelCNN, self).__init__()
        self.a_conv = MaskAConv(in_channels, 2*h_channels, 7, stride=1, padding=6, dilation=2)
        self.b_conv = nn.Sequential(
            *[MaskBConvResidualBlock(h_channels, kernel_size=3, stride=1, padding=2, dilation=2) for _ in range(9)]
        )
        self.out_block = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(2*h_channels, out_h_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_h_channels),
            nn.ReLU(),
            nn.Conv2d(out_h_channels, in_channels, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, x):
        batch_size, in_channels, h, w = x.shape
        x = self.a_conv(x)
        x = self.b_conv(x)
        x = self.out_block(x)
        # x = x.view(batch_size, in_channels, self.discrete_channel, h, w)
        return x    #.permute(0, 1, 3, 4, 2)