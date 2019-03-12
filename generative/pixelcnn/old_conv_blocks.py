import torch.nn as nn
from masked_conv import MaskedConv2d

# masked conv blocks as implemented by Jaemin Cho

class MaskBConvResidualBlock(nn.Module):
    def __init__(self, h=64, kernel_size=7, stride=1, padding=3, dilation=1):
        super(MaskBConvResidualBlock, self).__init__()
        self.net = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(2*h, h, 1),
            nn.BatchNorm2d(h),
            nn.ReLU(),
            nn.Conv2d(h, h, (kernel_size, 1), stride, (padding, 0), dilation=(dilation, 1)),
            nn.BatchNorm2d(h),
            nn.ReLU(),
            MaskedConv2d('B', h, h, (1, kernel_size), stride, (0, padding), dilation=(1, dilation)),
            nn.BatchNorm2d(h),
            nn.ReLU(),
            nn.Conv2d(h, 2*h, 1),
            nn.BatchNorm2d(2*h)
        )
        print('MaskBConvResidualBlock {}'.format(h))

    def forward(self, x):
        z = self.net(x)
        return z + x


# def mask_a_conv(*args, **kwargs):
#     if 'out_channels' in kwargs:
#         c_out = kwargs['out_channels']
#     elif len(args) > 1:
#         c_out = args[1]
#     else:
#         raise(Exception('number of output channels not provided for Mask A convolution'))
#     return nn.Sequential(
#         MaskedConv2d('A', *args, **kwargs),
#         nn.BatchNorm2d(c_out)
#     )

class MaskAConv(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size=7, stride=1, padding=3, dilation=1):
        super(MaskAConv, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, (kernel_size, 1), stride, (padding, 0), dilation=(dilation, 1)),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(),
            MaskedConv2d('A', ch_out, ch_out, (1, kernel_size), stride, (0, padding), dilation=(1, dilation)),
            nn.BatchNorm2d(ch_out),
        )
        print('MaskAConv {}, {}'.format(ch_in, ch_out))

    def forward(self, x):
        z = self.net(x)
        return z # + x
