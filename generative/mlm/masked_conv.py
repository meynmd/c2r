import torch
import torch.nn.functional as F

# masked convolutional network, as implemented by jzbontar

class MaskedConv2d(torch.nn.Conv2d):
    def __init__(self, mask, *args, **kwargs):
        print('\t\t\tmasked conv {}: {} => {} channels'.format(args[2], args[0], args[1]))

        super(MaskedConv2d, self).__init__(*args, **kwargs)
        assert(mask in ('A', 'B'))
        self.mask_type = mask
        self.register_buffer('mask', self.weight.data.clone())
        _, _, kern_h, kern_w = self.weight.shape
        self.mask.fill_(1)
        self.mask[:, :, kern_h // 2 + (mask == 'B'):, kern_w // 2] = 0
        self.mask[:, :, :, kern_w // 2 + 1:] = 0
        self.something = None

    def forward(self, input):
        self.weight.data *= self.mask
        z = super(MaskedConv2d, self).forward(input)
        return z
