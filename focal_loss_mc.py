import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.):
        super(FocalLoss, self).__init__()
        self.alpha, self.gamma = alpha, gamma

    def forward(self, z, y):
        batch_size, num_cat = z.shape
        unsqueezed = y.unsqueeze(1)
        y_one_hot = torch.zeros(batch_size, num_cat)
        y_one_hot.scatter_(1, unsqueezed, 1.)

        p = F.softmax(z, dim=1)

