import torch
import torch.nn as nn
from torch import cuda
import torch.legacy.nn as legacy

class LSTM(nn.Module):

    def __init__(self, model, input_feed, num_hidden, num_layers, batch_size,
                 max_encoder_l, dropout, use_attention, use_lookup, vocab_size ):

        dropout = dropout or 0
        inputs = {}
        inputs[]

