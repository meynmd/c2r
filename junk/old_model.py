import torch.nn as nn
from junk import cnn


class Model(nn.Module):
    def __init__(self, optim):
        super(Model, self).__init__()
        self.optim = optim

    def create(self, phase="train", cnn_feature_size = 512, batch_size=64, valid_batch_size=64, input_feed=True,
               encoder_num_hidden=256, encoder_num_layers=3, target_embedding_size=128, targetVocabSize=128,
               maxEncoderLW=1000, maxEncoderLH=100, maxDecoderLength=1000, max_image_w=50000, max_image_h=256
    ):
        if phase == "train":
            train = True
        self.cnn_feature_size, self.batch_size, self.valid_batch_size = \
            cnn_feature_size, batch_size, valid_batch_size
        self.input_feed, self.encoder_num_hidden, self.encoder_num_layers = \
            input_feed, encoder_num_hidden, encoder_num_layers
        self.target_embedding_size, self.targetVocabSize, self.maxEncoderLW, self.maxEncoderLH = \
            target_embedding_size, targetVocabSize, maxEncoderLW, maxEncoderLH
        self.max_image_w, self.max_image_h = max_image_w, max_image_h

        self.mod_posEmbeddingFw = nn.Embedding(maxEncoderLH, 2 * encoder_num_layers * encoder_num_hidden)
        self.mod_posEmbeddingBw = nn.Embedding(maxEncoderLH, 2 * encoder_num_layers * encoder_num_hidden)
        self.mod_cnn = cnn.ConvNetModel()


