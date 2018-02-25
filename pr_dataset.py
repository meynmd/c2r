import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate

class DataList(list):
    pass

class PianoRollDataset(Dataset):
    # dataset representing music scores as 2d matrices (pitch x time)
    def __init__(self, data_root, label_dict_file, phase="train"):
        self.root = data_root
        self.xy_s = []
        self.x_counts = {}
        with open(data_root + "/" + label_dict_file, "r") as fp:
            name_num = [line.strip().split(",") for line in fp.readlines()]
            self.name2idx = {n : int(i) for n, i in name_num}
            self.idx2name = {int(i) : n for n, i in name_num}
        for d in os.listdir(data_root):
            filenames = glob.glob(data_root + "/" + d + "/" + phase + "/*.npy")
            if filenames:
                class_list = [(f, d) for f in filenames]
                self.xy_s += class_list
                self.x_counts[self.name2idx[d.replace("-", " ")]] = len(class_list)


    def __len__(self):
        return len(self.xy_s)

    def __getitem__(self, idx):
        x_path, y = self.xy_s[idx]
        x = torch.from_numpy(np.load(x_path))
        x = x.float()
        y = y.replace("-", " ")
        y = self.name2idx[y]
        return x, y

    def idx2onehot(self, idx, size):
        vec = torch.FloatTensor(size).fill_(0.)
        vec[idx] = 1.
        return vec

    def get_x_count(self, y):
        return self.x_counts[y]

