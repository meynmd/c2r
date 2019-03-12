import os, random
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate

class ComposerDataset(Dataset):
    # dataset representing music scores as 2d matrices (pitch x time)
    def __init__(self, data_root, label_dict_file, phase="train", seed=0, classname=None):
        random.seed(seed)
        self.tgt_classname = classname
        self.root = data_root
        self.xys = []
        self.x_counts = {}
        self.y2x = {}
        self.phase = phase

        with open(data_root + "/" + label_dict_file, "r") as fp:
            name_num = [line.strip().split(",") for line in fp.readlines()]
            self.name2idx = {n : int(i) for n, i in name_num}
            self.idx2name = {int(i) : n for n, i in name_num}

            data_dir = os.path.join(data_root, classname, phase)
            data_files = glob.glob(data_dir + "/*.npy")
            if data_files:
                self.xys = [(f, classname) for f in data_files]
                self.x_counts[classname.replace('-', ' ')] = len(data_files)
                self.y2x[classname.replace('-', ' ')] = data_files

    def __len__(self):
        return len(self.xys)

    def __getitem__(self, idx):
        x_path, y = self.xys[idx]
        x = np.load(x_path)
        # x = (x > 0.).astype(float)

        if self.phase == 'train':
            # transpose by random interval
            t = random.randint(0, 11)
            x = np.roll(x, t, axis=0)
            x[:t, :] = 0.

        x = torch.from_numpy(x).type(torch.float)
        y = y.replace("-", " ")
        y = self.name2idx[y]
        return x, y

    def get_idxs_from_class(self, class_label):
        pass

    def get_from_class(self, class_label):
        for x_path, y in self.y2x[class_label]:
            x = torch.from_numpy(np.load(x_path))
            x = x.float()
            y = y.replace("-", " ")
            yield (x, self.name2idx[y]), (x_path, y)

    def idx2onehot(self, idx, size):
        vec = torch.FloatTensor(size).fill_(0.)
        vec[idx] = 1.
        return vec

    def get_x_count(self, y):
        return self.x_counts[y]

    def get_y_count(self):
        return len(self.x_counts)

    def get_all_labels(self):
        return self.x_counts.keys()
