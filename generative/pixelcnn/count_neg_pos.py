import argparse
import os
import sys

import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader

import pr_dataset


def find_neg_pos_ratio(dataloader):
    # batch is a list of 2d tensors
    num_ones, num_elem = 0, 0
    for (_, (batch, _)) in enumerate(dataloader):
        num_ones += sum([torch.sum(x) for x in batch])
        num_elem += sum([torch.numel(x) for x in batch])
    return (num_elem - num_ones)/num_ones


def main(opts):
    # initialize data loaders
    dataset = pr_dataset.PianoRollDataset(os.getcwd() + "/" + opts.data_dir, "labels.csv", 'train')
    dataloader = DataLoader(
            dataset,
            batch_size=64,
            shuffle=False,
            num_workers=0,
            collate_fn=lambda b : list(list(l) for l in zip(*b))
        )

    np = find_neg_pos_ratio(dataloader)
    print('all,{}'.format(np))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="../../clean_preproc")
    args = parser.parse_args()
    main(args)

