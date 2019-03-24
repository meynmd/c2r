'''
calculate negatives / positives for entire training set
might need this if compensating for this inherently unbalanced data using positive weights
(but focal loss seems to work much better anyway and doesn't require this)
'''
import os, argparse, glob
import numpy as np


def count_divide(data_dir):
    data_files = glob.glob(os.path.join(data_dir, '*.npy'))
    num_ones, num_elem = 0., 0.
    for filename in data_files:
        arr = np.load(filename)
        num_ones += np.sum(arr)
        num_elem += np.size(arr)
    zero_one_ratio = (num_elem - num_ones) / num_ones

    return zero_one_ratio


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('data_root')
    p.add_argument('save')
    args = p.parse_args()

    data_root = os.path.abspath(args.data_root)
    zero_one = {}
    for filename in os.listdir(data_root):
        p = os.path.join(data_root, filename)
        if os.path.isdir(p):
            classname = filename
            zero_one[classname] = count_divide(os.path.join(p, 'train'))

    with open(args.save, 'w') as fp:
        for cat, pnr in zero_one.items():
            line = '{},{}\n'.format(cat, pnr)
            fp.write(line)
            print(line, end='')
