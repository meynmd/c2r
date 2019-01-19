import glob
import sys
import os

data_dir = os.path.join(os.getcwd(), sys.argv[1])
os.chdir(data_dir)
input_files = glob.glob('*.mid.npy')

y2idx, idx2y = {}, {}

for file in input_files:
    name = file.split('/')[-1].split('.')[0]
    if name not in y2idx:
        idx = len(y2idx)
        y2idx[name] = idx
        idx2y[idx] = name

contents = '\n'.join(['{},{}'.format(x, n) for x, n in idx2y.items()])
with open('labels.csv', 'w') as fp:
    fp.write(contents)
