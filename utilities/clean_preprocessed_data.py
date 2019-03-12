import os, glob, argparse, shutil
import editdistance
from itertools import product
from shutil import copy

def main(opts):
    output_path = os.path.join(os.getcwd(), opts.clean_data_root)
    data_root = os.path.join(os.getcwd(), opts.data_root)

    to_remove = set()
    for cat_dir in glob.glob(os.path.join(data_root, '*')):
        if not os.path.isdir(cat_dir):
            continue
        phase_paths = {p: os.path.join(cat_dir, p) for p in ('train', 'val', 'test')}
        phase_names = {p: glob.glob(os.path.join(phase_paths[p], '*.npy')) for p in ('train', 'val', 'test')}

        for i, (phase1, phase2) in enumerate([('train', 'val'), ('train', 'test'), ('val', 'test')]):
            for (name1, name2) in product(phase_names[phase1], phase_names[phase2]):
                short_name1, short_name2 = (os.path.basename(n).split('.')[0].split('(c)')[0] for n in (name1, name2))
                ed = editdistance.eval(short_name1, short_name2)
                if ed < 1:
                    if phase2 == 'val':
                        print('\tremoving {} in favor of {}'.format(name2, name1))
                        to_remove.add(name2)
                    elif phase1 == 'train':
                        print('\tremoving {} in favor of {}'.format(name1, name2))
                        to_remove.add(name1)
                    else:
                        print('\tremoving {} in favor of {}'.format(name2, name1))
                        to_remove.add(name2)

    print('\ncopying files...')
    for cat_dir in glob.glob(os.path.join(data_root, '*')):
        if not os.path.isdir(cat_dir):
            continue
        for phase in ('train', 'val', 'test'):
            clean_path = os.path.join(output_path, os.path.basename(cat_dir), phase)
            os.makedirs(clean_path)
            for data_file in glob.glob(os.path.join(cat_dir, phase, '*.npy')):
                if data_file not in to_remove:
                    copy(data_file, clean_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_root')
    parser.add_argument('clean_data_root')
    args = parser.parse_args()
    main(args)