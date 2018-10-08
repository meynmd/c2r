import os, glob, argparse, random, math
import shutil
import numpy as np


def available_regions(arr_len, reserved_idxs, chunk_length):
    start_idx = 0
    bounds = []
    for idx in sorted(reserved_idxs):

        if idx >= start_idx + 1:
            bounds.append((start_idx, idx))

            print('add region ({}, {})'.format(start_idx, idx))

        else:

            print('skipping region ({}, {}) because it is to short'.format(start_idx, idx))

        start_idx = idx + 1

    if start_idx + 1 <= arr_len:
        bounds.append((start_idx, arr_len // chunk_length))

    print()

    return bounds


def make_data(data_dir, output_dir, chunk_length, min_num_chunks=20, frac_val=0.2, max_overlap=0.8, label_file='labels.csv'):
    input_files = glob.glob(os.path.join(data_dir, '*.npy'))
    label2file = {os.path.basename(f).split('.')[0].split('(')[0] : f for f in input_files}

    label_fp = open(os.path.join(output_dir, label_file), 'w')
    count = 0
    for (label, file) in label2file.items():
        mat = np.load(file)
        mat = (mat > 0.).astype(float)
        h, w = mat.shape
        num_chunks = w // chunk_length
        if num_chunks < min_num_chunks:
            print('warning: {} is too short to make {} chunks; skipping'.format(file, min_num_chunks))
        else:
            label_fp.write('{},{}\n'.format(label, count))    # write category label
            count += 1

            class_path = os.path.join(output_dir, label)
            os.makedirs(class_path, exist_ok=True)
            os.makedirs(os.path.join(class_path, 'train'))
            os.makedirs(os.path.join(class_path, 'val'))

            # choose and write val set samples
            # extra_data_length = w - num_chunks * chunk_length
            # tf_offset = extra_data_length // 2
            tf_offset = 0
            num_val = int(frac_val * float(num_chunks))

            print('total size: {}\nnum_chunks: {}\nnum_val: {}'.format(w, num_chunks, num_val))

            val_idxs = frozenset(random.sample(range(num_chunks), k=num_val))

            print('val_idxs: {}'.format(sorted(list(val_idxs))))

            for j, idx in enumerate(val_idxs):
                lb, rb = j*chunk_length + tf_offset, (j + 1)*chunk_length + tf_offset
                x = mat[:, lb : rb]
                filename = os.path.join(class_path, 'val', '{}_{}-{}'.format(label, lb, rb))
                np.save(filename, x)

            # write train set regions from the rest of the data
            train_regions = available_regions(w, val_idxs, chunk_length)
            total_windows = 0
            for (start, stop) in train_regions:
                region_length = stop - start
                num_windows = math.ceil((region_length*chunk_length - chunk_length) / (chunk_length - chunk_length*max_overlap)) + 1
                num_windows = max(num_windows, 0)
                total_windows += num_windows

                print('\nregion ({}, {})\tregion size {}\nstride {}\nnum windows {}\nwindows:'.format(start*chunk_length, stop*chunk_length, chunk_length*(stop - start), chunk_length*max_overlap, num_windows))

                for j in range(num_windows):
                    window_start, window_stop = int(start*chunk_length + j*chunk_length*(1.-max_overlap)), int(start*chunk_length + j*chunk_length*(1. - max_overlap)) + chunk_length
                    window = mat[:, window_start : window_stop]

                    print((window_start, window_stop), end='\t')

                    filename = os.path.join(class_path, 'train', '{}_{}-{}'.format(label, window_start, window_stop))
                    np.save(filename, window)
                print()

            print('num training: {}\tnum val: {}\n'.format(total_windows, num_val))

    label_fp.close()
    return label2file



def main(opts):
    data_path = os.path.join(os.getcwd(), opts.data)
    output_path = os.path.join(os.getcwd(), opts.out)
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    os.makedirs(output_path)

    frac_val = float(opts.percent_val) / 100.
    train_win_overlap = float(opts.percent_overlap) / 100.

    make_data(data_path, output_path, opts.window_size, frac_val=frac_val, max_overlap=train_win_overlap)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data')
    p.add_argument('--out')
    p.add_argument('--window_size', type=int, default=500)
    p.add_argument('--percent_val', type=int, default=40)
    p.add_argument('--percent_overlap', type=int, default=50)
    args = p.parse_args()
    main(args)
