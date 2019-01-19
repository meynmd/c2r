import os, glob, argparse, random, math
import shutil
import numpy as np

def make_data(data_dir, output_dir, label_file='labels.csv'):
    input_files = glob.glob(os.path.join(data_dir, '*.npy'))
    label2file = {os.path.basename(f).split('.')[0].split('(')[0] : f for f in input_files}

    label_fp = open(os.path.join(output_dir, label_file), 'w')
    count = 0
    for (label, file) in label2file.items():
            label_fp.write('{},{}\n'.format(label, count))    # write category label
            count += 1

            class_path = os.path.join(output_dir, label)
            os.makedirs(class_path, exist_ok=True)
            os.makedirs(os.path.join(class_path, 'train'))
            os.makedirs(os.path.join(class_path, 'val'))


            copy_path = os.path.join(class_path, 'train', os.path.basename(file))
            shutil.copy(file, copy_path)

    label_fp.close()
    return label2file



def main(opts):
    data_path = os.path.join(os.getcwd(), opts.data)
    output_path = os.path.join(os.getcwd(), opts.out)
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    os.makedirs(output_path)

    make_data(data_path, output_path)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data')
    p.add_argument('--out')
    p.add_argument('--window_size', type=int, default=500)
    p.add_argument('--percent_val', type=int, default=40)
    p.add_argument('--percent_overlap', type=int, default=50)
    args = p.parse_args()
    main(args)
