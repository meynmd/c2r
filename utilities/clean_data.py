import os, glob, argparse, shutil


def main(opts):
    output_path = os.path.join(os.getcwd(), opts.clean_data_root)
    data_root = os.path.join(os.getcwd(), opts.data_root)
    for cat_dir in glob.glob(os.path.join(data_root, '*')):
        base_names = set()
        clean_cat_path = os.path.join(output_path, os.path.basename(cat_dir))
        os.makedirs(clean_cat_path)
        print('reading files from {}'.format(cat_dir))
        x_names = glob.glob(os.path.join(cat_dir, '*.mid'))
        for filename in x_names:
            base_name = os.path.basename(filename).split('(')[0]
            if base_name not in base_names:
                shutil.copy(filename, clean_cat_path)
                base_names.add(base_name)
            else:
                print('warning: file name {} has {} in common with at least another file; skipping'.format(filename, base_name))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_root')
    parser.add_argument('clean_data_root')
    args = parser.parse_args()
    main(args)