import os, glob, argparse, operator
from functools import reduce
import numpy as np

from pretty_midi import PrettyMIDI

def preprocess(file, freq):
    pm = PrettyMIDI(file)
    pr = pm.get_piano_roll(fs=freq)
    pr = (pr > 0.).astype(float)
    return pr


def main(args):
    dir_in, dir_out = (os.path.join(os.getcwd(), d) for d in (args.dir_in, args.dir_out))
    if not os.path.exists(dir_out):
        os.makedirs(dir_out)

    patterns = (os.path.join(dir_in, p) for p in ('*.mid', '*.MID'))

    for data_file in reduce(operator.add, [glob.glob(p) for p in patterns]):
        out_path = os.path.join(dir_out, os.path.splitext(os.path.basename(data_file))[0])
        if os.path.exists(out_path):
            print("{} already exists, skipping.".format(out_path))
        else:
            print("preprocessing {}".format(data_file))
            if True:
                roll = preprocess(data_file, args.fs)
                np.save(out_path, roll)
            # except:
            #     print("failed to preprocess {}, skipping".format(data_file))

    print("preprocessing complete\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir_in")
    parser.add_argument("--dir_out")
    parser.add_argument("--fs", type=int, default=100)
    args = parser.parse_args()
    main(args)
