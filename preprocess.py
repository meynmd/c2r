import argparse
import os
import numpy as np
import music21
import librosa
from lib import pretty_midi as pmidi

def main(args):
    for (root, dirs, files) in list(os.walk( "./" + args.dir_in ))[1:]:
        if len(files) == 0:
            continue
        label_name = root.split("/")[-1]
        out_dir = "./" + args.dir_out + "/" + label_name
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        for fname in files:
            print("preprocessing {}".format(fname), end="\r")
            roll = preprocess(root + "/" + fname)
            np.save(out_dir + "/" + fname, roll)


def preprocess(file):
    pm = pmidi.PrettyMIDI(file)
    return pm.get_piano_roll()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir_in")
    parser.add_argument("--dir_out")
    args = parser.parse_args()
    main(args)