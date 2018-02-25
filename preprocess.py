import argparse
import os
import numpy as np
import music21
import librosa
from lib import pretty_midi as pmidi

def preprocess(file, freq):
    pm = pmidi.PrettyMIDI(file)
    return pm.get_piano_roll(fs=freq)

def main(args):
    labels = set()
    for (root, dirs, files) in list(os.walk( "./" + args.dir_in ))[1:]:
        if len(files) == 0:
            continue
        phase_name = root.split("/")[-1]
        label_name = root.split("/")[-2]
        labels.add(label_name.replace("-", " "))
        out_dir = "./" + args.dir_out + "/" + label_name + "/" + phase_name
        print("out dir: " + out_dir)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        for fname in files:
            out_name = out_dir + "/" + fname
            if os._exists(out_name):
                print("{} already exists, skipping.".format(out_name))
            else:
                print("preprocessing {}".format(fname), end="\r")
                try:
                    roll = preprocess(root + "/" + fname, args.fs)
                    np.save(out_name, roll)
                except:
                    print("\nfailed to preprocess {}, skipping".format(fname))
    with open(args.dir_out + "/labels.csv", "w") as fp:
        c = 0
        for l in labels:
            fp.write("{},{}\n".format(l, c))
            c += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir_in")
    parser.add_argument("--dir_out")
    parser.add_argument("--fs", type=int, default=100)
    args = parser.parse_args()
    main(args)