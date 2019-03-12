import random
import argparse
import os
import numpy as np
import music21
import librosa
from lib import pretty_midi as pmidi
import mido

def preprocess(file, freq):
    mf = mido.MidiFile(file)
    fastest_tempo = min(m.tempo for m in mf if m.type == "set_tempo")
    tick_length = mido.tick2second(1, mf.ticks_per_beat, fastest_tempo)
    pm = pmidi.PrettyMIDI(file)
    return pm.get_piano_roll(fs=freq)

def main(args):
    labels = set()
    for (root, dirs, files) in list(os.walk( "./" + args.dir_in ))[1:]:
        if len(files) == 0:
            continue
        label_name = root.split("/")[-1]
        labels.add(label_name.replace("-", " "))
        idxs = list(range(len(files)))
        random.shuffle(idxs)
        out_dir = "./" + args.dir_out + "/" + label_name
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        num_val = args.val_percent * len(files) / 100.
        num_test = args.test_percent * len(files) / 100.
        for j, idx in enumerate(idxs):
            if j < num_val + num_test:
                if j < num_test:
                    phase = "test"
                else:
                    phase = "val"
            else:
                phase = "train"
            out_subdir = out_dir + "/" + phase
            if not os.path.exists(out_subdir):
                os.makedirs(out_subdir)
            out_name = out_subdir + "/" + files[idx]
            if os.path.exists(out_name):
                print("{} already exists, skipping.".format(out_name))
            else:
                print("preprocessing {}".format(files[idx]), end="\r")
                try:
                    roll = preprocess(root + "/" + files[idx], args.fs)
                    np.save(out_name, roll)
                except:
                    print("\nfailed to preprocess {}, skipping".format(out_name))
    with open(args.dir_out + "/labels.csv", "w") as fp:
        c = 0
        for l in labels:
            fp.write("{},{}\n".format(l, c))
            c += 1
    print("preprocessing complete\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir_in")
    parser.add_argument("--dir_out")
    parser.add_argument("--val_percent", type=int, default=10)
    parser.add_argument("--test_percent", type=int, default=10)
    parser.add_argument("--fs", type=int, default=100)
    args = parser.parse_args()
    main(args)