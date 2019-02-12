import argparse
import numpy as np
import pretty_midi
import lib.pretty_midi.examples.reverse_pianoroll as rev

p = argparse.ArgumentParser()
p.add_argument('piano_roll')
p.add_argument('output')
args = p.parse_args()

pr = np.load(args.piano_roll)
pr = 64.*pr
pm = rev.piano_roll_to_pretty_midi(pr)
pm.write(args.output)