import argparse
import os
import sys
import math
from copy import deepcopy
import numpy as np
import pretty_midi as pm


def write_scaled_midi(midfile, scale_arr, save_path):
    parsed = pm.PrettyMIDI(midfile)
    time_scale = parsed.get_end_time() / float(scale_arr.shape[0])
    mid = deepcopy(parsed)

    scale_arr = np.maximum(scale_arr, 0.05)
    scale_arr = np.minimum(scale_arr, 1.)
    for inst in mid.instruments:
        for note in inst.notes:
            note_timesteps = min(int(round(float(note.start) / time_scale)), scale_arr.shape[0] - 1)
            note.velocity = int(90*scale_arr[note_timesteps])
    mid.write(save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save', default='scaled_midi')
    parser.add_argument('--mid')
    parser.add_argument('--scale')
    args = parser.parse_args()

    midi_path = os.path.join(os.getcwd(), args.mid)
    scale_arr_path = os.path.join(os.getcwd(), args.scale)
    save_path = os.path.join(os.getcwd(), args.save)
    os.makedirs(save_path, exist_ok=True)

    raw_values = np.load(scale_arr_path)

    raw_values = np.concatenate([np.zeros(1), raw_values])

    normalized = raw_values - np.ones(raw_values.shape)*raw_values.min()
    normalized = normalized / np.max(normalized)
    normalized = normalized / 2. + np.ones(normalized.shape)*0.5

    write_scaled_midi(midi_path, normalized, save_path)