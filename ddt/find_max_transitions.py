import argparse, glob, os, sys, math
import numpy as np
from collections import defaultdict
from copy import deepcopy

import pretty_midi as pm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from torch.autograd import Variable
from torch import cuda

from encoder import Encoder

from matplotlib import pyplot as plt

# def run_net(net, piano_roll):
#     pr = torch.from_numpy(piano_roll).type(torch.float)
#     pr = (pr > 0.).type(torch.float)
#     pr = Variable(pr)
#     x = pr.unsqueeze(0).unsqueeze(0)
#     net.eval()
#     return net(x)


def output_transition_times(piano_roll, midi_file, transition_idxs, seq_len):
    parsed = pm.PrettyMIDI(midi_file)
    time_scale = parsed.get_end_time() / float(seq_len)
    return [transition_idxs[i].item()*time_scale for i in range(transition_idxs.shape[0])], time_scale


def upsample_h(arr, factor):
    resampled = np.zeros((factor*arr.shape[0], arr.shape[1]))
    for i in range(arr.shape[0]):
        resampled[i*factor : (i+1)*factor] = np.stack([arr[i, :] for j in range(factor)])
    return resampled


def upsample_w(arr, factor):
    resampled = np.zeros((arr.shape[0], factor*arr.shape[1]))
    for i in range(arr.shape[1]):
        resampled[:, i*factor : (i+1)*factor] = np.stack([arr[:, i] for j in range(factor)], 1)
    return resampled


def main(opts):
    save_path = os.path.join(os.getcwd(), opts.save_dir)
    os.makedirs(save_path, exist_ok=True)

    if opts.use_cuda is not None:
        print("using CUDA device {}".format(opts.use_cuda), file=sys.stderr)
        cuda_dev = int(opts.use_cuda)
    else:
        cuda_dev = None
    sys.stdout.flush()

    net = Encoder(
        1,
        1,
        rnn_size=opts.rnn_size,
        num_rnn_layers=opts.rnn_layers,
        use_cuda=cuda_dev,
        max_w=opts.max_w
    )

    saved_state = torch.load(opts.load, map_location='cpu')
    net.load_state_dict(saved_state, strict=False)
    if opts.use_cuda is not None:
        net = net.cuda(cuda_dev)

    for p in net.parameters():
        p.requires_grad = False

    prfile, midfile = args.pr, args.mid
    pr = np.load(prfile)
    x = Variable(torch.from_numpy(pr).type(torch.float))
    if opts.use_cuda is not None:
        x = x.cuda(cuda_dev)

    z = net(x.unsqueeze(0).unsqueeze(0))
    z = z.squeeze(1).cpu().numpy()
    np.save('lstm_features_{}'.format(os.path.basename(midfile).split('.')[0]), z)

    """
    z, dot_prods, transition_points, transition_idxs = net.locate_transitions(x.unsqueeze(0).unsqueeze(0), avg_window=20, points_max=100)
    trans_times, time_scale = output_transition_times(pr, midfile, transition_idxs, dot_prods.shape[0])
    print('transition times:')
    for t in trans_times:
        print('{}:{}'.format(int(t // 60), int(round(t)) % 60))
    print('\nsorted:')
    last = 0.
    for t in sorted(trans_times):
        if True: # t - last > 1.:
            print('{}:{}'.format(int(t // 60), int(round(t)) % 60))
        last = t


    f = plt.figure()
    upsampled_dps = upsample_h(dot_prods.unsqueeze(0).cpu().numpy(), 10)
    upsampled_z = z.transpose(0, 1).cpu().numpy()
    concat = np.concatenate((upsampled_z, upsampled_dps), 0)
    last_idx = 0
    for i in range(transition_idxs.shape[0]):
        idx = transition_idxs[i].item()
        # if idx - last_idx > 2:
        t = trans_times[i]
        #plt.vlines(idx, 0, concat.shape[0], linestyles='dashed', colors='r', linewidth=1, label='{}:{}'.format(int(t // 60), int(round(t)) % 60))
        last_idx = idx
    plt.imshow(concat)
    # plt.show()
    f.savefig('dotprod_z.png')
    """

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pr")
    parser.add_argument("--mid")
    parser.add_argument("-c", "--use_cuda", type=int, default=None)
    parser.add_argument("--load", default=None)
    parser.add_argument("-s", "--save_dir")
    parser.add_argument("--max_w", type=int, default=2048)
    parser.add_argument("--rnn_size", type=int, default=512)
    parser.add_argument("--rnn_layers", type=int, default=1)

    args = parser.parse_args()
    main(args)