import argparse, glob, os, sys
import math
import time
import random
import numpy as np
from collections import defaultdict
from copy import deepcopy

import subprocess

import pretty_midi as pm

from sklearn.decomposition import PCA

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from torch.autograd import Variable
from torch import cuda

from encoder import Encoder
from pr_dataset import PianoRollDataset

from matplotlib import pyplot as plt

def random_crop(tensor, w_new, h_new=None):
    h, w = tensor.shape[2], tensor.shape[3]
    top, left = 0, 0
    if w_new < w:
        left = np.random.randint(0, w - w_new)
    if h_new is None:
        return tensor[:, :, :, left: left + w_new]
    if h_new < h:
        top = np.random.randint(0, h - h_new)
    return tensor[top: top + h_new, left: left + w_new]


def make_batch(tensors):
    max_w = 3000
    max_w = min(max_w, min(t.shape[1] for t, _ in tensors))

    x_batch = []
    for tensor, _ in tensors:
        tensor = (tensor > 0.).type(torch.float)
        tensor = tensor.view(1, 1, tensor.shape[0], tensor.shape[1])
        if tensor.shape[3] > max_w:
            tensor = random_crop(tensor, max_w)
        # elif tensor.shape[3] < max_w:
        #     num_padding = max_w - tensor.shape[3]
        #     tensor = torch.nn.functional.pad(tensor, (0, num_padding))
        x_batch.append(tensor)
    return Variable(torch.cat(x_batch, 0))


def ddt(net, x, midfile, mean_vec, trans_vecs, window_size=1024):
    cuda_dev = net.use_cuda
    net.eval()

    original_dims = x.shape
    x = x.unsqueeze(0).unsqueeze(0)
    z = net(x).cpu().data
    # z = z.squeeze(0).transpose(0, 2).transpose(0, 1)
    z = z.view(-1, net.rnn_size)
    z = z - mean_vec.repeat(z.shape[0], 1)

    z_sum = torch.zeros(z.shape[0], trans_vecs.shape[0])
    dotprods = torch.zeros(z.shape[0], z.shape[1], trans_vecs.shape[0])
    for p in range(trans_vecs.shape[0]):
        for i in range(z.shape[0]):
            z_sum[i, p] = torch.dot(z[i, :] / z[i, :].norm(p=2), trans_vecs[p])

    # z_positive = (z_sum > 0.).type(torch.float)*z_sum
    # positive_counts = torch.zeros(z_sum.shape[-1])
    # for i in range(z_sum.shape[-1]):
    #     positive_counts[i] = z_positive[:, i].nonzero().shape[0]
    #
    # avg_z_positive = z_positive.view(-1, z_positive.shape[-1]).sum(0) / positive_counts
    # z_positive = (z_positive > avg_z_positive).type(torch.float) * z_positive
    # z_scale = torch.FloatTensor([2**(-i) for i in range(z_positive.shape[-1])]).unsqueeze(0)
    # z_positive *= z_scale
    # z_weighted_sum = z_positive.sum(-1)

    # np.save('zsum', z_sum)
    # np.save('z_weighted_sum', z_weighted_sum)

    return z_sum #[:, :, 0]

# represent the dot products by scaling velocity
def write_scaled_midi(midfile, dot_prods, save_path):
    parsed = pm.PrettyMIDI(midfile)
    time_scale = parsed.get_end_time() / float(dot_prods.shape[0])
    # pitch_scale = 128 / dot_prods.shape[0]

    for pc in range(dot_prods.shape[-1]):
        mid = deepcopy(parsed)

        scale_map = dot_prods[:, pc].cpu().numpy()
        scale_map = np.maximum(scale_map, 0.05)
        scale_map = np.minimum(scale_map, 1.)
        for inst in mid.instruments:
            for note in inst.notes:
                note_timesteps = min(int(round(float(note.start) / time_scale)), scale_map.shape[0] - 1)
                # note_pitchstep = int(note.pitch // pitch_scale)
                note.velocity = int(90*scale_map[note_timesteps])
        save_filename = '{}-pc{}-scaled_vel.mid'.format(os.path.basename(midfile).split('.')[0],  pc + 1)
        mid.write(os.path.join(save_path, save_filename))


def write_scaled_piano_roll(piano_roll_file, dot_prods, save_path):
    piano_roll = np.load(piano_roll_file)
    pr_height, pr_length = piano_roll.shape
    dp_height, dp_length = dot_prods.shape[0], dot_prods.shape[1]
    pitch_scale, time_scale = pr_height / dp_height, pr_length / dp_length

    for pc in range(dot_prods.shape[-1]):
        velocity_scaled_pr = deepcopy(piano_roll)
        scale_map = dot_prods[:, pc].cpu().numpy()
        scale_map = np.maximum(scale_map, 0.05)
        scale_map = np.minimum(scale_map, 1.)
        for j in range(dp_length):
                start_j, stop_j = (int(round(v)) for v in (j*time_scale, (j+1)*time_scale))
                velocity_scaled_pr[:, max(0, start_j):min(piano_roll.shape[1], stop_j)] *= dot_prods[j, pc]

        np.save('pr-{}-pc{}-scaled'.format(os.path.basename(piano_roll_file).split('.')[0], pc + 1), velocity_scaled_pr)

        upsampled = upsample_h(velocity_scaled_pr, 10)
        f = plt.figure()
        plt.imshow(upsampled)
        # plt.show()
        vis_name = 'vis-pr-{}-pc{}-scaled.png'.format(os.path.basename(piano_roll_file).split('.')[0], pc)
        f.savefig(os.path.join(save_path, vis_name))
        print('piano roll for pc {} saved.'.format(pc + 1))



def find_descriptors_and_mean_vector(net, dataset, classname):
    # cuda_dev = net.use_cuda
    net.eval()
    data = dataset.get_from_class(classname)

    # batch_size = net.batch_size
    descriptor_dim = net.rnn_size

    descriptors, means = [], []
    for j, ((x, label), _) in enumerate(data):
        x = x.unsqueeze(0).unsqueeze(0)

        # z <- [t, b, d]
        z = net(x)

        # z <- [t*b, d]
        z = z.view(-1, descriptor_dim)

        means.append(z.mean(0).cpu())
        descriptors.append(z.cpu())

    return torch.stack(means).mean(0), torch.cat(descriptors, dim=0)


def find_pc(descriptors, mean_vec, num_pc):
    normalized = descriptors - torch.stack([mean_vec for i in range(descriptors.shape[0])])
    # pca = PCA(n_components=0.2, svd_solver='full')
    pca = PCA(n_components=5)
    pca.fit(normalized.cpu().data.numpy())
    pcs, var = pca.components_, pca.explained_variance_

    # return torch.FloatTensor(pcs[0:num_pc, :])
    return torch.FloatTensor(pcs)


def upsample_h(arr, factor):
    resampled = np.zeros((factor*arr.shape[0], arr.shape[1]))
    for i in range(arr.shape[0]):
        resampled[i*factor : (i+1)*factor] = np.stack([arr[i, :] for j in range(factor)])
    return resampled


def make_batches(dataset, class_name, max_w, max_batch, overlap=0.25):
    batches = []
    for i, (x, _) in enumerate(dataset.get_from_class(class_name)):
        x = x[0]
        if x.shape[1] <= max_w:
            batches.append(x.unsqueeze(0))
        else:
            batch = []
            start = 0
            while start + max_w < x.shape[1]:
                this_sample = x[:, start : start + max_w]
                assert(this_sample.shape[1] == max_w)
                batch.append(this_sample)
                start += int(math.floor((1. - overlap)*max_w))
            batch.append(x[:, x.shape[1] - max_w : x.shape[1]])
            assert(batch[-1].shape[1] == max_w)

            if len(batch) > max_batch:
                num_batches = len(batch) // max_batch
                for j in range(num_batches + 1):
                    this_batch = batch[j*max_batch : min((j+1)*max_batch, len(batch))]
                    if len(this_batch) > 0:
                        batches.append(torch.stack(this_batch))
            else:
                batches = [torch.stack(batch)]

    return batches


def crop_and_batch(orig_tensor, window_len, stride, max_batch):
    assert(orig_tensor.shape[1] >= window_len)
    cropped_tensors, start, stop = [], 0, window_len
    while stop < orig_tensor.shape[1]:
        cropped_tensors.append(orig_tensor[:, start : stop].unsqueeze(0))
        start += stride
        stop += stride

    batches = []
    if len(cropped_tensors) > max_batch:
        num_batches = int(math.ceil(len(cropped_tensors) / max_batch))
        for i in range(num_batches):
            b = cropped_tensors[i*max_batch : min((i+1)*max_batch, len(cropped_tensors))]
            batches.append(torch.stack(b))
    else:
        batches = [torch.stack(cropped_tensors)]

    return batches


def find_plateau(series, tolerance, k=1):
    vals_peak, idxs_peak = torch.topk(series, k, dim=0)
    for peak, t_peak in zip(vals_peak, idxs_peak):
        thr_min = peak - tolerance
        r_min, l_max = find_min_max(series, t_peak, )
        # l_min, r_max =


def walk(series, peak_idx, window, threshold, step=1):
    cur_idx, prev_idx, cur_val, prev_val = peak_idx, peak_idx, series[peak_idx], series[peak_idx]
    for i in range(window):
        cur_idx = peak_idx + step*i
        if 0 >= cur_idx or cur_idx >= series.shape[0] - 1:
            return cur_idx
        cur_val = series[cur_idx]
        if prev_val - cur_val > threshold:
            return cur_idx

    while 0 < cur_idx and cur_idx < series.shape[0] - 1:
        cur_val = series[cur_idx]
        prev_val = series[prev_idx]
        if prev_val - cur_val > threshold:
            return cur_idx
        cur_idx += step
        prev_idx += step

    return cur_idx


def find_peak_regions(column, regions_max=5, window_threshold=3):
    vals_peak, idxs_peak = torch.topk(column, regions_max, dim=0)
    thr = 0.5*np.std(column.numpy())
    regions = set()
    for idx in idxs_peak:
        idx = idx.item()
        start, stop = (walk(column, idx, 3, thr, direction) for direction in [-1, 1])
        regions.add((start, stop))
    return list(regions)


def main(opts):
    save_path = os.path.join(os.getcwd(), opts.save_dir)
    os.makedirs(save_path, exist_ok=True)

    if opts.use_cuda is not None:
        print("using CUDA device {}".format(opts.use_cuda), file=sys.stderr)
        cuda_dev = int(opts.use_cuda)
    else:
        cuda_dev = None
    sys.stdout.flush()

    dataset = PianoRollDataset(os.getcwd() + "/" + opts.data_dir, "labels.csv", "train")

    # load the model
    net = Encoder(
        dataset.get_y_count(),
        opts.batch_size,
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

    dotprods, avg_dps = {}, {}
    for class_name in dataset.get_all_labels():
        # batches = make_batches(dataset, class_name, opts.max_w, opts.batch_size)

        mvec, dscr = find_descriptors_and_mean_vector(net, dataset, class_name)
        pcs = find_pc(dscr, mvec, num_pc=opts.num_pc)

        np.save(os.path.join(save_path, 'pc-{}'.format(class_name)), pcs.numpy())

        filename, midfile = args.pr, args.mid
        pr = np.load(filename)
        pr = torch.from_numpy(pr).type(torch.float)
        pr = (pr > 0.).type(torch.float)
        pr = Variable(pr)

        dotprods[class_name] = dp = ddt(net, pr, midfile, mvec, pcs, opts.max_w)
        avg_dps[class_name] = dp.mean(-1)

    other_dps = [dotprods[c].unsqueeze(0) for c in dotprods if c != opts.classname]
    other_avg = torch.stack(other_dps, 0).mean(0).squeeze(0)
    delta = dotprods[opts.classname] - other_avg


    # zero out any negative results
    # normalized = (delta > 0.).type(torch.float) * delta
    # alternatively, subtract the min
    normalized = delta - torch.ones(delta.shape)*delta.view(-1, delta.shape[-1]).min()

    # write out theme regions
    theme_region_path = os.path.join(save_path, 'theme')
    os.makedirs(theme_region_path, exist_ok=True)
    parsed = pm.PrettyMIDI(args.mid)
    time_scale = parsed.get_end_time() / float(normalized.shape[0])
    for i in range(normalized.shape[1]):
        peak_regions = find_peak_regions(normalized[:, i])
        output_file = os.path.join(
            theme_region_path,
            'theme-regions-{}-pc{}.csv'.format(os.path.basename(args.mid).split('.')[0], i + 1)
        )
        # print('peak regions for pc #{}'.format(i + 1))
        with open(output_file, 'w') as fp:
            for (start, stop) in peak_regions:
                if stop - start < 5:
                    continue
                else:
                    fp.write('{},{}\n'.format(int(round(time_scale*start)), int(round(time_scale*stop))))
                    # print('\t({}, {})'.format(start, stop))

    # if number of components is automatic
    # other_dps = [avg_dps[c].unsqueeze(0) for c in avg_dps if c != opts.classname]
    # other_avg = torch.stack(other_dps, 0).mean(0).squeeze(0)
    # delta = dotprods[class_name] - torch.stack([other_avg for _ in range(dotprods[class_name].shape[-1])], dim=1)

    # normalized = dotprods[class_name]
    # normalized = (delta > 0.).type(torch.float) * delta
    # normalized = normalized / normalized.view(-1, normalized.shape[-1]).max()

    # z_scale = torch.FloatTensor([2**(-i) for i in range(normalized.shape[-1])]).unsqueeze(0)
    # combined = (normalized*z_scale).sum(1)
    # combined = combined / combined.view(-1).max()
    # combined = combined.unsqueeze(-1)

    # write_scaled_midi(args.mid, combined, save_path)
    # write_scaled_midi(args.mid, normalized, save_path)

    # write_scaled_piano_roll(args.pr, normalized, save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="pprocessed-full")
    parser.add_argument("--pr")
    parser.add_argument("--mid")
    parser.add_argument("--classname")
    parser.add_argument("-c", "--use_cuda", type=int, default=None)
    parser.add_argument("--load", default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("-p","--num_pc", type=int, default=1)
    parser.add_argument("-s", "--save_dir")
    parser.add_argument("--max_w", type=int, default=2048)
    parser.add_argument("--rnn_size", type=int, default=512)
    parser.add_argument("--rnn_layers", type=int, default=1)

    args = parser.parse_args()
    main(args)