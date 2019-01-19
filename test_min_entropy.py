import argparse
import glob
import os, sys, math
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from torch.autograd import Variable
from torch import cuda

import encoder
import pr_dataset

def random_crop(tensor, w_new, h_new=None):
    h, w = tensor.shape[0], tensor.shape[1]
    top, left = 0, 0
    if w_new < w:
        left = np.random.randint(0, w - w_new)
    if h_new is None:
        return tensor[:, left : left + w_new]
    if h_new < h:
        top = np.random.randint(0, h - h_new)
    return tensor[top : top + h_new, left : left + w_new]


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


def test(model, data, num_per_class, cuda_dev, stride, tf_output_file):
    # overlap_frac = 0.5
    # stride = int(round((1. - overlap_frac)*model.max_w))

    correct_total, xy_count = 0, 0
    labels = data.idx2name.items()
    num_classes = len(data.get_all_labels())
    correct = torch.zeros(num_classes)
    confusion = torch.zeros((num_classes, num_classes))
    predict_confusion = torch.zeros((num_classes, num_classes))
    for i, label in labels:
        print('inferencing on class {}'.format(label))
        # get (x, y)'s from dataset object, select num_per_class of them
        class_datapoints = list(data.get_from_class(label))
        # random.shuffle(class_datapoints)
        num_d = min(num_per_class, len(class_datapoints))
        probs = torch.zeros(num_classes) # [0. for j in range(len(data.get_all_labels()))]
        preds = torch.zeros(num_classes)
        for (x, y), (composer, x_path) in class_datapoints[:num_d]:
            model.eval()
            if x.shape[1] > model.max_w:
                # x = random_crop(x, model.max_w)

                x_batches = crop_and_batch(x, model.max_w, stride, model.batch_size)
            else:
                x_batches = [x.unsqueeze(0).unsqueeze(0)]

            tf_per_batch = model.batch_size*stride

            best_ent, best_z, best_timeframe = float('inf'), None, None
            for j, batch in enumerate(x_batches):
                z = model(batch)
                # record softmax "probabilities" for each label
                h = -1 * (F.softmax(z, 1) * F.log_softmax(z, 1)).sum(1)
                h_min, idx_min = torch.min(h, 0)
                h_min, idx_min = h_min.item(), idx_min.item()
                if h_min < best_ent:
                    start_tf = j*tf_per_batch + idx_min*stride
                    best_ent, best_z, best_timeframe = h_min, z[idx_min, :], start_tf

            tf_output_file.write('{},{},{},{}\n'.format(composer, os.path.basename(x_path).split('.')[0], best_timeframe, best_timeframe + model.max_w))

            p_z = F.softmax(best_z, 0).cpu()
            probs = probs + p_z

            # make prediction
            _, y_hat = torch.max(p_z), torch.argmax(p_z)
            preds[y_hat.data.item()] += 1
            c = int(y_hat.data.item() == y)
            correct[i] += c
            correct_total += c
            xy_count += 1

        # column i shows probabilities that composer i is classified as...
        confusion[:, i] = probs / num_d
        correct[i] /= num_d
        predict_confusion[:, i] = preds / num_d

    print("*"*10, "Probabilities", "*"*10, "\n")
    print(" " * 11, end="")
    short_label = []
    for _, label in labels:
        l = min(8, len(label))
        short_label.append("{:>8}".format(label[:l]))
    for sl in short_label:
        print("{}".format(sl), end="   ")
    print()
    for j, sl in enumerate(short_label):
        print(sl, end="   ")
        for i in range(confusion.shape[1]):
            print("{:8.1%}".format(confusion[j][i]), end="   ")
        print()

    print("\n" + "*"*10, "Predictions", "*"*10, "\n")
    print(" " * 11, end="")
    short_label = []
    for _, label in labels:
        l = min(8, len(label))
        short_label.append("{:>8}".format(label[:l]))
    for sl in short_label:
        print("{}".format(sl), end="   ")
    print()
    for j, sl in enumerate(short_label):
        print(sl, end="   ")
        for i in range(predict_confusion.shape[1]):
            print("{:8.1%}".format(predict_confusion[j][i]), end="   ")
        print()
    print()
    for sl in short_label:
        print(sl, end="   ")
    print()
    for i in range(correct.shape[0]):
        print("{:8.2%}".format(correct[i]), end="   ")

    print("\n\nAverage accuracy: {:.3%}\n".format(float(correct_total) / xy_count))


def main(opts):
    if opts.use_cuda is not None:
        print("using CUDA device {}".format(opts.use_cuda), file=sys.stderr)
        cuda_dev = int(opts.use_cuda)
    else:
        cuda_dev = None

    # load the data
    dataset = pr_dataset.PianoRollDataset(os.getcwd() + "/" + opts.data_dir, "labels.csv", "test")

    # load the model
    enc = encoder.Encoder(
        dataset.get_y_count(),
        opts.batch_size,
        rnn_size=opts.rnn_size,
        num_rnn_layers=opts.rnn_layers,
        use_cuda=cuda_dev,
        max_w=opts.max_w
    )

    if opts.load:
        saved_state = torch.load(opts.load, map_location='cpu')
        enc.load_state_dict(saved_state)
    if cuda_dev is not None:
        enc = enc.cuda(cuda_dev)

    for param in enc.parameters():
        param.requires_grad = False

    timeframe_output_file = open(args.tf_file, 'w')
    timeframe_output_file.write('composer,piece,begin_tf,end_tf')

    test(enc, dataset, 1000, opts.use_cuda, opts.stride, timeframe_output_file)

    timeframe_output_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rnn_size", type=int, default=128)
    parser.add_argument("--rnn_layers", type=int, default=3)
    parser.add_argument("--data_dir", default="preprocessed")
    parser.add_argument("--max_epochs", type=int, default=1000)
    parser.add_argument("--max_w", type=int, default=5000)
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("-m", "--model_dir", default="model")
    parser.add_argument("-e", "--num_epochs", type=int, default=5)
    parser.add_argument("--tf_file", default="best_timeframes.csv")
    parser.add_argument("--num_batch_valid", type=int, default=1)
    parser.add_argument("--beam_size", type=int, default=5)
    parser.add_argument("-s", "--seed", type=int, default=0)
    parser.add_argument("-c", "--use_cuda", type=int, default=None)
    parser.add_argument("-l", "--init_lr", type=int, default=5)
    parser.add_argument("--load", default=None)
    parser.add_argument("--stride", type=int, default=100)

    args = parser.parse_args()
    main(args)