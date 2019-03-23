import argparse, glob, os, sys, math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from torch.autograd import Variable
from torch import cuda

import cnn
import pr_dataset
import torch.nn.functional as F

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


def test(model, data, num_per_class, max_w, stride, batch_size, cuda_dev): #, tf_output_file):
    correct_total, xy_count = 0, 0
    labels = data.idx2name.items()
    num_classes = len(data.get_all_labels())
    correct = torch.zeros(num_classes)
    confusion = torch.zeros((num_classes, num_classes))
    predict_confusion = torch.zeros((num_classes, num_classes))
    for i, label in labels:
        print('inferencing on class {}'.format(label))

        class_datapoints = list(data.get_from_class(label))
        num_d = min(num_per_class, len(class_datapoints))

        probs = torch.zeros(num_classes) # [0. for j in range(len(data.get_all_labels()))]
        preds = torch.zeros(num_classes)
        if cuda_dev is not None:
            probs, preds = probs.cuda(cuda_dev), preds.cuda(cuda_dev)

        for (x, y), (composer, x_path) in class_datapoints[:num_d]:
            model.eval()
            if x.shape[1] > max_w:
                x_batches = crop_and_batch(x, max_w, stride, batch_size)
            else:
                x_batches = [x.unsqueeze(0).unsqueeze(0)]

            tf_per_batch = batch_size*stride

            best_ent, best_z, best_timeframe = float('inf'), None, None
            for j, batch in enumerate(x_batches):
                z = model(batch)
                # find entropy for each location
                h = -1 * (F.softmax(z, 1) * F.log_softmax(z, 1)).sum(1)
                h_min = torch.min(h)
                if h_min < best_ent:
                    best_ent, best_z, best_h = h_min, z, h

            best_idx = torch.argmin(best_h)
            b, r, c = np.unravel_index(best_idx.item(), best_h.shape)
            p = F.softmax(best_z[b, :, r, c])
            probs = probs + p
            y_h = torch.argmax(p, dim=0).cpu().data.item()
            preds[y_h] += 1
            xy_count += 1
            inc = int(y_h == y)
            correct[i] += inc
            correct_total += inc

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


def load_data(path):
    data = []
    for d in os.listdir(path):
        filenames = glob.glob(path + "/" + d + "/*.npy")
        data += [(np.load(f), d) for f in filenames]
    return data


def main(opts):
    # training script
    if opts.cuda is not None:
        print("using CUDA device {}".format(opts.cuda))
        cuda_dev = int(opts.cuda)
    else:
        cuda_dev = None
    sys.stdout.flush()

    # initialize data loader
    dataset = pr_dataset.PianoRollDataset(os.getcwd() + "/" + opts.data_dir, "labels.csv", "test")
    dataloader = DataLoader(
            dataset,
            batch_size=opts.batch_size,
            shuffle=True,
            num_workers=4,
            collate_fn=lambda b : list(list(l) for l in zip(*b))
    )

    # set up the model
    net = cnn.FCNN(dataset.get_y_count(), opts.batch_size, cuda_dev, opts.max_w)

    saved_state = torch.load(opts.load, map_location='cpu')
    net.load_state_dict(saved_state)
    if cuda_dev is not None:
        net = net.cuda(cuda_dev)

    # model, data, num_per_class, cuda_dev, stride, tf_output_file
    with torch.no_grad():
        test(net, dataset, 5000, max_w=opts.max_w, stride=100, batch_size=opts.batch_size, cuda_dev=cuda_dev)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="clean_preproc")
    parser.add_argument("--max_w", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("-c", "--cuda", type=int, default=None)
    parser.add_argument("--load", default=None)

    args = parser.parse_args()
    main(args)