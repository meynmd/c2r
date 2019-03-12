import argparse, sys, math, os, glob
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from torch.autograd import Variable

import pr_dataset
# import pixel_cnn
import v2.cnn as auto

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


'''
find mean cross entropy for test set from each class with the given model
'''
def find_cross_entropy(net, class_datapoints, cuda_dev=None, batch_size=8, max_w=1024, left_pad=0, stride=None):
    net.eval()
    stride = stride or max_w
    class_ce_sum, num_class_batches = 0., 0
    cross_entropies = torch.zeros(len(class_datapoints))
    for i, ((x, y), (_, _)) in enumerate(class_datapoints):
        if left_pad > 0:
            x = F.pad(x, (left_pad, 0), 'constant', 0.)
        if x.shape[1] > max_w:
            x_batches = crop_and_batch(x, max_w, stride, batch_size)
        else:
            x_batches = [x.unsqueeze(0).unsqueeze(0)]

        num_class_batches += len(x_batches)

        sample_ce = torch.zeros(len(x_batches))
        for j, batch in enumerate(x_batches):
            target = batch.clone()
            batch, target = batch.cuda(cuda_dev), target.cuda(cuda_dev)
            z = net(batch)
            sample_ce[j] = F.binary_cross_entropy_with_logits(z, target)

            # class_ce_sum += bce.cpu().item()

        avg_ce = torch.mean(sample_ce)

        cross_entropies[i] = avg_ce.item()

    # avg_class_ce = class_ce_sum / num_class_batches
    return cross_entropies #avg_class_ce


def main(opts):
    # training script
    if opts.use_cuda is not None:
        print("using CUDA device {}".format(opts.use_cuda))
        cuda_dev = int(opts.use_cuda)
    else:
        cuda_dev = None
    sys.stdout.flush()

    # initialize data loader
    dataset = pr_dataset.PianoRollDataset(os.path.join(os.getcwd(), opts.data_dir), 'labels.csv', 'test')

    weights_dir = os.path.join(os.getcwd(), opts.weights)

    with open(os.path.join(weights_dir, 'avg_entropies.csv'), 'r') as fp:
        values = [line.strip().split(',') for line in fp.readlines()]
        class_avg_entropies = {c.replace('-', ' ') : float(v) for c, v in values}

    labels = dataset.idx2name.items()
    if opts.classes is not None:
        selected_labels = [s.replace('-', ' ') for s in opts.classes]
        labels = [l for i, l in labels if l in selected_labels]
    print (labels)
    label2idx = {l : i for (i, l) in enumerate(labels)}

    # net = pixel_cnn.PixelCNN(1, 32, 64)
    net = auto.AutoEncoder(opts.batch_size, cuda_dev, opts.max_w)
    for param in net.parameters():
        param.requires_grad = False

    num_classes = len(labels)
    class_ces = np.zeros([num_classes, num_classes])
    accuracy = {}
    for i, data_label in enumerate(labels):
        print('identifying class {}'.format(data_label), file=sys.stderr)
        class_datapoints = list(dataset.get_from_class(data_label))
        cross_entropies = torch.zeros(len(class_datapoints), len(labels)).cpu()
        for j, model_label in enumerate(labels):
            weights_file = os.path.join(weights_dir, model_label.replace(' ', '-') + '.pt')
            saved_state = torch.load(weights_file, map_location='cpu')
            # initialize network
            # in_channels, h_channels, discrete_channels

            net.load_state_dict(saved_state)
            if cuda_dev is not None:
                net = net.cuda(cuda_dev)

            # find_cross_entropy returns an array of cross entropies indexed by example
            #cross_entropies[:, j]
            cross_entropies[:, j] = find_cross_entropy(
                net, class_datapoints, cuda_dev, batch_size=opts.batch_size, max_w=opts.max_w, left_pad=opts.left_pad,
                stride=100
            ).cpu()

            # cross_entropies[:, j] /= class_avg_entropies[model_label]   # normalize by mean entropy for this model
            print('\t{} model gives cross entropy:\t{}'.format(model_label, torch.mean(cross_entropies[:, j])), file=sys.stderr)

        predictions = torch.argmin(cross_entropies, dim=1)
        num_correct = torch.sum(predictions == i).cpu().item()
        accuracy[i] = num_correct / float(predictions.shape[0])
        print('accuracy for class {}: {}'.format(data_label, accuracy[i]))

        class_ces[i, :] = torch.mean(cross_entropies, dim=0)


    # print("*"*10, "Normalized cross entropies:", "*"*10, "\n")
    # print(" " * 11, end="")


    short_label = []
    for label in labels:
        l = min(8, len(label))
        short_label.append("{:>8}".format(label[:l]))
    print(11*' ', end='')
    for sl in short_label:
        print("{}".format(sl), end="   ")
    print()
    for i, sl in enumerate(short_label):
        print(sl, end="   ")
        for j in range(class_ces.shape[1]):
            print("{:8.4}".format(class_ces[i][j]), end="   ")
        print()

    print()
    predicted_classes = np.argmin(class_ces, axis=1)
    for i, label in enumerate(labels):
        print('{} predicted as {}'.format(label, labels[predicted_classes[i]]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="../../preprocessed")
    parser.add_argument("--max_w", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("-c", "--use_cuda", type=int, default=None)
    parser.add_argument("--weights", default=None)
    parser.add_argument("--left_pad", type=int, default=0)
    parser.add_argument("--classes", nargs='*')

    args = parser.parse_args()
    main(args)

