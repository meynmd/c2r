import argparse
import glob
import os
import sys
import time
import random
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from torch.autograd import Variable
from torch import cuda

# import encoder
import crnn
import pr_dataset


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


def make_batch(tensors, max_w, cuda_dev=None):
    batch_size = len(tensors)
    max_w = min(max_w, min(t.shape[1] for t in tensors))

    if max_w % 16 != 0:
        max_w = 16*(max_w // 16 + 1)

    x_batch = []
    for tensor in tensors:
        # tensor = (tensor > 0.).type(torch.float)
        tensor = tensor.view(1, 1, tensor.shape[0], tensor.shape[1])
        if tensor.shape[3] > max_w:
            tensor = random_crop(tensor, max_w)
        elif tensor.shape[3] < max_w:
            tensor = torch.nn.functional.pad(tensor, (0, max_w - tensor.shape[3], 0, 0))
        assert(tensor.shape[3] == max_w)
        x_batch.append(tensor)

    if cuda_dev is None:
        x_batch = Variable(torch.cat(x_batch, 0))
    else:
        x_batch = Variable(torch.cat(x_batch, 0)).cuda(cuda_dev)

    return x_batch


def train(model, phase, dataloader, batch_size, loss_fn, optim, num_epochs=50, model_dir="model", cuda_dev=None, lr_decay=0.):
    best_loss = float("inf")
    phases = [phase]
    if phase == "val":
        num_epochs = 1
    else:
        phases.append("val")

    for epoch in range(num_epochs):
        optim.param_groups[0]['lr'] *= (1. - lr_decay)

        print("\n" + 80*"*" + "\nEpoch {}\tlr {}\n".format(epoch + 1, optim.param_groups[0]['lr']))

        # phases ["train", "val"], or ["val"]
        for phase in phases:
            running_loss, err = 0., 0.
            if phase == "train":
                model.train()
            else:
                model.eval()

            # dataloader should provide whatever batch size was specified when instantiated
            for i, data in enumerate(dataloader[phase]):
                x, y = data
                batch_size = len(x)
                y = Variable(torch.LongTensor(y))
                if cuda_dev is not None:
                    y = y.cuda(cuda_dev)

                x = make_batch(x, model.max_w, cuda_dev)

                optim.zero_grad()

                # run and calc loss
                z = model(x)
                loss = loss_fn(z, y)


                # update model
                if phase == "train":
                    running_loss += loss.data.item()
                    loss.backward()
                    optim.step()
                else:
                    # make best prediction and find err
                    _, y_hat = torch.max(z, 1)
                    err_i = (y_hat.data != y.data).sum().item() / float(z.shape[0])
                    err += err_i
                    val_loss = loss.data.item()

            # print progress
            if phase == 'train':
                avg_loss = running_loss / len(dataloader['train'])
            else:
                avg_loss = val_loss
            print("{} loss: {:.5}".format(phase, avg_loss))
            if phase != 'train':
                print("{} err: {:.0%}".format(phase, float(err) / float(i + 1)))

            # save model if best so far, or every 100 epochs
            if phase == "val" and val_loss < best_loss:
                if epoch > 99:
                    save_name = "model-loss{:.3}-epoch{}.pt".format(avg_loss, epoch + 1)
                    # save_name += "-".join(time.asctime().split(" ")[:-1]).replace(":", ".")
                else:
                    save_name = "model-best"
                save_path = os.path.join(os.getcwd(), model_dir, save_name)
                torch.save(model.state_dict(), save_path)
                print("Model saved to {}".format(save_path))
                best_loss = val_loss
            elif phase == "val" and (epoch + 1) % 100 == 0 and epoch > 0:
                save_name = "checkpoint-epoch{}-loss{:.3}.pt".format(epoch + 1, avg_loss)
                save_path = "{}/{}".format(model_dir, save_name)
                torch.save(model.state_dict(), save_path)
                print("Model checkpoint saved to {}".format(save_path))

                # if avg_loss < best_loss or (epoch % 99 == 0 and epoch >= 99):
                #     if epoch > 99:
                #         save_name = "model-rnn{}-loss{:.3}-epoch{}.pt".format(model.rnn_size, avg_loss, epoch + 1)
                #         # save_name += "-".join(time.asctime().split(" ")[:-1]).replace(":", ".")
                #     else:
                #         save_name = "model-best"
                #     save_path = "{}/{}".format(model_dir, save_name)
                #     torch.save(model.state_dict(), save_path)
                #     print("Model saved to {}".format(save_path))
                # elif (epoch + 1) % 100 == 0 and epoch > 0:
                #     save_name = "checkpoint-rnn{}-epoch{}-loss{:.3}".format(model.rnn_size, epoch + 1, avg_loss)
                #     save_path = "{}/{}".format(model_dir, save_name)
                #     torch.save(model.state_dict(), save_path)
                #     print("Model saved to {}".format(save_path))

            sys.stdout.flush()
    print()


def load_data(path):
    data = []
    for d in os.listdir(path):
        filenames = glob.glob(path + "/" + d + "/*.npy")
        data += [(np.load(f), d) for f in filenames]
    return data


def main(opts):
    # training script
    if opts.use_cuda is not None:
        print("using CUDA device {}".format(opts.use_cuda))
        cuda_dev = int(opts.use_cuda)
    else:
        cuda_dev = None
    torch.manual_seed(opts.seed)
    print("random seed {}".format(opts.seed))
    sys.stdout.flush()

    # initialize data loader
    datasets = { p : pr_dataset.PianoRollDataset(os.getcwd() + "/" + opts.data_dir, "labels.csv", p)
                 for p in ("train", "val") }
    dataloaders = {
        p : DataLoader(
            datasets[p],
            batch_size=opts.batch_size if p == "train" else 8,
            shuffle=True,
            num_workers=0,
            collate_fn=lambda b : list(list(l) for l in zip(*b))
        ) for p in ("train", "val")
    }

    # set up the model
    # enc = encoder.Encoder(
    enc = crnn.Encoder(
        datasets["train"].get_y_count(),
        opts.batch_size,
        rnn_size=opts.rnn_size,
        num_rnn_layers=opts.rnn_layers,
        use_cuda=cuda_dev,
        max_w=opts.max_w
    )
    if opts.load:
        enc.load_state_dict(torch.load(opts.load))
    if cuda_dev is not None:
        enc = enc.cuda(cuda_dev)

    # set up the loss function and optimizer
    class_probs = torch.FloatTensor(max(datasets["train"].x_counts.keys()) + 1)
    for idx, count in datasets["train"].x_counts.items():
        class_probs[idx] = count
    class_probs /= sum(class_probs)
    lf = nn.CrossEntropyLoss(weight=torch.FloatTensor(
        torch.FloatTensor([1. for x in class_probs]) - class_probs).cuda(cuda_dev)
    )
    optim = torch.optim.SGD(enc.parameters(), float(opts.init_lr), momentum=0.9)
    # optim = torch.optim.RMSprop(enc.parameters(), 0.001)
    #enc.parameters(), float(opts.init_lr), momentum=0.95)

    # (model, phase, dataloader, batch_size, loss_fn, optim, num_epochs=50)
    os.makedirs(opts.model_dir, exist_ok=True)

    sys.stdout.flush()
    train(enc, "train", dataloaders, opts.batch_size, lf, optim, opts.max_epochs, cuda_dev=cuda_dev, model_dir=opts.model_dir, lr_decay=0.00025)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rnn_size", type=int, default=128)
    parser.add_argument("--rnn_layers", type=int, default=1)
    parser.add_argument("--data_dir", default="preprocessed")
    parser.add_argument("--max_epochs", type=int, default=12500)
    parser.add_argument("--max_w", type=int, default=5000)
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("-m", "--model_dir", default="crnn-512-model")
    parser.add_argument("--num_batch_valid", type=int, default=1)
    parser.add_argument("-s", "--seed", type=int, default=0)
    parser.add_argument("-c", "--use_cuda", type=int, default=None)
    parser.add_argument("-l", "--init_lr", default="10e-5")
    parser.add_argument("--load", default=None)

    args = parser.parse_args()
    main(args)