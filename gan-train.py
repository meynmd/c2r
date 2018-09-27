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

import generator, discriminator
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


def make_batch(tensors, max_w):
    batch_size = len(tensors)
    max_w = min(max_w, min(t.shape[1] for t in tensors))
    x_batch = []
    for tensor in tensors:
        tensor = tensor.view(1, 1, tensor.shape[0], tensor.shape[1])
        if tensor.shape[3] > max_w:
            tensor = random_crop(tensor, max_w)
        x_batch.append(tensor)

    return Variable(torch.cat(x_batch, 0))


def train(model_generator, model_discriminator, dataloader, loss_fn, optim, phase='train', batch_size=1,
          num_batches_val=1, save_dir="gan_model", cuda_dev=None, learning_rate_init=1e-5, lr_decay=None):

    phases = [phase]
    if phase == 'train':
        phases.append('val')

    # phases ["train", "val"], or ["val"]
    for phase in phases:
        running_loss, err = 0., 0.
        if phase == "train":
            model_generator.train()
            model_discriminator.train()
        else:
            model_generator.eval()
            model_discriminator.eval()

        # dataloader should provide whatever batch size was specified when instantiated
        for i, data in enumerate(dataloader[phase]):
            real_gt = Variable(torch.ones(batch_size, 1), requires_grad=False)
            fake_gt = Variable(torch.zeros(batch_size, 1), requires_grad=False)

            # generator
            init_h = Variable(torch.FloatTensor(np.random.normal(0, 1, (model_generator.num_layers, batch_size, model_generator.rnn_dim))))
            init_c = Variable(torch.FloatTensor(np.random.normal(0, 1, (model_generator.num_layers, batch_size, model_generator.rnn_dim))))

            generated = model_generator(init_h, init_c)

            # TODO: loss and backprop

            # discriminator
            # run and calc loss on real data
            x_real, _ = data
            if cuda_dev is not None:
                x_real = x_real.cuda(cuda_dev)

            x_real = make_batch(x_real, model_discriminator.max_w)

            # optim.zero_grad()

            z_real = model_discriminator(x_real)

            # run on fake data
            x_fake = generated.unsqueeze(1)
            z_fake = model_discriminator(x_fake)

            print('z_real: {}\nz_fake: {}\n'.format(z_real.mean().item(), z_fake.mean().item()))

            # loss = loss_fn(z, y)
            # running_loss += loss.data[0]

            # update model
            # if phase == "train":
            #     loss.backward()
            #     optim.step()

            # make best prediction and find err
            # _, y_hat = torch.max(z, 1)
            # err += (y_hat.data[0] != y.data[0])

        # print progress
        # avg_loss = running_loss / float(i + 1)
        # print("{} err: {:.0%}\t{} loss: {:.5}".format(
        #     phase, err / float(i + 1), phase, avg_loss)
        # )

        # save model if best so far, or every 100 epochs
        # if phase == "val":
        #     if avg_loss < best_loss:
        #         best_loss = avg_loss
        #         if epoch > 99:
        #             save_name = "model_epoch{}_loss{:.3}_".format(epoch + 1, avg_loss)
        #             save_name += "-".join(time.asctime().split(" ")[:-1]).replace(":", ".")
        #         else:
        #             save_name = "model_best".format(epoch + 1, avg_loss)
        #         save_name += "_rnn-size{}.pt".format(model.rnn_size)
        #         save_path = "{}/{}".format(model_dir, save_name)
        #         torch.save(model.state_dict(), save_path)
        #         print("Model saved to {}".format(save_path))
        #     sys.stdout.flush()
    # print()


def load_data(path):
    data = []
    for d in os.listdir(path):
        filenames = glob.glob(path + "/" + d + "/*.npy")
        data += [(np.load(f), d) for f in filenames]
    return data


def main(opts):
    # training script
    if opts.use_cuda is not None:
        print("using CUDA device {}".format(opts.use_cuda), file=sys.stderr)
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
            batch_size=opts.batch_size if p == "train" else 1,
            shuffle=True,
            num_workers=2,
            collate_fn=lambda b : list(list(l) for l in zip(*b))
        ) for p in ("train", "val")
    }

    # set up the model
    gen = generator.Generator(use_cuda=opts.use_cuda)
    disc = discriminator.Discriminator(use_cuda=opts.use_cuda)
    if opts.load_gen:
        gen.load_state_dict(torch.load(opts.load_gen))
    if opts.load_disc:
        gen.load_state_dict(torch.load(opts.load_disc))
    if cuda_dev is not None:
        gen = gen.cuda(cuda_dev)
        disc = disc.cuda(cuda_dev)

    # set up the loss function and optimizer
    # class_probs = torch.FloatTensor(max(datasets["train"].x_counts.keys()) + 1)
    # for idx, count in datasets["train"].x_counts.items():
    #     class_probs[idx] = count
    # class_probs /= sum(class_probs)
    lf = nn.CrossEntropyLoss()
    optim_gen = torch.optim.SGD(gen.parameters(), lr=10**(-opts.init_lr), momentum=0.9)

    # train(model, phase, dataloader, batch_size, loss_fn, optim, num_epochs=50)
    train(gen, disc, dataloaders, lf, optim_gen, cuda_dev=cuda_dev)


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
    parser.add_argument("--num_batch_valid", type=int, default=1)
    parser.add_argument("--beam_size", type=int, default=5)
    parser.add_argument("-s", "--seed", type=int, default=0)
    parser.add_argument("-c", "--use_cuda", type=int, default=None)
    parser.add_argument("-l", "--init_lr", type=int, default=5)
    parser.add_argument("--load_gen", default=None)
    parser.add_argument("--load_disc", default=None)

    args = parser.parse_args()
    main(args)