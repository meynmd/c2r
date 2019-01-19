import os
import argparse

p = argparse.ArgumentParser()
p.add_argument("logfile")
p.add_argument("outfile")
args = p.parse_args()

outfile = open(os.path.join(os.getcwd(), args.outfile), 'w')
outfile.write('epoch,train_loss,val_loss\n')

with open(os.path.join(os.getcwd(), args.logfile), 'r') as fp:
    lines = fp.readlines()
    for line in lines:
        line = line.strip().split()
        if len(line) < 1:
            continue
        if line[0] == "Epoch":
            epoch = line[1]
            train_loss = None
            val_loss = None
        elif line[0] == "train":
            train_loss = line[2]
        elif line[0] == "val" and line[1] == "loss:":
            val_loss = line[2]
            outfile.write('{},{},{}\n'.format(epoch, train_loss, val_loss))
            epoch, train_loss, val_loss = None, None, None

outfile.close()

