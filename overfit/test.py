
with open('data/labels.csv') as fp:
    idx2y = {x : y for (x, y) in [line.strip().split(',') for line in fp.readlines()]}

for ix, y in idx2y.items():
    print("{} : {}".format(ix, y))