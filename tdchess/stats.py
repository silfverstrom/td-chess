import numpy as np
import pdb
import random
import sys
import scipy
import scipy.stats


def normalise(Y):
    mi = min(Y)
    mx = max(Y)
    Y_norm = []
    for y in Y:
        d = mx - mi
        if d == 0:
            d = 1
        y_ = (y - mi) / d
        Y_norm.append(y_)
    return Y_norm

if __name__ == '__main__':
    input_file = sys.argv[1]

    print("Reading input")
    fr = open(input_file, 'r')

    Y = []
    while True:
        line = fr.readline()
        if not line:
            break

        line = line.split(',')
        val = float(line[1])
        # artificial
        mx = 2000
        mi = -2000
        #normalise
        if val > mx or val < mi:
            continue
        Y.append(val)

    fr.close()


    #Y = list(filter(lambda y: y >= -2000 and y <= 2000, Y))
    print("Normalising")
    mi = min(Y)
    mx = max(Y)
    print("len", len(Y))
    print("Max", mx)
    print("min", mi)
    mean = np.mean(Y)
    std = np.std(Y)
    print('mean', mean)
    print('std', std)

    Y_ = normalise(Y)
    print(Y[0:100])

    print(Y_[0:100])

    print("mean normalised", np.mean(Y_))
    print("std normalised", np.std(Y_))

    #Y_ = scipy.stats.zscore(Y)
    #print(Y[0:100])
    #print(Y_[0:100])
    #print(np.mean(Y_))

