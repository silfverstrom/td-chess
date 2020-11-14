import numpy as np
import pdb
import random
import sys

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
    output_file = sys.argv[2]

    print("Reading input")
    fr = open(input_file, 'r')

    Y = []
    while True:
        line = fr.readline()
        if not line:
            break

        Y.append(float(line))

    fr.close()


    print("Normalising")
    mi = min(Y)
    mx = max(Y)
    print("Max", mx)
    print("min", mi)
    fw = open(output_file, 'w')

    for y in Y:
        d = mx - mi
        if d == 0:
            d = 1
        y_ = (y - mi) / d
        fw.write(str(y_)+'\n')

    fw.close()

    print("Done");

