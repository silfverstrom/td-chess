import numpy as np
import tensorflow as tf
import os

X = []
Y = []
fens = []

files = []
for path in os.listdir('data'):
    data = np.load('data/'+path)
    X.extend(data['X'])
    Y.extend(data['Y'])
    fens.extend(data['F'])
    print('Parsed', path)

print('Saving')
np.save('data/large_X', X)
np.save('data/large_Y', Y)
np.save('data/large_F', fens)
