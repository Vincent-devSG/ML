# importing useful library

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Read the data...
name_file = './data_ffnn_3classes.txt'
columns = ['x1', 'x2', 'y']
data_in = pd.read_csv(name_file, names=columns, sep='\t')

x1 = np.asarray(data_in['x1'])
x2 = np.asarray(data_in['x2'])
y = np.asarray(data_in['y'])


def FFNN():
    # Define the size of inputs, outputs and hidden
    input_size = 71  # nb of lines N
    output_size = 1  # nb of output y
    hidden_size = 4  # We did choose this

    # Define learning rates
    alpha1 = 1e-3
    alpha2 = 1e-3

    # Define convergence criteria
    epsilon = 10
    E = 1e5
    delta_E = 1e5

    # Define iteration counter
    itera = 0

    X = np.vstack((x1, x2))
    Y = generateY(y)
    V, W = generateVW(X, output_size, hidden_size)
    print(W)
    Xbar = generateXbar(x1, x2)

    while (abs(delta_E) > epsilon):
        itera += 1
        FWP(V, W, Xbar)


def generateY(y):
    outputs = []
    for elem in y:
        if elem not in outputs:
            outputs.append(elem)

    Y = [[0 for j in range(len(y))] for i in range(len(outputs))]

    for i in range(0, len(outputs)):
        for j in range(0, len(y)):
            if y[j] == i:
                Y[i][j] = 1
            else:
                Y[i][j] = 0
    return Y


def generateVW(X, output_size, hidden_size):
    input_size = X.shape[1]
    V = np.random.randn(hidden_size, input_size + 1)
    W = np.random.randn(output_size, hidden_size + 1)
    return V, W


def generateXbar(x1, x2):
    X = np.vstack((x1, x2))
    Xbar = np.empty((X.shape[0], X.shape[1] + 1))
    Xbar[:, 0] = 1
    Xbar[:, 1:] = X
    return Xbar


def FWP(V, W, Xbar):
    XbarTranspose = np.transpose(Xbar)
    Xbarbar = np.dot(V, XbarTranspose)
    F = 1 / (1 + np.exp(-Xbarbar))
    Fbar = np.vstack((np.ones((1, F.shape[1])), F))
    FbarTranspose = np.transpose(Fbar)
    Fbarbar = np.dot(W, FbarTranspose)


FFNN()
