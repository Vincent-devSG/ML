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
    input_size = 2  # nb of col input x1, x2
    output_size = 3  # nb of different value output y
    hidden_size = 4  # We did choose this

    # Define learning rates
    alpha1 = 1e-3
    alpha2 = 1e-3

    # Define convergence criteria
    epsilon = 0.001
    E = 1e5
    pE = E
    delta_E = 1e5

    # Define iteration counter
    itera = 0

    # Create X, Y, V, W and Xbar | INITIAL
    X = np.hstack((x1.reshape(-1, 1), x2.reshape(-1, 1)))
    Y = generateY(y)
    V, W = generateVW(input_size, output_size, hidden_size)
    Xbar = generateXbar(X)

    while (abs(delta_E) > epsilon):
        itera += 1
        E, G, Fbarbar, Xbarbar, Fbar, F = FWP(V, W, Xbar, Y)
        V, W = BWP(V, W, Xbar, Y, G, Fbar, F, alpha1, alpha2)
        delta_E = E - pE
        pE = E
        print("itera: ", itera, "Error", E)
        print("delta Error", delta_E)



def generateY(y):
    Y = np.zeros((y.shape[0], 3))
    for i in range(y.shape[0]):
        Y[i, int(y[i])] = 1
    return Y


def generateVW(input_size, output_size, hidden_size):
    V = np.random.rand(input_size + 1, hidden_size)
    W = np.random.rand(hidden_size + 1, output_size)
    return V, W


def generateXbar(X):
    # Create a matrix of ones with the same number of rows as X
    ones = np.ones((X.shape[0], 1))

    # Concatenate the matrix of ones with X along the second axis
    Xbar = np.concatenate((ones, X), axis=1)
    return Xbar


def FWP(V, W, Xbar, Y): # Forward propagation
    # XbarTranspose = np.transpose(Xbar)
    # Calculate X bar bar
    Xbarbar = np.matmul(Xbar, V)
    F = 1 / (1 + np.exp(-Xbarbar))

    # Create Fbar
    ones = np.ones((F.shape[0], 1))
    Fbar = np.concatenate((ones, F), axis=1)

    # FbarTranspose = np.transpose(Fbar)
    # Calculate F bar bar
    Fbarbar = np.matmul(Fbar, W)

    # Apply the signoid to find G and E
    G = 1 / (1 + np.exp(-Fbarbar))
    E = (1 / 2) * np.sum((G - Y) ** 2)
    return E, G, Fbarbar, Xbarbar, Fbar, F


def BWP(V, W, Xbar, Y, G, Fbar, F, alpha_1, alpha_2): # backward propagation
    # calculate the new matrix W
    dG = (G - Y) * G * (1 - G)
    dW = alpha_1 * np.matmul(Fbar.T, dG)
    W -= dW

    # Calculate the new matrix V
    dFbar = np.matmul(dG, W.T)
    dF = dFbar[:, 1:] * F * (1 - F)
    dV = alpha_2 * np.matmul(Xbar.T, dF)
    V -= dV

    return V, W


FFNN()
