# importing useful library
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

start_time = time.time()

# Read the data...
name_file = './data_pca.txt'
columns = ['x', 'y']
data_in = pd.read_csv(name_file, names=columns, sep=' ')

x = np.asarray(data_in['x'])
y = np.asarray(data_in['y'])
X = (x, y)


def PCA():

    # Compute the mean of the data cloud
    mu = np.mean(X, axis=1)

    x_centered = []
    y_centered = []

    # Center the input examples to the origin
    for i in range(len(x)):
        x_centered.append(x[i] - mu[0])
        y_centered.append(y[i] - mu[1])

    X_centered = np.array([x_centered, y_centered])

    # Compute the covariance matrix
    X_centered_cov = 1 / len(x) * np.dot(X_centered, X_centered.T)

    # Compute the P-largest eigen vectors of the covariance matrix (P = 1)
    eigen_values, eigen_vectors = np.linalg.eig(X_centered_cov)

    # We chose P = 1
    P = 1

    #  Sort the eigen values in descending order
    eigen_values_sorted = np.sort(eigen_values)[::-1]

    # Sort the eigen vectors in descending order
    eigen_vectors_sorted = eigen_vectors[:, eigen_values.argsort()[::-1]]

    # Keep the P-largest eigen vectors
    eigen_vectors = eigen_vectors_sorted[:, :P]

    # Project X_centered onto mu_p (eigen vectors)
    Y_projected = np.dot(X_centered.T, eigen_vectors)

    print(Y_projected.shape)
    print(Y_projected)


PCA()
