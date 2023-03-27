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

    # Compute the P-largest eigen vectors of the covariance matrix
    eigen_values, eigen_vectors = np.linalg.eig(X_centered_cov)

    # Project X_centered onto mu_p (eigen vectors)
    Y_projected = np.dot(eigen_vectors.T, X_centered)

    # Represent Y_projected on the N-dimensional space
    plt.figure()
    plt.plot(Y_projected[0], Y_projected[1], 'ro')
    plt.title('PCA')
    plt.show()


PCA()
