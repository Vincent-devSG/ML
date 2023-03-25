# importing useful library
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

start_time = time.time()

# Read the data...
name_file = './data_kmeans.txt'
columns = ['x', 'y']
data_in = pd.read_csv(name_file, names=columns, sep=' ')

x = np.asarray(data_in['x'])
y = np.asarray(data_in['y'])
X = (x, y)

# Defile epsilon
epsilon = 0.0001

# Define the number of centroids/cluster
nb_centroids = 3


def K_means(x, y, epsilon, nb_centroids):
    # Initialisation of the clusters
    global cluster

    # Initialisation of the centroids
    centroids = np.zeros((nb_centroids, 2))
    old_centroids = np.zeros((nb_centroids, 2))

    # Initialisation of the centroids randomly but close to the clusters
    for i in range(nb_centroids):
        centroids[i][0] = np.random.uniform(min(x), max(x))
        centroids[i][1] = np.random.uniform(min(y), max(y))

    centroids[0][0] = 0
    centroids[0][1] = 6
    centroids[1][0] = 2
    centroids[1][1] = 0
    centroids[2][0] = 8
    centroids[2][1] = 3

    # While the convergence is not reached
    while np.linalg.norm(centroids - old_centroids) > epsilon:

        # Calculate the distance between each point and each centroid
        distance = np.zeros((len(x), nb_centroids))
        for i in range(len(x)):
            for j in range(nb_centroids):
                distance[i][j] = np.linalg.norm([x[i], y[i]] - centroids[j])

        # Assign each point to the closest centroid (index of the nearest centroid)
        cluster = np.zeros(len(x))
        for i in range(len(x)):
            cluster[i] = np.argmin(distance[i])

        # Update the centroids
        for i in range(nb_centroids):
            centroids[i][0] = np.mean(x[cluster == i])
            centroids[i][1] = np.mean(y[cluster == i])

        # Update the old centroids
        old_centroids = centroids.copy()

    return centroids, cluster


centroids, cluster = K_means(x, y, epsilon, nb_centroids)

# Plot the data
plt.figure()
plt.scatter(x, y, c=cluster, cmap='rainbow')
plt.scatter(centroids[:, 0], centroids[:, 1], c='black')
plt.show()

print(" centroids = \n", centroids)  # print the centroids

print("--- %s seconds ---" % (time.time() - start_time))
