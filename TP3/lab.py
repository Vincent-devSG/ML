# importing useful library
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Define the area the robot can move into

height, weight = 3, 4
area = [[-0.02 for x in range(weight)] for y in range(height)]
area = np.asarray(area)

# Define the obstacle at position (2, 2) Reward (4, 3) Pit (4, 2)
area[1][1] = 0
area[2][3] = 1
area[1][3] = -1


# Define weight of movement Direction
# [NORTH, SOUTH, EAST, WEST]
NORTH = np.asarray([0.8, 0, 0.1, 0.1])
SOUTH = np.asarray([0, 0.8, 0.1, 0.1])
EAST = np.asarray([0.1, 0.1, 0.8, 0])
WEST = np.asarray([0.1, 0.1, 0, 0.8])

D = [NORTH, SOUTH, EAST, WEST]

moves = {
    'N': [-1, 0],
    'S': [1, 0],
    'E': [0, 1],
    'W': [0, -1]
}


# Define the robot position
def nextStep(h, w, D):
    # Calculate the next position based on the probabilities of moving in different directions
    directions = ['N', 'S', 'E', 'W']
    probabilities = D

    chosen_direction = np.random.choice(directions, p=probabilities)
    next_h = h + moves[chosen_direction][0]
    next_w = w + moves[chosen_direction][1]

    # Check if the next position is valid
    if next_h < 0 or next_h >= height or next_w < 0 or next_w >= weight or area[next_h][next_w] == 0:
        return h, w
    else:
        return next_h, next_w


def optimalValueFunction():
    # Initialize the value matrix
    V = np.zeros((height, weight))
    V_old = np.ones((height, weight))
    itera = 0

    # Finally, assign the discount factor (γ) to be 0.99 and epsilon
    GAMMA = 0.99
    EPSILON = 0.00001
    delta = 0.1
    delta_list = []

    # Repeat until delta(V- V_old) > epsilon
    while abs(delta) > EPSILON:
        itera += 1
        print("itera: ", itera)

        # Calculate the value of each state
        for i in range(height):
            for j in range(weight):
                if area[i][j] != 0:

                    sums = [0] * 4  # Initialize sums with a list of zeros
                    # calculate sums of Psa(s')*V(s')
                    for k in range(4):
                        next_h, next_w = nextStep(i, j, D[k])
                        sums[k] = 0  # Initialize sums[k] to zero before the inner loop
                        for x in range(4):
                            if (D[k][x] != 0):
                                sums[k] += D[k][x] * V[next_h][next_w]
                    if area[i][j] == -1 or area[i][j] == 1:
                        V[i][j] = area[i][j]
                    else:
                        V[i][j] = area[i][j] + GAMMA * max(sums)

        # calculate delta
        delta = np.max(np.abs(V - V_old))
        delta_list.append(delta)

        # Update the value matrix and old matrix
        V_old = V.copy()

    return V, delta_list


def optimalPolicy(V):
    # Hence, the optimal policy is the action ('N', 'S', 'W', 'E') that maximizes the future expected pay-off
    policy = [['' for x in range(weight)] for y in range(height)]
    policy = np.asarray(policy)

    for i in range(height):
        for j in range(weight):
            if area[i][j] != 0:
                sums = [0] * 4
                for k in range(4):
                    next_h, next_w = nextStep(i, j, D[k])
                    sums[k] = 0
                    for x in range(4):
                        if (D[k][x] != 0):
                            sums[k] += D[k][x] * V[next_h][next_w]
                # Assign 'N', 'S', 'W', 'E' to the optimal policy
                if area[i][j] == -1 or area[i][j] == 1:
                    policy[i][j] = 'G'
                else:
                    policy[i][j] = list(moves.keys())[sums.index(max(sums))]
    return policy


V, delta_list = optimalValueFunction()
print(V)
policy = optimalPolicy(V)
print(policy)

plt.figure(5)
plt.plot(delta_list)
plt.xlabel('Iterations')
plt.ylabel('Error')
plt.show()
