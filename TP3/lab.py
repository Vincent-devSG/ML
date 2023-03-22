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

print(area)

# Define weight of movement Direction
# [NORTH, SOUTH, EAST, WEST]
NORTH = np.asarray([0.8, 0, 0.1, 0.1])
SOUTH = np.asarray([0, 0.8, 0.1, 0.1])
EAST = np.asarray([0.1, 0.1, 0.8, 0])
WEST = np.asarray([0.1, 0.1, 0, 0.8])

moves = {
    'N': [-1, 0],
    'S': [1, 0],
    'E': [0, 1],
    'W': [0, -1]
}

# Finally, assign the discount factor (γ) to be 0.99
GAMMA = 0.99


# Define the robot position
def nextStep(h, w, D):
    # Calculate the next position based on the probabilities of moving in different directions
    directions = ['N', 'S', 'E', 'W']
    probabilities = D

    chosen_direction = np.random.choice(directions, p=probabilities)
    print(chosen_direction)
    next_h = h + moves[chosen_direction][0]
    next_w = w + moves[chosen_direction][1]

    # Check if the next position is valid
    if next_h < 0 or next_h >= height or next_w < 0 or next_w >= weight or area[next_h][next_w] == 0:
        return h, w
    else:
        return next_h, next_w

def optimalValueFunction():
    # Initialize the value function to be 0 for all states
    V = np.zeros((height, weight))
    # Repeat until convergence