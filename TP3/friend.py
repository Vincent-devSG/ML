# Reinforcement Learning

import numpy as np
import random
import matplotlib.pyplot as plt

NORTH = 0
SOUTH = 1
WEST = 2
EAST = 3

MIN = -100

def get_next_state_value(x, y, action, V):
    weights = {
        NORTH: (0.8, 0, 0.1, 0.1),
        SOUTH: (0, 0.8, 0.1, 0.1),
        WEST: (0.1, 0.1, 0.8, 0),
        EAST: (0.1, 0.1, 0, 0.8)
    }

    north_weight, south_weight, west_weight, east_weight = weights[action]

    if x - 1 >= 0 and not (x - 1 == 1 and y == 1):
        north_value = north_weight * V[(x - 1),y]
    else:
        north_value = north_weight * V[x,y]

    if x + 1 < V.shape[0] and not (x + 1 == 1 and y == 1):
        south_value = south_weight * V[(x + 1),y]
    else:
        south_value = south_weight * V[x,y]

    if y - 1 >= 0 and not (x == 1 and y - 1 == 1):
        west_value = west_weight * V[x,(y - 1)]
    else:
        west_value = west_weight * V[x,y]

    if y + 1 < V.shape[1] and not (x == 1 and y + 1 == 1):
        east_value = east_weight * V[x,(y + 1)]
    else:
        east_value = east_weight * V[x,y]

    return north_value + south_value + west_value + east_value


error_list = []

reward = np.full((3, 4), -0.02)
reward[1][3] = -1
reward[0][3] = 1
reward[1][1] = 0

discount = 0.99

V = np.zeros((3,4))
epsilon = 0.00001
deltaV = np.full((3,4), 1)

def checkEnd(x, y):
    return (i == 1 and j == 3) or (i == 0 and j == 3)

def evaluateStateValue(i, j, V):
    if checkEnd(i, j) or (i == 1 and j == 1):
        return reward[i,j]
    max = MIN
    for action in range(0, 4):
        value = get_next_state_value(i, j, action, V)
        if value > max:
            max = value

    return reward[i,j] + discount * max

print(V)

for iter in range(0, 1000):
    V_old = np.array(V)
    V_new = np.array(V)
    #V_new = V
    for i in range(0, 3):
        for j in range(0, 4):
            V_new[i,j] = evaluateStateValue(i, j, V)

    V = V_new

    deltaV = np.abs(V_new - V_old)
    error_list.append(np.max(deltaV))

    print("Iteration: ", iter)
    print(V)

    if np.max(deltaV) <= epsilon:
        print(f"Algorithms converged after {iter+1} iterations:")
        print(V)
        break

    if iter == 999:
        print(f"Algorithms did not converged after {iter+1} iterations:")
        print(V)

P = np.zeros((3,4))

def evaluatePolicy(i, j, V):
    if checkEnd(i, j):
        return reward[i,j]
    action = P[i,j]
    value = get_next_state_value(i, j, action, V)
    return reward[i,j] + discount * value

for i in range(0, 3):
    for j in range(0, 4):
        if checkEnd(i,j) or (i == 1 and j == 1):
            P[i,j] = -1
            continue
        max = 0
        for action in range(0, 4):
            value = get_next_state_value(i, j, action, V)
            if value > max:
                max = value
                P[i,j] = action

print("Final policy (0 - North, 1 - South, 2 - West, 3 - East):")
print(P)

plt.figure(5)
plt.plot(error_list)
plt.xlabel('Iterations')
plt.ylabel('Error')
plt.show()
