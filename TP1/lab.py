# importing useful library

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# We read the file and store into an array the data
filenameData70 = 'data_lab1_70.txt'
filenameData30 = 'data_lab1_30.txt'

columns = ['x', 'y']
data70Point = pd.read_csv(filenameData70,
                          names=columns,
                          sep=' ')

x = np.asarray(data70Point['x'])
y = np.asarray(data70Point['y'])


# We define the function SSE
def SSE(x, y, theta):
    return sum((theta[0] + theta[1] * x - y) ** 2)


def BatchGradientDescent(x, y):  # BGD

    # Init parameters
    THETA = np.random.rand(2)
    ERROR = 10e3
    old_ERROR = 10e3
    d_ERROR = 10e3
    EPSILON = 10e-5
    iteration = 0
    learning_rate = 10e-2

    # Repeat until convergence
    while d_ERROR > EPSILON:
        # Calculate the new value of THETA
        THETA[0] = THETA[0] - learning_rate * (1 / len(x)) * sum((THETA[0] + THETA[1] * x - y))
        THETA[1] = THETA[1] - learning_rate * (1 / len(x)) * sum((THETA[0] + THETA[1] * x - y) * x)

        # Calculate the new value of ERROR
        ERROR = SSE(x, y, THETA)

        # Calculate the new value of d_ERROR
        d_ERROR = abs(ERROR - old_ERROR)

        # Update the value of ERROR
        old_ERROR = ERROR

        # Update the value of iteration
        iteration = iteration + 1

    return THETA, iteration


def StochasticGradientDescent(x, y):  # SGD

    # Init parameters
    THETA = np.random.rand(2)
    ERROR = 10e3
    old_ERROR = 10e3
    d_ERROR = 10e3
    EPSILON = 10e-5
    iteration = 0
    learning_rate = 10e-3

    # Repeat until convergence
    while d_ERROR > EPSILON:

        # Calculate the new value of THETA
        for j in range(2):
            random_index = np.random.randint(0, len(x))
            THETA[j] = THETA[j] - learning_rate * (1 / len(x)) * sum((THETA[0] + THETA[1] * x - y) * x)

        # Calculate the new value of ERROR
        ERROR = SSE(x, y, THETA)

        # Calculate the new value of d_ERROR
        d_ERROR = abs(ERROR - old_ERROR)

        # Update the value of ERROR
        old_ERROR = ERROR

        # Update the value of iteration
        iteration = iteration + 1

    return THETA, iteration


def ClosedFormSolution(x, y):  # CFS

    # Init parameters
    THETA = np.random.rand(2)
    ERROR = 10e3
    old_ERROR = 10e3
    d_ERROR = 10e3
    EPSILON = 10e-5
    iteration = 0
    learning_rate = 10e-2

    # Calculate the new value of THETA
    THETA[1] = sum((x - np.mean(x)) * (y - np.mean(y))) / sum((x - np.mean(x)) ** 2)
    THETA[0] = np.mean(y) - THETA[1] * np.mean(x)

    # Calculate the new value of ERROR
    ERROR = SSE(x, y, THETA)

    # Calculate the new value of d_ERROR
    d_ERROR = abs(ERROR - old_ERROR)

    # Update the value of ERROR
    old_ERROR = ERROR

    # Update the value of iteration
    iteration = iteration + 1

    return THETA, iteration


# We call the function BatchGradientDescent and we print the result
THETA, iteration = BatchGradientDescent(x, y)
print("THETA = ", THETA)
print("iteration = ", iteration)

# Test the model with the data of 30%
data30Point = pd.read_csv(filenameData30, names=columns, sep=' ')
x2 = np.asarray(data30Point['x'])
y2 = np.asarray(data30Point['y'])

Y_PREDICTED = np.array([])
for i in range(len(x2)):
    Y_PREDICTED = np.append(Y_PREDICTED, [0] + THETA[1] * x2[i])

# Plot the input data, the model and the trained data, the Y_PREDICTED (thin dotted orange line)
plt.plot(x, y, 'bo', label='Input data')
plt.plot(x, THETA[0] + THETA[1] * x, 'r', label='Model')
plt.plot(x2, y2, 'go', label='Trained data')
plt.plot(x2, Y_PREDICTED, color='orange', ls='--', label='Y_PREDICTED')

# The legend contains the number of iteration
plt.legend(loc='upper left', title='Iteration: ' + str(iteration))

# The title of the plot
plt.title('Batch Gradient Descent')
plt.show()


# We call the function StochasticGradientDescent and we print the result
THETA, iteration = StochasticGradientDescent(x, y)
print("THETA = ", THETA)
print("iteration = ", iteration)

# Test the model with the data of 30%
data30Point = pd.read_csv(filenameData30, names=columns, sep=' ')
x2 = np.asarray(data30Point['x'])
y2 = np.asarray(data30Point['y'])

Y_PREDICTED = np.array([])
for i in range(len(x2)):
    Y_PREDICTED = np.append(Y_PREDICTED, [0] + THETA[1] * x2[i])

# Plot the input data, the model and the trained data, the Y_PREDICTED (thin dotted orange line)
plt.plot(x, y, 'bo', label='Input data')
plt.plot(x, THETA[0] + THETA[1] * x, 'r', label='Model')
plt.plot(x2, y2, 'go', label='Trained data')
plt.plot(x2, Y_PREDICTED, color='orange', ls='--', label='Y_PREDICTED')

# The legend contains the number of iteration
plt.legend(loc='upper left', title='Iteration: ' + str(iteration))

# The title of the plot
plt.title('Stochastic Gradient Descent')
plt.show()

# We call the function ClosedFormSolution and we print the result

THETA, iteration = ClosedFormSolution(x, y)
print("THETA = ", THETA)
print("iteration = ", iteration)

# Test the model with the data of 30%
data30Point = pd.read_csv(filenameData30, names=columns, sep=' ')
x2 = np.asarray(data30Point['x'])
y2 = np.asarray(data30Point['y'])

Y_PREDICTED = np.array([])
for i in range(len(x2)):
    Y_PREDICTED = np.append(Y_PREDICTED, [0] + THETA[1] * x2[i])


# Plot the input data, the model and the trained data, the Y_PREDICTED (thin dotted orange line)
plt.plot(x, y, 'bo', label='Input data')
plt.plot(x, THETA[0] + THETA[1] * x, 'r', label='Model')
plt.plot(x2, y2, 'go', label='Trained data')
plt.plot(x2, Y_PREDICTED, color='orange', ls='--', label='Y_PREDICTED')

# The legend contains the number of iteration
plt.legend(loc='upper left', title='Iteration: ' + str(iteration))

# The title of the plot
plt.title('Closed Form Solution')
plt.show()
