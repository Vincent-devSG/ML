# importing useful library

import numpy as np
import matplotlib.pyplot as plt

# We read the file and store into an array the data

input_data = []
output_data = []
data = []
training_data = []
test_data = []
file = open('data_lab1.txt', 'r')

# initialize values
theta = np.random.rand(2)
error = pow(10, 3)
gradiant_error = pow(10, 3)
epsilon = 10
it = 0
learning_rate = pow(10, -3)

for line in file:
    input_output = line.strip().split()
    input_value = float(input_output[0])
    output_value = float(input_output[1])
    input_data.append(input_value)
    output_data.append(output_value)
    data.append((input_value, output_value))

break_line = int(len(data) * 0.7)
training_data = data[:break_line]
test_data = data[break_line:]

x_train = []
y_train = []

for i in range(0, len(training_data)):
    x_train.append(training_data[i][0])
    y_train.append(training_data[i][1])

x_test = []
y_test = []
print(training_data[4][1])

for i in range(0, len(test_data)):
    x_test.append(test_data[i][0])
    y_test.append(test_data[i][1])

plt.figure(5)
plt.plot(x_train, y_train)
plt.title("training data")
plt.xlabel("Input data")
plt.ylabel("Output data")
plt.show()

plt.plot(x_test, y_test)
plt.title("test data")
plt.xlabel("Input data")
plt.ylabel("Output data")
plt.show()


while abs(gradiant_error) > epsilon:
    it += 1

    for n in range(2):
        sum = 0
        for i in range(len(x_train)):
            value_x = x_train[i]
            value_y = y_train[i]
            if n == 0:
                sum += (theta[0] + theta[1] * value_x - value_y)
            elif n == 1:
                sum += (theta[0] + theta[1] * value_x - value_y) * value_x

        theta[n] = theta[n] - learning_rate * (1.0 / len(x_train)) * sum

sse = 0
for i in range(len(x_train)):
    x_value = x_train[i]
    y_value = y_train[i]
    sse += (theta[0] + theta[1]*x_value - y_value)**2

mse = sse / (2*len(x_train))

if it > 1:
    delta_error = abs(mse - prev_mse)
else:
    delta_error = np.inf  # set delta_error to infinity for the first iteration
prev_mse = mse  # store the current MSE as prev_MSE for the next iteration

if it == 1:
    prev_mse = mse
else:
    prev_mse = curr_mse
mse_history.append(mse)
curr_mse = sse(training_data, theta)
delta_mse = abs(curr_mse - prev_mse)