import matplotlib.pyplot as plt
import numpy as np

from elman_network import *

def get_y_result(y,y1,y2,noise):
    x = ((0.8-(0.5*np.exp(-(y1**2))))*(y1))-((0.3+(0.9*np.exp(-(y1**2))))*(y2)) + (0.1*np.sin(np.pi*y1)) + noise
    return x

# random noise
noise_input = np.random.uniform(-0.05,0.05,153)

data_y = np.zeros(153).tolist()

train_y = np.zeros(125).tolist()

test_y = np.zeros(25).tolist()
test_y_for_plot = np.zeros(25).tolist()

# generate data
data_y[0] = 0.5
data_y[1] = 1
for i in range(2, 153):
    data_y[i] = get_y_result(data_y[i],data_y[i-1],data_y[i-2], noise_input[i])

# normalize data between 0.1 - 0.9
data_ymax = max(data_y) 
data_ymin = min(data_y)

for i in range(0, 153):
    data_y[i] = 0.1 + (8/10)*(data_y[i] - data_ymin) / (data_ymax - data_ymin)

# train data
for i in range (3,128):
    train_y[i-3] = np.array([data_y[i]])
    train_y[i-3].shape = (1,1)

# test data
for i in range (128,153):
    test_y[i-128] = np.array([data_y[i]])
    test_y[i-128].shape = (1,1)
    test_y_for_plot[i - 128] = data_y[i]

number_of_repeat = 1
average_train_error = 0
average_test_error = 0
average_iteration = 0
test_value = 0
for i in range(number_of_repeat):
    neural_network = ElmanNetwork(7, 3, 1)
    train_results = neural_network.train(5000, 0.001, 0.5, 0.3, train_y)
    average_train_error += train_results[0]
    average_iteration += train_results[1]

    test_results = neural_network.test(test_y)
    average_test_error = test_results[0]
    test_result_array = test_results[1]

average_train_error = average_train_error / number_of_repeat
average_test_error = average_test_error / number_of_repeat
average_iteration = average_iteration / number_of_repeat

print(average_train_error)
print(average_test_error)
print(average_iteration)

plt.plot(test_result_array,linestyle ="-", marker = "x")
plt.plot(test_y_for_plot,linestyle ="-",marker = "o")
plt.show()

# print("AAA")