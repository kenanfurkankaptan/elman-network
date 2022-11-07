
import matplotlib.pyplot as plt
import numpy as np


class ElmanNetwork:

    def sigmoid(self, x):
        return (1 / (1 + np.exp(-x)))

    def sigmoid_derivative(self, x):
        s = self.sigmoid(x)
        return s * (1 - s)

    def __init__(self, hidden_layer_size, input_size, exit_layer_size):

        self.hidden_layer_size = hidden_layer_size
        self.input_size = input_size
        self.exit_layer_size = exit_layer_size

        self.weight_wu = np.random.uniform(low=-1, high=1, size=(self.hidden_layer_size, self.input_size))
        self.weight_wx = np.random.uniform(low=-1, high=1, size=(self.hidden_layer_size, self.hidden_layer_size))
        self.weight_wy = np.random.uniform(low=-1, high=1, size=(self.exit_layer_size, self.hidden_layer_size))

        self.previous_hidden_layer = np.zeros(self.hidden_layer_size)
        self.previous_hidden_layer.shape = (len(self.previous_hidden_layer), 1)

        self.test_x_init = 0

        return


    def train(self, epoch, min_error, learning_coef, momentum_coef, train_y):

        self.train_x = np.zeros(125).tolist()

        self.previous_hidden_layer = np.zeros(self.hidden_layer_size)
        self.previous_hidden_layer.shape = (len(self.previous_hidden_layer), 1)

        iteration_number_LE = 0
        average_train_error_LE = 0
        for i in range(epoch):

            wu_momentum = np.zeros((self.hidden_layer_size, self.input_size))
            wx_momentum = np.zeros((self.hidden_layer_size, self.hidden_layer_size))
            wy_momentum = np.zeros((self.exit_layer_size, self.hidden_layer_size))

            average_train_error = 0

            for idx, val_x in enumerate(self.train_x):
                input_train = np.empty(self.input_size)
                input_train.shape = (self.input_size, 1)

                for j in range(self.input_size):
                    if(j > idx):
                        input_train[j] = np.array([0])
                    else:
                        # input_train[j] = self.train_x[idx-j]
                        input_train[j] = np.array([2])

                v = np.matmul(self.weight_wu, input_train) + np.matmul(self.weight_wx, self.previous_hidden_layer)
                x = self.sigmoid(v)
                y = np.matmul(self.weight_wy, x)

                err = train_y[idx] - y

                if(idx < len(self.train_x)-1):
                    self.train_x[idx+1] = err
                else:
                    self.test_x_init = err
                
                back_propagation = np.matmul(np.transpose(self.weight_wy), err) * self.sigmoid_derivative(v)

                weight_wy_update = np.matmul(err, np.transpose(x)),
                weight_wy_update = weight_wy_update[0]
                weight_wu_update = np.matmul(back_propagation, np.transpose(input_train))
                weight_wx_update = np.matmul(back_propagation, np.transpose(self.previous_hidden_layer))

                self.weight_wy += (learning_coef * weight_wy_update) + (momentum_coef * wy_momentum)
                self.weight_wu += (learning_coef * weight_wu_update) + (momentum_coef * wu_momentum)
                self.weight_wx += (learning_coef * weight_wx_update) + (momentum_coef * wx_momentum)

                wy_momentum = weight_wy_update
                wu_momentum = weight_wu_update
                wx_momentum = weight_wx_update

                self.previous_hidden_layer[:] = x

                average_train_error += (err*err)/2

            average_train_error = average_train_error/len(train_y)

            iteration_number_LE += 1
            average_train_error_LE = average_train_error

            if(average_train_error < min_error):
                break

        return average_train_error_LE, iteration_number_LE

    def test(self, test_y):
        average_test_error = 0

        test_result = np.zeros(25)

        test_x = np.zeros(25).tolist()
        test_x[0] = self.test_x_init

        for idx, val_x in enumerate(test_x):
            input_test = np.empty(self.input_size)
            input_test.shape = (self.input_size, 1)
            for j in range(self.input_size):
                if(j > idx):
                    input_test[j] = self.train_x[len(self.train_x)+idx-j]
                else:
                    input_test[j] = test_x[idx-j]

            v = np.matmul(self.weight_wu, input_test) + np.matmul(self.weight_wx, self.previous_hidden_layer)
            x = self.sigmoid(v)
            y = np.matmul(self.weight_wy, x)

            test_result[idx] = y

            self.previous_hidden_layer[:] = x

            err = test_y[idx] - y
            average_test_error += (err*err)/2

            if(idx < len(test_x)-1):
                test_x[idx+1] = err

        average_test_error = average_test_error / len(self.train_x)

        return average_test_error, test_result
