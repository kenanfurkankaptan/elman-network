import matplotlib.pyplot as plt
import numpy as np


def get_y_result(y,y1,y2,noise):
    x = ((0.8-(0.5*np.exp(-(y1**2))))*(y1))-((0.3+(0.9*np.exp(-(y1**2))))*(y2))+ (0.1*np.sin(np.pi*y1)) + noise
    return x

def random_noise():
    return np.random.uniform(-0.05,0.05)

noise_input = np.random.uniform(-0.05,0.05,153)

data_y = np.zeros(153).tolist()

train_x = np.zeros(125).tolist()
train_y = np.zeros(125).tolist()

test_x = np.zeros(25).tolist()
test_y = np.zeros(25).tolist()
test2_y = np.zeros(25).tolist()
test_result = np.zeros(25)

data_y[0] = 0.5
data_y[1] = 1

train_x[0] = np.array([2])
train_x[0].shape = (1, 1)

for i in range(2, 153):
    data_y[i] = get_y_result(data_y[i],data_y[i-1],data_y[i-2], noise_input[i])



data_ymax = max(data_y) 
data_ymin = min(data_y)
for i in range(0, 153):
    data_y[i] = 0.1 + (8/10)*(data_y[i] - data_ymin) / (data_ymax + abs(data_ymin))


for i in range (3,128):                      # egitim kumesi degerleri atandi
    train_y[i-3] = np.array([data_y[i]])
    train_y[i-3].shape = (1,1)


for i in range (128,153):
    test_y[i-128] = np.array([data_y[i]])   # test kumesi degerleri atandi
    test_y[i-128].shape = (1,1)
    test2_y[i - 128] = data_y[i]




def elman_network(hidden_layer_size, epoch, min_error, learning_rate, momentum, past_input_number):

    input_size = past_input_number
    exit_layer_size = 1

    weight_wu = np.random.uniform(low = -1, high = 1, size=(hidden_layer_size, input_size))
    weight_wx = np.random.uniform(low = -1, high = 1, size=(hidden_layer_size, hidden_layer_size))
    weight_wy = np.random.uniform(low = -1, high = 1, size=(exit_layer_size, hidden_layer_size))

    previous_hidden_layer = np.zeros(hidden_layer_size)
    previous_hidden_layer.shape = (len(previous_hidden_layer), 1)

    def sigmoid(x):
        return (1 / (1 + np.exp(-x)))
        
    def sigmoid_derivative(x):
        s = sigmoid(x)
        return s * (1 - s)

    

    # iteration_number and average_train_error of last epoch
    iteration_number_LE = 0
    average_train_error_LE = 0

    for i in range(epoch):

        wu_momentum = np.zeros((hidden_layer_size, input_size))
        wx_momentum = np.zeros((hidden_layer_size, hidden_layer_size))
        wy_momentum = np.zeros((exit_layer_size, hidden_layer_size))

        average_train_error = 0


        
        for idx, val_x in enumerate(train_x):

            input_train = np.empty(input_size)
            input_train.shape = (input_size, 1)

            for j in range(input_size):
                if(j>idx):
                    input_train[j] = np.array([0])
                else:
                    input_train[j] = train_x[idx-j]


            v = np.matmul(weight_wu, input_train) + np.matmul(weight_wx, previous_hidden_layer)
            x = sigmoid(v)
            y = np.matmul(weight_wy, x)


            err = train_y[idx] - y

            if(idx<len(train_x)-1):
                train_x[idx+1] = err
            else:
                test_x[0] = err
                

            temp2 = np.matmul(np.transpose(weight_wy), err) * sigmoid_derivative(v)

            weight_wy_update =  np.matmul(err, np.transpose(x)),
            weight_wu_update =  np.matmul(temp2, np.transpose(input_train))
            weight_wx_update =  np.matmul(temp2, np.transpose(previous_hidden_layer))

            weight_wy += learning_rate * weight_wy_update[0] + momentum * wy_momentum
            weight_wu += learning_rate *  weight_wu_update + momentum * wu_momentum
            weight_wx += learning_rate * weight_wx_update + momentum * wx_momentum

            wy_momentum = weight_wy_update[0]
            wu_momentum = weight_wu_update
            wx_momentum = weight_wx_update

            previous_hidden_layer[:] = x

            average_train_error += (err*err)/2

        average_train_error = average_train_error/len(train_y)

        iteration_number_LE += 1
        average_train_error_LE = average_train_error

        print(average_train_error)
        print(i)

        if(average_train_error<min_error):
            print(average_train_error)
            print(i)
            break

    

    average_test_error = 0
    for idx, val_x in enumerate(test_x):

        input_test = np.empty(input_size)
        input_test.shape = (input_size, 1)
        for j in range(input_size):
            if(j>idx):
                input_test[j] = train_x[len(train_x)+idx-j]
            else:
                input_test[j] = test_x[idx-j]


        v = np.matmul(weight_wu, input_test) + np.matmul(weight_wx, previous_hidden_layer)
        x = sigmoid(v)
        y = np.matmul(weight_wy, x)


        test_result[idx]= y

        previous_hidden_layer[:] = x

        err = test_y[idx] - y
        average_test_error += (err*err)/2


        if(idx < len(test_x)-1):
            test_x[idx+1] = err



    average_test_error = average_test_error/len(test_x)

    return average_train_error_LE, average_test_error, iteration_number_LE,test_result



number_of_repeat = 1
average_train_error = 0
average_test_error = 0
average_iteration = 0
test_value = 0
for i in range(number_of_repeat):
    temp = elman_network(7, 5000, 0.001, 0.5, 0.0, 3)
    average_train_error += temp[0]
    average_test_error += temp[1]
    average_iteration += temp[2]
    test_value = temp[3]

average_train_error = average_train_error / number_of_repeat
average_test_error = average_test_error / number_of_repeat
average_iteration = average_iteration / number_of_repeat

print(average_train_error)
print(average_test_error)
print(average_iteration)

plt.plot(test_value,linestyle ="-", marker = "x")
plt.plot(test2_y,linestyle ="-",marker = "o")
plt.show()


print("AAA")