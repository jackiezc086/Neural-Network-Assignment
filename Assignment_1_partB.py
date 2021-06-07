import tkinter as tk
from tkinter import filedialog
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def forwardPropa(input, weight, bias):
    return sigmoid(np.dot(input,weight)+bias)

# Load data file(.csv)
root = tk.Tk()
root.withdraw()
file_path = filedialog.askopenfilename()
xy = np.loadtxt(file_path, delimiter=',', dtype=np.float32)

repeat_time = 100000

# Set input & output value for training
x_data = xy[0:-5, 0:-5]
y_data = xy[0:-5, -5:]

# Set input & output value for test
x_data_t = xy[-5:, 0:-5]
y_data_t = xy[-5:, -5:]

# Set weight and bias for hidden layer
h_weight = np.random.rand(64, 10)
h_bias = np.zeros((1, 10))
# h_bias = np.ones((1, 10))

h2_weight = np.random.rand(10, 10)
h2_bias = np.zeros((1, 10))
# h2_bias = np.ones((1, 10))

# Set weight and bias for output layer
o_weight = np.random.rand(10, 5)
o_bias = np.zeros((1, 5))
# o_bias = np.ones((1, 5))

# Set leaning rate
l_rate = 0.01

ep = np.zeros(repeat_time)

# Training
for i in range(repeat_time):
    for x, t in zip(x_data,y_data):
        inputX = x.reshape(1,-1)
        targetY = t.reshape(1,-1)

        # Forwardpropagation
        h = forwardPropa(inputX,h_weight, h_bias)
        h2 = forwardPropa(h, h2_weight, h2_bias)
        y = forwardPropa(h2,o_weight, o_bias)

        ep[i] = mse = ((targetY - y) ** 2).mean(axis=None)

        # Backpropagation
        o_delta = -(targetY-y)*(y*(1-y))
        o_weight -= l_rate * np.dot(h2.T, o_delta)
        o_bias -= l_rate * o_delta

        h2_delta = o_delta.dot(o_weight.T) * (h2 * (1 - h2))
        h2_weight -= l_rate * np.dot(h.T, h2_delta)
        h2_bias -= l_rate * h2_delta

        h_delta = h2_delta.dot(h2_weight.T) * (h * (1 - h))
        h_weight -= l_rate * np.dot(inputX.T, h_delta)
        h_bias -= l_rate * h_delta

# Testing
correct = 0
for x, t in zip(x_data_t,y_data_t):
    inputX = x.reshape(1,-1)
    targetY = t.reshape(1,-1)
    h_test = sigmoid(np.dot(inputX,h_weight)+h_bias)
    h2_test = sigmoid(np.dot(h_test, h2_weight) + h2_bias)
    y_test = sigmoid(np.dot(h2_test,o_weight)+o_bias)
    y_test = np.argmax(y_test)
    targetY = np.argmax(targetY)
    print("Prediction: {:1}".format(y_test), "\t Actual: {:1}".format(targetY))
    if y_test == targetY:
        correct += 1

plt.plot(ep)
plt.ylabel("MSE")
plt.xlabel("number of epochs")
plt.title("Learning Curve for Assignment 1 PartB")
plt.show()

print("Accuracy : {:3}".format(correct / len(y_data_t) * 100))