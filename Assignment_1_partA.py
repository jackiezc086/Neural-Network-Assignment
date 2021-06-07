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

repeat_time = 15000

# Set input & output value
x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]

# Set weight and bias for hidden layer
h_weight = np.random.rand(2, 2)
h_bias = np.zeros((1, 2))
# h_bias = np.ones((1, 2))

# Set weight and bias for output layer
o_weight = np.random.rand(2, 1)
o_bias = np.zeros((1, 1))
# o_bias = np.ones((1, 1))

# Set leaning rate
l_rate = 0.1

ep = np.zeros(repeat_time)

# Training
for i in range(repeat_time):
    for x, t in zip(x_data,y_data):
        inputX = x.reshape(1,-1)
        targetY = t.reshape(1,-1)

        # Forwardpropagation
        h = forwardPropa(inputX,h_weight, h_bias)
        y = forwardPropa(h,o_weight, o_bias)
        ep[i] = 1/2*pow((targetY-y),2)

        # Backpropagation
        o_delta = -(targetY-y)*(y*(1-y))
        o_weight -= l_rate * np.dot(h.T, o_delta)
        o_bias -= l_rate * o_delta

        h_delta = o_delta.dot(o_weight.T) * (h*(1-h))
        h_weight -= l_rate * np.dot(inputX.T, h_delta)
        h_bias -= l_rate * h_delta

# Testing
correct = 0
for x, t in zip(x_data,y_data):
    inputX = x.reshape(1,-1)
    targetY = t.reshape(1,-1)
    h_test = sigmoid(np.dot(inputX,h_weight)+h_bias)
    y_test = sigmoid(np.dot(h_test,o_weight)+o_bias)
    y_test = np.round(y_test)
    print("Prediction: \t",y_test, "\nActual: \t",targetY)
    if y_test == targetY:
        correct += 1

print("Accuracy : {:3}".format(correct / len(y_data) * 100))
plt.plot(ep)
plt.ylabel("MSE")
plt.xlabel("number of epochs")
plt.title("Learning Curve for Assignment 1 PartA")
plt.show()
print("done")
# h = sigmoid(np.dot(x_data,h_weight)+h_bias)
# y = sigmoid(np.dot(h,o_weight)+o_bias)
# # y = np.round(y)
# # print(y)
