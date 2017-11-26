# -*- coding: utf-8 -*-
"""
Created on Sat Nov 25 14:53:21 2017

@author: Андрій
"""

import matplotlib.pyplot as plt
import numpy as np
import random
from sklearn.datasets import load_boston

def linierRegression(features, weights):
    """
    Performs simple linier regression
    """
    return np.dot(features, weights)

def regressionGradient(x, h, y, m):
    x_trans = x.transpose()
    deviation = h - y
    return np.dot(x_trans, deviation) / m

def squareloss(guess, correct):
    diff = guess - correct
    cost = np.multiply(diff, diff)
    return np.average(cost)

def loss(guess, correct):
    diff = guess - correct
    return np.average(np.abs(diff))

def generate_samples(m):
    """
    Generates sample data (input and output)
    Supervised learning.
    """
    b = random.randint(100, 500)
    x_data = np.array([np.ones(m), np.arange(m)])
    y_data = np.array([i + b + random.random() * 50 for i in range(m)])
    return x_data.transpose(), y_data

def gradient_descent(x, y, alpha, theta, m, iterations):
    """
    Gradient descent implementation for linear regression.
    """
    for i in range(iterations):
        h = linierRegression(x, theta)
        gradient = regressionGradient(x, h, y, m)
        theta = theta - alpha * gradient
        if i%100 == 0:
            print("iter: {0} sqrloss: {1} loss: {2}".format(i,squareloss(h,y), loss(h,y)))
    return theta

#load data with shuffle
dataset = load_boston()
y = np.array(dataset.target)
m = len(y)
y = y.reshape(m,1)
x = np.ones(m).reshape(m,1)
tmp = dataset.data
x = np.concatenate((x, dataset.data, y),  axis=1)
np.random.shuffle(x)
y = x[:,-1:]
x = x[:,:-1]


#x, y = generate_samples(100)

alpha = 0.000005

# Blue: samples
#plt.plot(x[:,1], y, 'bo')

# Initial theta.
theta = np.ones((x.shape[1],1))
#divide train/test 80/20
train_size = int(x.shape[0]*0.8)
# Descent down
x_train = x[:train_size,:]
y_train = y[:train_size,:]
x_test = x[train_size:,:]
y_test = y[train_size:,:]
theta = gradient_descent(x_train, y_train, alpha, theta, m, 40000)
#Let's get test set accuracy
h = linierRegression(x_test, theta)
L1 = loss(h, y_test)
L2 = squareloss(h, y_test)
print("loss: {0}, sqrloss: {1}".format(L1, L2))

# Generate our best hypothesis line and plot
h = np.dot(x, theta)
#plt.plot(x[:,1], h, 'ro')

# Show all plots
plt.show()
