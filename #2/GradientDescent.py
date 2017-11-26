# -*- coding: utf-8 -*-
"""
Created on Sat Nov 25 14:53:21 2017

@author: Андрій
"""

import matplotlib.pyplot as plt
import numpy as np
import random

def linierRegression(x, theta):
    """
    Performs simple linier regression
    """
    return np.dot(x, theta)

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
        if i%1000 == 0:
            print("iter: {0} sqrloss: {1} loss: {2}".format(i,squareloss(h,y), loss(h,y)))
    return theta

m = 100
x, y = generate_samples(m)
alpha = 0.0005

# Blue: samples
plt.plot(x[:,1], y, 'bo')

# Initial theta.
theta = np.array([1, 1])
# Descent down
theta = gradient_descent(x, y, alpha, theta, m, 40000)

# Generate our best hypothesis line and plot
h = np.dot(x, theta)
plt.plot(x[:,1], h, 'ro')

# Show all plots
plt.show()
