#! /usr/bin/env python3

import sys
from random import randint, sample, random
from math import e

import numpy as np
from mnist import MNIST
import matplotlib.pyplot as plt

LR = 0.01
B = random()
T = 0
EPS = 1.0e-15

def get_data(k=500):
    mndata = MNIST("./data")
    images, labels = mndata.load_training()
    indices = sample([n for n in range(len(images))], k=k)
    return (np.array([normalize(images[i]) for i in indices]),
            np.array([labels[i] for i in indices]))

def normalize(X):
    mn = min(X)
    mx = max(X)
    return np.array([(x-mn)/(mx-mn) for x in X])

def sigmoid(z):
    return 1/(1 + e**(-z))

def relu(z):
    return max(0, z)

def tanh(z):
    return (e**(z) - e**(-z))/(e**(z) + e**(-z))

def initialize_weights(shape):
    return np.random.random(shape)

def activation(W, x, f=sigmoid):
    z = np.dot(np.transpose(W), x) + B
    return f(z)

def onehot(n):
    ohv = np.zeros(10)
    ohv[n] = 1
    return ohv

def update(W, x, d, f):
    yhat = activation(W, x, f)
    return (W + LR*(d - yhat)*x, d-yhat)

def binary_image(x):
    b = np.zeros((784,))
    for i in range(len(b)):
        if x[i] > T:
            b[i] = 1
    return b

def train(X, D, W=None, k=500, epochs=10, f=sigmoid):
    #X, D = get_data(k=k)
    #D = [onehot(d) for d in D]
    #X = [binary_image(x) for x in X]
    if W is None:
        W = initialize_weights((len(X[0]), 10))
    err = 0
    errors = []
    mserrors = []
    for i in range(epochs):
        for n in range(10):
            for x, d in zip(X, D):
                W[:, n], err = update(W[:, n], x, d[n], f)
                errors.append(err)
        mse = (sum(errors)/k)**2
        print(f"MSE Iteration {i}: {mse}")
        mserrors.append(mse)
        if mse < EPS:
            break
        errors = []
    return W, mserrors

def test(X, D, W, k=100):
    #X, D = get_data(k=k)
    correct = 0

    for x, d in zip(X, D):
        y = np.dot(np.transpose(W), x)
        if np.argmax(y) == d:
            correct += 1

    return correct / k

def n_tests(k=500, iterations=20, f=sigmoid):
    X, D = get_data(k=k)
    tX, tD = get_data(k=100)
    Doh = [onehot(d) for d in D]
    #X = [binary_image(x) for x in X]
    accuracy = []
    W = None
    for i in range(iterations):
        W, _ = train(X, Doh, W=W, k=k, epochs=1, f=f)
        accuracy.append(test(tX, tD, W, k=1000))
    plt.plot(accuracy)
    plt.show()
