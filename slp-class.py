#! /usr/bin/env python3

from random import sample, uniform
from math import exp

import numpy as np
from mnist import MNIST
import matplotlib.pyplot as plt


class SingleLayerPerceptron:
    def __init__(self, train_X=None, train_D=None, test_X=None, test_D=None,
                 W=None, ntrain=500, ntest=100, iterations=20, dims=None,
                 learning_rate=0.01, bias=None, threshold=None, epsilon=1.0e-10,
                 activation_function="sigmoid", dataset="digits", complete=False):
        if dataset in {"mnist", "balanced", "digits", "bymerge", "letters", "byclass"}:
            self.dataset = dataset
            self.dims = dims
        else:
            self.dataset = "digits"
            self.dims = (784, 10)
            
        if all((train_X, train_D)):
            self.dims = (len(train_X[0]), len(set(train_D)))
            self.train_X = np.array([self._normalize(x) for x in train_X])
            self.train_D = np.array([self._onehot(d) for d in train_D])
        else:
            self.train_X, tD = self._get_data(k=ntrain, train=True, complete=complete)
            if dataset == "letters":
                tD = np.subtract(tD, 1)
            self.train_D = np.array([self._onehot(d) for d in tD])
            
        if all((test_X, test_D)):
            self.test_X = np.array([self._normalize(x) for x in test_X])
            self.test_D = test_D
        else:
            self.test_X, self.test_D = self._get_data(k=ntest, complete=complete)
            if dataset == "letters":
                self.test_D = np.subtract(self.test_D, 1)
        
        self.threshold = threshold
        if self.threshold:
            self.train_X = np.array([self._binary(x) for x in self.train_X])
            self.test_X = np.array([self._binary(x) for x in self.test_X])

        self.activation_function = activation_function.lower().strip()
        if self.activation_function == "relu":
            self.f = self._relu
        elif self.activation_function == "tanh":
            self.f = self._tanh
        elif self.activation_function == "linear":
            self.f = self._linear
        elif self.activation_function == "heaviside":
            self.f = self._heaviside
        else:
            self.f = self._sigmoid
            self.activation_function = "sigmoid"
            
        self.W = self._init_weights(W)
        self.learning_rate = learning_rate
        if bias:
            self.bias = bias
        else:
            self.bias = uniform(-1, 1)
        self.epsilon = epsilon
        self.iterations = iterations
        self.mean_squared_errors = []
        self.accuracy = []


    def _get_data(self, k, train=False, complete=False):
        if self.dataset == "digits":
            mndata = MNIST("./data")
        else:
            mndata = MNIST("./emnist_data/")
            mndata.gz = True
            mndata.select_emnist(self.dataset)
        if train:
            images, labels = mndata.load_training()
        else:
            images, labels = mndata.load_testing()
        self.dims = (len(images[0]), len(set(labels)))
        if complete:
            return (np.array([self._normalize(i) for i in images]),
                    np.array(labels))
        indices = sample([n for n in range(k)], k=k)
        return (np.array([self._normalize(images[i]) for i in indices]),
                np.array([labels[i] for i in indices]))


    def _normalize(self, X):
        mn = min(X)
        mx = max(X)
        return np.array([(x - mn) / (mx - mn) for x in X])


    def _binary(self, x):
        b = np.zeros(x.shape)
        for i in range(len(b)):
            if x[i] > self.threshold:
                b[i] = 1
        return b


    def _onehot(self, n):
        vec = np.zeros(self.dims[1])
        vec[n] = 1
        return vec


    def _sigmoid(self, z):
        return 1/(1 + exp(-z))


    def _relu(self, z):
        return max(0, z)


    def _tanh(self, z):
        return (exp(z) - exp(-z))/(exp(z) + exp(-z))


    def _heaviside(self, z):
        if z > 0:
            return 1
        return 0


    def _linear(self, z):
        return z


    def _init_weights(self, W=None):
        if W is not None:
            return W
        return np.random.random(self.dims)


    def _apply_weights(self, w, x):
        return np.dot(np.transpose(w), x) + self.bias


    def _activate(self, z):
        return self.f(z)


    def _error(self, w, x, d):
        z = self._apply_weights(w, x)
        yhat = self._activate(z)
        return d - yhat


    def _update (self, w, x, error):
        return w + self.learning_rate * error * x


    def _epoch(self, X, D):
        errors = []
        for n in range(self.dims[1]):
            for x, d in zip(X, D):
                error = self._error(self.W[:, n], x, d[n])
                self.W[:, n] = self._update(self.W[:, n], x, error)
                errors.append(error)
        return (sum(errors)/len(errors))**2


    def train(self):
        self.mean_squared_errors = []
        for i in range(self.iterations):
            mse = self._epoch(self.train_X, self.train_D)
            self.mean_squared_errors.append(mse)
            print(f"MSE iteration {i}: {mse}")
            if mse < self.epsilon:
                print(f"Reached epsilon {self.epsilon} at {i} iterations")
                break

    def test(self):
        correct = 0
        for x, d in zip(self.test_X, self.test_D):
            y = np.dot(np.transpose(self.W), x)
            if np.argmax(y) == d:
                correct += 1
        return correct / len(self.test_X)

    def test_iterations(self):
        self.accuracy = []
        for i in range(self.iterations):
            _ = self._epoch(self.train_X, self.train_D)
            accuracy = self.test()
            print(f"Accuracy at iteration {i}: {accuracy}")
            self.accuracy.append(accuracy)

    def reset(self):
        self.W = self._init_weights()
        self.accuracy = []
        self.mean_squared_errors = []

    def plot_mse(self):
        if self.mean_squared_errors == []:
            self.train()
        plt.plot(self.mean_squared_errors)
        plt.show()

    def plot_accuracy(self, W=None):
        if self.accuracy == []:
            self.test_iterations()
        plt.plot(self.accuracy)
        plt.show()
