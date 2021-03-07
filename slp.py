#! /usr/bin/env python3

from argparse import ArgumentParser
from random import sample, uniform
from math import exp

import numpy as np
from mnist import MNIST
import matplotlib.pyplot as plt


class SingleLayerPerceptron:
    def __init__(self, train_X=None, train_D=None, test_X=None, test_D=None,
                 W=None, ntrain=500, ntest=100, iterations=20, dims=None,
                 learning_rate=0.01, bias=None, threshold=None, epsilon=1.0e-7,
                 activation_function=None, dataset="digits", complete=False, zero=False):
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
        if activation_function:
            self.f = activation_function
        else:
            self.f = lambda z: 1/(1+exp(-z))
        self.zero = zero
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
        self.letters = {n: {'correct': 0, 'incorrect': 0} for n in range(self.dims[1])}

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

    def _init_weights(self, W=None):
        if W is not None:
            return W
        if self.zero:
            return np.zeros(self.dims)
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
                self.letters[d]['correct'] += 1
            else:
                self.letters[d]['incorrect'] += 1
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
        self.letters = {n: {'correct': 0, 'incorrect': 0} for n in range(self.dims[1])}

    def plot_mse(self):
        if self.mean_squared_errors == []:
            self.train()
        plt.plot(self.mean_squared_errors, label="mean squared error")
        plt.title("mean squared error vs. iterations")
        plt.legend()
        plt.show()

    def plot_accuracy(self):
        if self.accuracy == []:
            self.test_iterations()
        plt.plot(self.accuracy, label="accuracy")
        plt.title("accuracy vs. iterations")
        plt.legend()
        plt.show()

    def plot_letters(self):
        self.reset()
        self.train()
        _ = self.test()
        correct = [self.letters[n]['correct'] for n in range(self.dims[1])]
        incorrect = [self.letters[n]['incorrect'] for n in range(self.dims[1])]
        plt.bar(np.arange(len(correct)), correct, label='correct')
        plt.bar(np.arange(len(incorrect)), incorrect, bottom=correct, label='incorrect')
        plt.title("accuracy per character")
        plt.legend()
        plt.show()


def sigmoid(z):
    return 1/(1 + exp(-z))


def relu(z):
    return max(0, z)


def tanh(z):
    return (exp(z) - exp(-z))/(exp(z) + exp(-z))


def heaviside(z):
    if z > 0:
        return 1
    return 0


def linear(z):
    return z


def setup_argparser():
    parser = ArgumentParser(description="Single Layer Perceptron")
    parser.add_argument("--train",
                        help="number of training examples",
                        type=int,
                        dest="ntrain",
                        default=500,
                        required=False)
    parser.add_argument("--test",
                        help="number of testing examples",
                        type=int,
                        dest="ntest",
                        default=100,
                        required=False)
    parser.add_argument("-f", "--activation-function",
                        help="activation function",
                        choices=["sigmoid", "linear", "relu", "tanh", "heaviside"],
                        dest="f",
                        default="sigmoid",
                        required=False)
    parser.add_argument("-d", "--dataset",
                        help="EMNIST dataset",
                        choices=["balanced", "bymerge", "byclass", "mnist", "letters", "digits"],
                        default="digits",
                        dest="dataset",
                        required=False)
    parser.add_argument("-c", "--complete",
                        help="entire selected dataset",
                        dest="complete",
                        action="store_true",
                        required=False)
    parser.add_argument("-l", "--learning-rate",
                        help="learning rate",
                        type=float,
                        dest="learning_rate",
                        default=0.01,
                        required=False)
    parser.add_argument("-b", "--bias",
                        help="bias term",
                        type=float,
                        dest="bias",
                        default=uniform(-1, 1),
                        required=False)
    parser.add_argument("-z", "--zero-weights",
                        help="zero initial weights",
                        dest="zero",
                        action="store_true",
                        required=False)
    parser.add_argument("-e", "--epsilon",
                        help="error epsilon",
                        type=float,
                        dest="epsilon",
                        default=1.0e-7,
                        required=False)
    parser.add_argument("-i", "--iterations",
                        help="number of iterations",
                        type=int,
                        dest="iterations",
                        default=20,
                        required=False)
    return parser


if __name__ == '__main__':
    args = setup_argparser().parse_args()
    if args.f == "linear":
        f = linear
    elif args.f == "tanh":
        f = tanh
    elif args.f == "relu":
        f = relu
    elif args.f == "heaviside":
        f = heaviside
    else:
        f = sigmoid
    slp = SingleLayerPerceptron(ntrain=args.ntrain,
                                ntest=args.ntest,
                                iterations=args.iterations,
                                epsilon=args.epsilon,
                                learning_rate=args.learning_rate,
                                bias=args.bias,
                                complete=args.complete,
                                zero=args.zero,
                                dataset=args.dataset,
                                activation_function=f)

    slp.plot_mse()
    slp.reset()
    slp.plot_accuracy()
    slp.plot_letters()
