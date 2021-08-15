
import numpy as np

def perceptron(X, theta):
    preds = np.zeros(X.shape[0])
    for i in range(len(X)):
        h = X[i].dot(theta)
        if h > 0:
            preds[i] = 1
        else:
            preds[i] = 0

    return preds


def perceptron_cost(X, y, theta):
    n = len(y)
    percep = perceptron(X, theta)
    sum = 0
    for i in range(n):
        h = X[i].dot(theta)
        sum += (y[i] - percep[i]) * h
    J = (-sum) / n

    return J


def update_perceptron(X, y, theta, alpha=0.03):
    n = len(y)
    percep = perceptron(X, theta)
    sum = 0
    for i in range(n):
        sum += np.dot((y[i] - percep[i]), X[i])
    theta = theta + ((alpha/n) * sum)
    return theta



