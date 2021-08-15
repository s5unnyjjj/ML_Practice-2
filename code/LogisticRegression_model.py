
import numpy as np

def sigmoid(x):
    sig_x = np.zeros(x.shape)
    sig_x = 1 / (1 + np.exp(-x))

    return sig_x

def logistic_regression_cost(theta, X, y):
    n = len(y)
    z = X.dot(theta)
    h = sigmoid(z)
    J = (y * np.log(h) + (1-y) * np.log(1-h))
    sum = 0
    for i in range(n):
        sum += (y[i] * np.log(h[i]) + (1 - y[i]) * np.log(1 - h[i]))
    J = -1 * (sum/n)

    return J


def logistic_regression_gradient(theta, X, y):
    n = len(y)
    grad = np.zeros(theta.shape)
    z = X.dot(theta)
    h = sigmoid(z)
    grad = (1/n) * np.dot(X.transpose(), sigmoid(X.dot(theta)) - y)

    return grad


def logistic_regression_predict(theta, X):
    n = X.shape[0]
    preds = np.zeros(n)

    z = X.dot(theta)
    h = sigmoid(z)
    for i in range(n):
       if h[i] >= 0.5:
           preds[i] = 1
       else:
           preds[i] = 0

    return preds

