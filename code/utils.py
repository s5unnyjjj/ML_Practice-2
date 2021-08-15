
import matplotlib.pyplot as plt
import numpy as np
import pandas

from LogisticRegression_model import *
from Perceptron_model import *
from KNN_model import *

def plotData(data):
    fig, ax = plt.subplots(figsize=(9, 5))
    results_accepted = data[data.accepted == 1]
    results_rejected = data[data.accepted == 0]
    ax.scatter(results_accepted.test1, results_accepted.test2, marker='+', c='b', s=40)
    ax.scatter(results_rejected.test1, results_rejected.test2, marker='o', c='r', s=30)
    ax.set_ylim([20, 105])
    ax.legend(['Accepted', 'Not acceptted'], loc='best')
    ax.grid(True)
    ax.set_xlabel('Test 1 score', fontsize=14)
    ax.set_ylabel('Test 2 score', fontsize=14)
    plt.setp(ax.get_xticklabels(), fontsize=12)
    plt.setp(ax.get_yticklabels(), fontsize=12)
    return ax


def plotData_annotate(data, X, result_knn):
    ax = plotData(data)
    i = 0
    for xy in zip(X[:, 0], X[:, 1]):
        ax.annotate('(%s)' % int(result_knn[i]), xy=xy, textcoords='data', size=8)
        i += 1

    return ax


def plotResult(data, X, optimal_theta, initial_theta):
    ax = plotData(data)
    x_plot = np.array([np.max(X[:, 1]), np.min(X[:, 1])])
    y_plot = (-optimal_theta[0] - optimal_theta[1] * x_plot) / (optimal_theta[2])
    plt_1 = ax.plot(x_plot, y_plot, 'k--')
    y_plot_initial = (-initial_theta[0] - initial_theta[1] * x_plot) / (initial_theta[2])
    plt_2 = ax.plot(x_plot, y_plot_initial, 'k:')

    optimal_theta = optimal_theta / optimal_theta[2]
    initial_theta = initial_theta / initial_theta[2]
    ax.legend(['Final: ${:.2f} + {:.2f} x_1 + {:.2f} x_2 = 0$'.format(*list(optimal_theta)),
               'Initial: ${:.2f} + {:.2f} x_1 + {:.2f} x_2 = 0$'.format(*list(initial_theta))],
              loc='best')
    plt.show()


def plotResult_comparison(data, X, optimal_theta1, optimal_theta2):
    ax = plotData(data)
    x_plot = np.array([np.max(X[:, 1]), np.min(X[:, 1])])
    y_plot = (-optimal_theta1[0] - optimal_theta1[1] * x_plot) / (optimal_theta1[2])
    plt_1 = ax.plot(x_plot, y_plot, 'k--')
    y_plot2 = (-optimal_theta2[0] - optimal_theta2[1] * x_plot) / (optimal_theta2[2])
    plt_2 = ax.plot(x_plot, y_plot2, 'k:')

    optimal_theta1 = optimal_theta1 / optimal_theta1[2]
    optimal_theta2 = optimal_theta2 / optimal_theta2[2]

    ax.legend(['Logistic regression: ${:.2f} + {:.2f} x_1 + {:.2f} x_2 = 0$'.format(*list(optimal_theta1)),
               'Perceptron: ${:.2f} + {:.2f} x_1 + {:.2f} x_2 = 0$'.format(*list(optimal_theta2))],
              loc='best')
    plt.show()


def plot_accuracy_k(accuracy_from_dif_k, mode):
    plt.figure(figsize=(8, 5))
    plt.plot(accuracy_from_dif_k, linewidth=2.0)
    plt.ylabel(mode + ' accuracy', fontsize=15)
    plt.xlabel('k', fontsize=15)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.show()


def vis_decision_boundary(x_tra, y_tra, k, typ='k--'):
    ax = plt.gca()

    lim0 = plt.gca().get_xlim()
    lim1 = plt.gca().get_ylim()

    x_ = np.linspace(lim0[0], lim0[1], 100)
    y_ = np.linspace(lim1[0], lim1[1], 100)
    xx, yy = np.meshgrid(x_, y_)
    pred = predictKNN(np.concatenate([xx.ravel()[:, None], yy.ravel()[:, None]], axis=1), x_tra, y_tra, k)
    ax.contourf(xx, yy, pred.reshape(xx.shape), cmap=plt.cm.coolwarm, alpha=0.4)

    ax.set_xlim(lim0)
    ax.set_ylim(lim1)
    ax.set_title('k= ' + str(k), fontsize=15, fontweight='bold')


def plotData_sub(data, i):
    # plt.figure(figsize=(8,5))
    fig, ax = plt.subplots(figsize=(9, 5))
    ax = plt.subplot(1, 4, i)
    results_accepted = data[data.accepted == 1]
    results_rejected = data[data.accepted == 0]
    ax.scatter(results_accepted.test1, results_accepted.test2, marker='+', c='b', s=40)
    ax.scatter(results_rejected.test1, results_rejected.test2, marker='o', c='r', s=30)
    ax.set_ylim([20, 105])
    ax.legend(['Accepted', 'Not acceptted'], loc='best')
    ax.grid(True)
    ax.set_xlabel('Test 1 score', fontsize=14);
    ax.set_ylabel('Test 2 score', fontsize=14);
    plt.setp(ax.get_xticklabels(), fontsize=12);
    plt.setp(ax.get_yticklabels(), fontsize=12);
    return ax


def boundary_plot(data1, X, y, k_arr):
    for i in k_arr:
        plotData(data1)
        vis_decision_boundary(X, y, i)
        plt.show()
        plt.tight_layout


def vis_decision_boundary_contour(data, X, w, typ='k--'):
    plotData(data)
    lim0 = plt.gca().get_xlim()
    lim1 = plt.gca().get_ylim()
    x_ = np.linspace(lim0[0], lim0[1], 100)
    y_ = np.linspace(lim1[0], lim1[1], 100)
    xx, yy = np.meshgrid(x_, y_)

    x_tra_ = np.concatenate([xx.ravel()[:, None], yy.ravel()[:, None]], axis=1)
    x_tra_ = np.insert(x_tra_, 0, np.ones(len(x_tra_)), 1)
    y = np.dot(x_tra_, w)
    pred = sigmoid(y)
    levels = np.array([0.0, 0.15, 0.3, 0.4, 0.5, 0.7, 0.85, 1.0])
    plt1 = plt.contourf(xx, yy, pred.reshape(xx.shape), cmap=plt.cm.coolwarm, alpha=0.4, levels=levels)
    plt.colorbar(plt1)

    plt.gca().set_xlim(lim0)
    plt.gca().set_ylim(lim1)