
import time
import scipy.optimize
import scipy.spatial.distance
from Perceptron_model import *
from LogisticRegression_model import *
from KNN_model import *
from utils import *

def func_perceptron(x, y, train_data):
    initial_theta = np.array([-400., 5., 1.])
    print('\n### Check ###')
    print('Initial theta : ', initial_theta)

    cost = perceptron_cost(x, y, initial_theta)
    print('\n### Check ###')
    print('Cost at initial theta : ', cost)
    theta_updated = update_perceptron(x, y, initial_theta, alpha=0.03)
    print('\n### Check ###')
    print('Updated theta : ', theta_updated)

    perceptron_optimal_theta = initial_theta
    n_iter = 100
    alpha = 0.03
    old_cost = np.Inf
    for i in range(n_iter):
        perceptron_optimal_theta = update_perceptron(x, y, perceptron_optimal_theta, alpha)
        cost = perceptron_cost(x, y, perceptron_optimal_theta)
        if np.mod(i, 1) == 0:
            print('[Iter : {:2d}] Cost {:.4f}'.format(i, cost))
        if cost < 0.1 or cost / old_cost > 1:
            print('Converged')
            break
        old_cost = cost
    print('Training Finished')

    print('\n### Check ###')
    print('perceptron optimal theta : ', perceptron_optimal_theta)

    plotResult(train_data, x, perceptron_optimal_theta, initial_theta)

    return perceptron_optimal_theta


def func_logistic(x, y, train_data, percep_theta):
    print('\n### Check ###')
    initial_theta = np.array([-0.03, -0.01, 0.01])

    logisticCost = logistic_regression_cost(initial_theta, x, y)
    print('Cost at initial theta : ', logisticCost)

    print('\n### Check ###')
    logisticGrad = logistic_regression_gradient(initial_theta, x, y)
    print('Gradient at initial theta : ', logisticGrad)

    start = time.time()

    logistic_optimal_theta = scipy.optimize.fmin_ncg(f=logistic_regression_cost,
                                                     x0=initial_theta,
                                                     fprime=logistic_regression_gradient,
                                                     args=(x, y)
                                                     )

    stop = time.time()
    print('Time : ', stop - start)

    print('\n### Check ###')
    print('Logistic Optimal theta : ', logistic_optimal_theta)

    plotResult(train_data, x, logistic_optimal_theta, initial_theta)

    vis_decision_boundary_contour(train_data, x, logistic_optimal_theta, 'k--')

    plotResult_comparison(train_data, x, logistic_optimal_theta, percep_theta)

    return logistic_optimal_theta

def func_pred(train_x, train_y, test_x, test_y, logistic_theta):
    pred = logistic_regression_predict(logistic_theta, np.array([[1, 45, 85]]))
    print('\n### Check ###')
    print('Accepted(y=1) or not(y=0) : ', int(pred.item()))

    print('\n### Check ###')
    print('Logistic Training Accuracy : ' + str(np.mean(logistic_regression_predict(logistic_theta, train_x) == train_y)))
    print('Logistic Test Accuracy: ',
          str(np.mean(logistic_regression_predict(logistic_theta, test_x) == test_y)))


def func_knn(train_x, train_y, train_data, test_x, test_y):
    k = 3
    result_knn = predictKNN(targetX=train_x, dataSet=train_x, labels=train_y, k=k)

    ax = plotData_annotate(train_data, train_x, result_knn)

    print('\n### Check ###')
    print('K-nearest neighbors, k = ' + str(k) + ', training accuracy : ' + str(np.mean(result_knn == train_y)))

    k_arr = [1, 5, 20, 40]
    boundary_plot(train_data, train_x, train_y, k_arr)

    accuracy_from_dif_k = np.zeros((80,))

    for dif_k in range(1, 80):
        result_knn = predictKNN(targetX=train_x, dataSet=train_x, labels=train_y, k=dif_k)
        accuracy = np.mean(result_knn == train_y)
        accuracy_from_dif_k[dif_k] = accuracy

    plot_accuracy_k(accuracy_from_dif_k, 'training')

    accuracy_from_dif_k = np.zeros((80,))

    for dif_k in range(1, 80):
        result_knn = predictKNN(targetX=test_x, dataSet=train_x, labels=train_y, k=dif_k)
        accuracy = np.mean(result_knn == test_y)
        accuracy_from_dif_k[dif_k] = accuracy

    plot_accuracy_k(accuracy_from_dif_k, 'test')


if __name__ == "__main__":
    trainData = pandas.read_csv("data_train.txt", header=None, names=['test1', 'test2', 'accepted'])
    testData = pandas.read_csv("data_test.txt", header=None, names=['test1', 'test2', 'accepted'])

    ax = plotData(trainData)

    trainX = trainData[['test1', 'test2']].values
    trainY = trainData.accepted.values
    after_trainX = np.insert(trainX, 0, np.ones(len(trainX)), 1)

    testX = testData[['test1', 'test2']].values
    testY = testData.accepted.values
    after_testX = np.insert(testX, 0, np.ones(len(testX)), 1)

    percepTheta = func_perceptron(after_trainX, trainY, trainData)

    logisticTheta = func_logistic(after_trainX, trainY, trainData, percepTheta)

    func_pred(after_trainX, trainY, after_testX, testY, logisticTheta)

    func_knn(trainX, trainY, trainData, testX, testY)