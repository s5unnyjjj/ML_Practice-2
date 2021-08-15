
import numpy as np

def euclideanDistance(targetX, dataSet):
    m = dataSet.shape[0]
    distances = np.zeros(m)

    dist = 0.0
    for i in range(m):
        dist += (targetX[i] - dataSet[i])**2

    distances = np.sqrt(dist)

    return distances


def getKNN(targetX, dataSet, labels, k):
    closest_data = np.zeros(k)

    distance = list()
    for i in range(len(dataSet)):
        dist = euclideanDistance(targetX, dataSet[i])
        distance.append((i, dist))

    distance.sort(key=lambda x: x[1])

    for j in range(k):
        closest_data[j] = distance[j][0]
    return closest_data


def predictKNN(targetX, dataSet, labels, k):
    m = targetX.shape[0]
    predicted_array = np.zeros((m,))

    num_zero = 0
    num_one = 0
    value_res = 0
    index = 0
    for i in range(m):
        value = getKNN(targetX[i], dataSet, labels, k)
        for i in range(k):
            idx = int(value[i])
            if labels[idx] == 0:
                num_zero += 1
            elif labels[idx] == 1:
                num_one += 1
        if num_zero > num_one:
            value_res = 0
        elif num_zero < num_one:
            value_res = 1
        predicted_array[index] = value_res
        index += 1
        num_zero = 0
        num_one = 0
    return predicted_array