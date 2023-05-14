import numpy as np
from math import sqrt
from statistics import mode


def NNClassifier(training, testing, training_labels, testing_labels, k):
    # preallocate labels
    predicted_labels = np.zeros(len(testing_labels)).astype(str)

    for i in range(len(predicted_labels)):
       predicted_labels[i] = knn(training, training_labels, testing[i], k)

    # return % where prediction matched actual
    success = sum(predicted_labels == testing_labels) / len(testing_labels)
    return(success)


def knn(data, data_labels, vector, k):
    # preallocate distance array
    distances = np.zeros(len(data_labels))

    # calculate distances
    for i in range(len(distances)):
        distances[i] = sqrt(((data[i].astype(float)-vector.astype(float)) ** 2).sum())
    # set labels
    indices = np.argsort(distances)

    # take vote amongs top labels
    to_vote = data_labels[indices]
    
    return mode(to_vote[:k])