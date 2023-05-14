from numpy.random import shuffle
import numpy as np
from sklearn import preprocessing

def datasetInfo(dataset):
    """
    Takes the dataset and returns the following statistics as a dictionary:
        rows: number of rows in the dataset
        columns: number of columns in the dataset
        benign: Number of benign entries in dataset
        malignant: Number of malignant entries in dataset
    """
    # return a dictionary
    ret = {}

    ret["rows"] = len(dataset)
    ret["columns"] = len(dataset.iloc[0])
    ret["benign"] = sum(dataset["diagnosis"] == "B")
    ret["malignant"] = sum(dataset["diagnosis"] == "M")
    return ret


def splitDataset(dataset, test_percentage=20):
    le = preprocessing.LabelEncoder()
    """
    Takes the dataset as a dataframe

    Shuffles data, then returns 4 subsets of the dataset as numpy matrices:
        Training data without labels (100-test_percentage percent of the data)
        Training data labels (column vector!)
        Testing data without labels (test_percentage percent of the data)
        Testing data labels (column vector!)
    """
    # first, extract the data into numpy from pandas
    mat = dataset.values
    print(mat)
    # now shuffle
    shuffle(mat)
    # now slice and dice!
    training_data = mat[int((test_percentage/100)*len(mat)):]
    testing_data = mat[:int((test_percentage/100)*len(mat))]
    # split out labels
    training_labels = training_data[ : , : 1 ]
    training_labels = training_labels.reshape(len(training_labels))
    #print(training_data_labels)
    #print(training_labels)
    training_data = training_data[ : , 1 : ]
    testing_labels = testing_data[ : , : 1 ]
    testing_labels = testing_labels.reshape(len(testing_labels))
    #print(testing_data_labels)
    #print(testing_labels)
    testing_data = testing_data[ : , 1 : ]
    return training_data, training_labels, testing_data, testing_labels




