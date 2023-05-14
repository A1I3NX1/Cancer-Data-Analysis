from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


def SKLearnKnnClassifier(training, testing, training_labels, testing_labels, k):
    # instantiate model
    knn = KNeighborsClassifier(n_neighbors=k)
    

    # fit model to training data
    knn.fit(training, training_labels)
    # predict test labels
    predicted_labels = knn.predict(testing)
    # return % where prediction matched actual
    success = sum(predicted_labels == testing_labels) / len(testing_labels)
    return (success)


def SKLearnSVMClassifier(training, testing, training_labels, testing_labels):
    # instantiate model
    svm = SVC(kernel="rbf")

    # fit model to training data
    svm.fit(training, training_labels)
    # predict test labels
    predicted_labels = svm.predict(testing)
    # return % where prediction matched actual
    success = sum(predicted_labels == testing_labels) / len(testing_labels)
    return(success)