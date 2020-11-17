import numpy as np
import pandas as pd
from tree import DecisionTreeClassifier, Criterion



class RandomForest:
    def __init__(self, attr_headers, continuous_attr_header, criterion, data, n_estimators=100):
        self.criterion = criterion
        self.n_estimators = n_estimators
        self.attr_headers = attr_headers
        self.contionuous_attr_headers = continuous_attr_header
        self.estimators = []
        self.index_attr = []
        self.attr_values = self.set_attribute_values(data)

    def set_attribute_values(self, data):
        return [np.unique(data[:, i]) for i in range(data.shape[1])]

    def fit(self, X, Y):

        n_attr = X.shape[1]
        n_data = X.shape[0]
        m = int(np.sqrt(n_attr))

        for estimator in range(self.n_estimators):
            index_attr = np.array(range(n_attr))
            np.random.seed(estimator)
            np.random.shuffle(index_attr)
            index_attr = index_attr[:m]
            self.index_attr.append(index_attr)

            #bagging
            index_data = np.random.choice(n_data, int(n_data/10), replace=True)

            x = X[index_data, :][:, index_attr]
            y = Y[index_data]
            t = self.attr_headers[index_attr]
            model = DecisionTreeClassifier(self.attr_headers[index_attr], self.contionuous_attr_headers, self.criterion)
            t = np.array(self.attr_values)[index_attr]
            model.set_labels(np.array(self.attr_values)[index_attr])
            model.fit(x, y)
            self.estimators.append(model)

    def predict_single(self, X):
        predictions = np.array([])

        for i in range(len(self.estimators)):
            #if len(X.shape) == 1:
            x = X[self.index_attr[i]]
            prediction = np.array([self.estimators[i].predict(x)])
            # else:
            #     x = X[:, self.index_attr[i]]
            #     prediction = self.estimators[i].predict(x)

            predictions = np.hstack((predictions, prediction))

        classes, count = np.unique(predictions, return_counts=True)
        return classes[np.argmax(count)]

    def predict(self, X):
        if len(X.shape) == 1:
            return self.predict_single(X)
        else:
            return np.array([self.predict_single(x) for x in X])

