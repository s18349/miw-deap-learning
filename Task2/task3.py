import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from collections import Counter
from matplotlib.colors import ListedColormap
from logistic_regression_gd import LogisticRegressionGD


class BinaryMultiClassifier(object):
    def __init__(self, eta=0.1, n_iter=1000, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        self.classes = list(Counter(y).keys())
        (self.classes).sort()
        self.classifiers = [LogisticRegressionGD(self.eta, self.n_iter, random_state=self.random_state)
                            for i in range(len(self.classes))]

        for i in range(len(self.classes)):
            X_copy = X.copy()
            y_copy = y.copy()
            y_copy[(y != self.classes[i])] = 0
            y_copy[(y == self.classes[i])] = 1
            self.classifiers[i].fit(X_copy, y_copy)

    def predict(self, X):
        res = self.predict_all(X, self.classifiers, 0)
        return res

    def predict_percentage(self, X):
        res = self.predict_all_percentage(X, self.classifiers, 0)
        return res

    def predict_all(self, X, cls, i):
        if i == len(self.classes) - 3:
            return np.where(cls[0].predict(X) == 1, i, np.where(cls[-1].predict(X) == 1, i+2, i+1))
        else:
            return np.where(cls[0].predict(X) == 1, i, self.predict_all(X, cls[1:], i+1))

    def predict_all_percentage(self, X, cls, i):
        if i == len(self.classes) - 3:
            p_0 = cls[0].predict_exact(X)
            p_1 = cls[-1].predict_exact(X)
            return np.where(p_0 >= 0.0, [i, p_0[0]], np.where(p_1 >= 0.0, [i+2, p_1[0]], i+1))
        else:
            p_0 = cls[0].predict_exact(X)
            return np.where(p_0 >= 0.0, [i, p_0[0]], self.predict_all(X, cls[1:], i+1))


def classify_probability(sample, cl):
    cls, prc = cl.predict_percentage([sample])
    return [cls, prc]


def main():

    iris = datasets.load_iris()
    X = iris["data"][:, [2, 3]]
    y = iris["target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=1, stratify=y)

    cl = BinaryMultiClassifier(eta=0.05, n_iter=600, random_state=1)
    cl.fit(X_train, y_train)

    for sample in X_test:
        print(classify_probability(sample, cl))


if __name__ == '__main__':
    main()
