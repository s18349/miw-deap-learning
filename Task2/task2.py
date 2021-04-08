import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from collections import Counter
from matplotlib.colors import ListedColormap
from logistic_regression_gd import LogisticRegressionGD


def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):

    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1], alpha=0.8, c=cmap(
            idx), marker=markers[idx], label=cl, edgecolor='black')


class BinaryMultiClassifier(object):
    def __init__(self, eta=0.1, n_iter=10, random_state=1):
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

    def predict_all(self, X, cls, i):
        if i == len(self.classes) - 3:
            return np.where(cls[0].predict(X) == 1, i, np.where(cls[-1].predict(X) == 1, i+2, i+1))
        else:
            return np.where(cls[0].predict(X) == 1, i, self.predict_all(X, cls[1:], i+1))


def main():

    iris = datasets.load_iris()
    X = iris["data"][:, [2, 3]]
    y = iris["target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=1, stratify=y)

    cl = BinaryMultiClassifier(eta=0.05, n_iter=600, random_state=1)
    cl.fit(X_train, y_train)

    plot_decision_regions(X=X_test,
                          y=y_test, classifier=cl)
    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')
    plt.legend(loc='upper left')
    plt.show()


if __name__ == '__main__':
    main()
