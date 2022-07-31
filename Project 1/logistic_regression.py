"""
University of Liege
ELEN0062 - Introduction to machine learning
Project 1 - Classification algorithms
"""
# ! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

from plot import plot_boundary
from data import make_balanced_dataset, make_unbalanced_dataset
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.svm import LinearSVC

from plot import plot_boundary


class LogisticRegressionClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, n_iter=10, learning_rate=1):
        self.n_iter = n_iter
        self.learning_rate = learning_rate

        self.w0_ = 0
        self.w_ = np.array([0, 0])

    def fit(self, X, y):
        """Fit a logistic regression models on (X, y)
        
        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            The training input samples.
            
        y : array-like, shape = [n_samples]
            The target values.
            
        Returns
        -------
        self : object
            Returns self.
        """
        # Input validation
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be 2 dimensional")
        n_instances, n_features = X.shape

        y = np.asarray(y)
        if y.shape[0] != X.shape[0]:
            raise ValueError("The number of samples differs between X and y")

        n_classes = len(np.unique(y))
        if n_classes != 2:
            raise ValueError("This class is only dealing with binary "
                             "classification problems")

        # Default parameters are such that P=1/2 for each class
        self.w0_ = 0
        self.w_ = np.array([0, 0])

        # Train the parameters
        for t in range(0, self.n_iter):
            grad = 0
            prob_array = self.predict_proba(X)[:, 1]
            x_prime = np.hstack((np.full((n_instances, 1), 1), X))
            for i in range(0, n_instances):
                grad = grad + (prob_array[i] - y[i]) * x_prime[i]
            grad = grad / n_instances

            self.w0_ = self.w0_ - self.learning_rate * grad[0]
            self.w_ = self.w_ - self.learning_rate * grad[1:3]

        return self

    def predict(self, X):
        """Predict class for X.
        
        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples.
            
        Returns
        -------
        y : array of shape = [n_samples]
            The predicted classes, or the predict values.
        """

        y = np.array([+1 if p >= 0.5 else 0
                      for p in self.predict_proba(X)[:, 1]])

        return y

    def predict_proba(self, X):
        """Return probability estimates for the test data X.
        
        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples.
            
        Returns
        -------
        p : array of shape = [n_samples, n_classes]
            The class probabilities of the input samples. Classes are ordered
            by lexicographic order.
        """

        probas = 1 / (1 + np.exp(- self.w0_ - np.dot(X, self.w_)))
        comp_probas = 1 - probas

        N = X.shape[0]

        return np.hstack((comp_probas.reshape(N, 1), probas.reshape(N, 1)))


if __name__ == "__main__":
    # Question 5.
    X, y = make_unbalanced_dataset(3000, 0)
    est = LogisticRegressionClassifier(15, 0.7)
    est.fit(X[0:1000], y[0:1000])
    plot_boundary("logreg", est, X[1000:3000], y[1000:3000],
                  title="Logistic Regression Boundary")

    # Question 6
    accuracy = []
    for seed in range(1, 6):
        X, y = make_unbalanced_dataset(3000, seed)
        est.fit(X[0:1000], y[0:1000])

        test_res = est.predict(X[1000:3000])
        correct = y[1000:3000] == test_res
        accuracy = accuracy + [np.count_nonzero(correct) / len(correct)]

    print("Average accuracy: " + str(100 * np.mean(accuracy)) + "%")
    print("Std Dev.: " + str(100 * np.std(accuracy)) + "%")

    # Question 7
    max_tries = 80
    iterations_values = np.arange(1, max_tries + 1, 1)
    error_rate = []
    for iterations in iterations_values:
        est = LogisticRegressionClassifier(iterations, 0.7)
        X, y = make_unbalanced_dataset(3000, 0)
        est.fit(X[0:1000], y[0:1000])

        test_res = est.predict(X[1000:3000])
        correct = y[1000:3000] == test_res
        error_rate = \
            error_rate + [100 - 100 * np.count_nonzero(correct) / len(correct)]

        if iterations in [1, 2, 3, 10, 20, 30]:
            plot_boundary("fig/logreg_" + str(iterations), est, X[1000:3000], y[1000:3000],
                          title=(str(iterations) + " Iterations"))

    plt.plot(iterations_values, error_rate)
    plt.xlim(1, max_tries)
    plt.ylim(0, np.max(error_rate))
    plt.xlabel("Number of iterations")
    plt.ylabel("Error Rate (%)")
    plt.savefig('{}.pdf'.format("error_study"))
    pass