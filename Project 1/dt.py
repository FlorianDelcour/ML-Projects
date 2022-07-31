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
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.metrics import confusion_matrix, accuracy_score


# (Question 1)

def buildDecisionTree(min_samples_split, random_state):
    """Build a decision tree with a given minimum number of samples needed to split an internal node.
        Parameters
        ----------
        min_samples_split: int
            The minimum number of samples needed to split an internal node.
        random_state : int
            Seed for the random number generator.
        Return
        ------
        clf : DecisionTreeClassifier
            The decision tree trained on a learning set of 1000 samples.
        X_test : List of input samples (float)
            The input samples in the testing set.
        y_test : List of outputs (int)
            The outputs of the samples in the testing set.
        accuracy_score : int
            The accuracy score of the decision tree built.
        """
    # Build learning and testing sets
    X, y = make_unbalanced_dataset(3000, random_state)

    # Divide the data set into a learning set and a testing set
    number_of_ls = 1000
    X_train, X_test = X[:number_of_ls, :], X[number_of_ls:, :]
    y_train, y_test = y[:number_of_ls], y[number_of_ls:]

    # Load Decision Tree Algorithm and fit the model parameters
    clf = DecisionTreeClassifier(min_samples_split=min_samples_split)
    clf = clf.fit(X_train, y_train)

    # Use the model on the testing set
    predictions = clf.predict(X_test)
    return clf, X_test, y_test, accuracy_score(predictions, y_test)


if __name__ == "__main__":

    min_samples_split = [2, 8, 32, 64, 128, 500]
    mean_accuracies = []
    std_accuracies = []
    for i in min_samples_split:
        # Q1.1.a) We build a decision tree for each of the min_samples_split values given and we observe the boundaries
        # obtained by training it on the learning set of samples, compared to the samples in the testing set
        clf, X_test, y_test = buildDecisionTree(i, 0)[0:3]
        fname = "min_samples_split_" + str(i)
        plot_boundary(fname, clf, X_test, y_test,
                      title="Decision tree boundary (min_samples_split = " + str(i) + ")")

        # Q1.3) We build five decision trees for each min_samples_split value and we train them each on different
        # learning set of samples, then we compute the average accuracies for each min_samples_split value
        accuracy = np.zeros(5, dtype=float)
        for seed in range(5):
            accuracy[seed] = buildDecisionTree(i, seed)[3]

        mean_accuracies.append(np.mean(accuracy))
        std_accuracies.append(np.std(accuracy))
        print("Here is the mean accuracy for min_samples_split = " + str(i) + " : " + str(mean_accuracies[-1]))
        print("Here is the standard deviation of the accuracy for min_samples_split = " + str(i) + " : " +
              str(std_accuracies[-1]))