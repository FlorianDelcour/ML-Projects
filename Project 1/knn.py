"""
University of Liege
ELEN0062 - Introduction to machine learning
Project 1 - Classification algorithms
"""
#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

from plot import plot_boundary
from data import make_balanced_dataset, make_unbalanced_dataset
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score

# (Question 2)

def buildKNN(X_train, y_train, n_neighbors):
    """Build a K-Nearest Neighbors model with K = n_neighbors.
        Parameters
        ----------
        X_train: 2D Array
            The learning set (attributes) to train our model
        y_train : 1D Array
            The learning set (labels) to train our model
        n_neighbors : parameter k of the k-NN model
        Return
        ------
        knn : KNeighborsClassifier
            The KNN-model trained on the learning set with K = n_neighbors.
        """
    # Load K-Neighbors Algorithm and fit the model parameters
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn = knn.fit(X_train, y_train)
    
    return knn

def accuracyCompute(knn, X_test, y_test):
    """Compute the accuracy of the KNN-model
        Parameters
        ----------
        knn : KNeighborsClassifier
            KNN-model already trained
        X_test : 2D Array of float
            The testing set (attributes) to score our model
        y_test : 1D Array of boolean
            The testing set (labels) to score our model
        Return
        ------
        accuracy_score : float
            The accuracy of the KNN-model
        """
    y_predicted = knn.predict(X_test)
    return accuracy_score(y_predicted, y_test)

def KCrossValidation(X, y, n_neighbors):
    """Ten-folds cross validation algorithm
    Parameters
    ----------
    X : 2D Array of float
        Dataset (attributes)
    y : 1D Array of boolean
        Dataset (labels)
    n_neighbors : 
        n_neighbors : parameter k of the k-NN model
    Return
    ------
    np.mean(accuracies) : float
        Mean of the accuracies of the cross validation alorithm
    """
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    accuracies = cross_val_score(knn, X, y, cv = 10, scoring = 'accuracy')
    return np.mean(accuracies)

if __name__ == "__main__":
    
    n_neighbors = [1, 5, 50, 100, 500]
    mean_accuracies = []
    # Build learning and testing sets
    X, y = make_unbalanced_dataset(3000, 0)

    # Divide the data set into a learning set and a testing set
    number_of_ls = 1000
    X_train, X_test = X[:number_of_ls,:], X[number_of_ls:,:]
    y_train, y_test = y[:number_of_ls], y[number_of_ls:]
    
    for i in n_neighbors:
        
        ## Q2.1
        
        knn = buildKNN(X_train, y_train, i)
        fname = str(i) + "_neighbors"
        if False:
            plot_boundary(fname, knn, X_test, y_test, title = "K-Nearest Neighbors (k = " + str(i)+ ")")
        
        ## Q2.2
        mean_accuracies.append(KCrossValidation(X_train, y_train, i))
        if i==5: # We found that n_neighbors = 5 was the optimal value
            print("Q2.2 : Optimal accuracy with the final model : {:.2f}" .format(accuracyCompute(knn, X_test, y_test)))
    print("Q2.1 : Here is the mean accuracy for n_neighbors = 1; 5; 50; 100; 500 respectively :\n{}" .format(mean_accuracies))
    
    ## Q2.3
    
    LS_sizes = [50, 150, 250, 350, 450, 500]
    X_test, y_test = make_unbalanced_dataset(500, 0)
    n_neighbors_optimal = []
    for size in LS_sizes:
        mean_accuracies = []
        
        for n_neighbors in range(1, size+1):
            accuracy = []
            for i in range(10):
                X_train, y_train = make_unbalanced_dataset(size, i)
                knn = buildKNN(X_train, y_train, n_neighbors)
                accuracy.append(accuracyCompute(knn, X_test, y_test))
            mean_accuracies.append(np.mean(accuracy))
        
        ## Q2.3.a
        plt.plot(np.arange(1, size+1), mean_accuracies, label = "LS size of " + str(size))
        plt.xlabel('Number of neighbors', fontsize = 14)
        plt.ylabel('Mean Accuracy', fontsize = 14)
        plt.title('Evolution of mean test accuracies, fixed test set of size 500 and ten generations of training set', fontsize = 16)
        plt.rc('xtick', labelsize=14)
        plt.rc('ytick', labelsize=14)
        
        ## Q2.3.b
        n_neighbors_optimal.append(mean_accuracies.index(max(mean_accuracies)) + 1)
    plt.legend(prop={'size':14})
    
    plt.figure()
    plt.plot(LS_sizes, n_neighbors_optimal)
    plt.xlabel('size of LS', fontsize = 14)
    plt.ylabel('Optimal value of n_neighbors', fontsize = 14)
    plt.title('Optimal value of n_neighbors with respect to the training set size (LS)', fontsize = 16)  
    plt.rc('xtick', labelsize=14)
    plt.rc('ytick', labelsize=14)
    
    
    pass
