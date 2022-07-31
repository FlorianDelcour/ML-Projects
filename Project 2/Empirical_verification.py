import numpy as np
from sklearn.metrics import accuracy_score
import random
import math


def generate(N, sigma):
    # We draw N samples
    classes = np.array([])
    samples = []
    for i in range(N):
        random.seed(3*i)
        # We generate a number between 0 and 100, if it is smaller than 25 (because the probability of the positive
        # class is 0.25) then we draw a sample with the distribution of the positive class. Otherwise we draw a sample
        # with the distribution of the negative class
        if random.randrange(0, 100) < 25:
            classes = np.append(classes, 1)
            x1, x2 = np.random.multivariate_normal([-1.5, -1.5], [[pow(sigma, 2), 0], [0, pow(sigma, 2)]])
            samples.append([x1, x2])
        else:
            classes = np.append(classes, 0)
            x1, x2 = np.random.multivariate_normal([1.5, 1.5], [[sigma ** 2, 0], [0, sigma ** 2]])
            samples.append([x1, x2])

    return classes, samples


def predict(samples, sigma):
    # To predict the class we use the condition derived for the Bayes model, which is that the negative class is
    # predicted if x1 > -(ln(3) * sigma^2 / 3) - x2, so we just have to verify for each sample if the values of x1
    # and x2 verify this inequality or not, and we can predict the class from this.
    predicted = np.array([])
    k = -pow(sigma, 2)*(math.log(3))/3
    for [x1, x2] in samples:
        if (x1 + x2) > k:
            predicted = np.append(predicted, 0)
        else:
            predicted = np.append(predicted, 1)

    return predicted


if __name__ == "__main__":
    N = 10000
    sigma = 1.6
    class_of_samples, samples = generate(N, sigma)
    predictions = predict(samples, sigma)
    error = 1-accuracy_score(class_of_samples, predictions)
    print(error)


