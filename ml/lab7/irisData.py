from __future__ import division

import csv
import numpy as np
import random
from random import randrange
from math import exp
import copy

data = []
with open('irisData.csv', 'rb') as csv_file:
    iris_data_reader = csv.reader(csv_file, delimiter=',', quotechar='|')
    for row in iris_data_reader:
        if row[4] == 'Iris-setosa':
            data.append([float(1), float(row[0]), float(row[1]), float(row[2]), float(row[3]), 1])
        elif row[4] == 'Iris-versicolor':
            data.append([float(1), float(row[0]), float(row[1]), float(row[2]), float(row[3]), 2])
        else:
            data.append([float(1), float(row[0]), float(row[1]), float(row[2]), float(row[3]), 3])


def normalize_dataset(data):
    minmax = list()
    for i in range(len(data[0])):
        col_values = [row[i] for row in data]
        value_min = min(col_values)
        value_max = max(col_values)
        minmax.append([value_min, value_max])

    for row in data:
        for i in range(len(row) - 1):
            if minmax[i][1] != minmax[i][0]:
                row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])


# normalize_dataset(data)
np.random.shuffle(data)
training_data = data[:135]
test_data = data[135:150]


def logistic_function(x):
    return 1.0 / (1.0 + exp((-1) * x))


def cross_validation_split(dataset, n_folds):
    """ Split data set for n folds. Need for k-fold validation"""
    dataset_split = []
    dataset_copy = copy.copy(dataset)
    fold_size = int(len(dataset) / n_folds)
    for i in range(n_folds):
        fold = []
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split


def coefficients_sgd(train, learning_rate, n_epoch, learning_class, regularization_rate):
    """Calculates coefficients for separting vector (which will be used in logistic regression function)
    Use stochastic gradient algorithm: select random vector from training data, calculate gradient on this vector
    and change separating vector.

    train -- train data
    learning_rate --  Used to limit the amount each coefficient is corrected each time it is updated.
    n_epoch -- count of learning epochs
    learning_class -- class which will be learned (there is tree classes for iris data)
    regularization_rate -- parameter which will be used for l2 regularization
    """
    w = np.array([0, 0, 0, 0, 0], dtype=np.float32)
    for epoch in range(n_epoch):
        random_row = random.choice(train)
        temp_w = copy.copy(w)
        if learning_class == random_row[5]:
            y = 1
        else:
            y = -1
        for i in range(0, 5):
            w[i] = w[i] - learning_rate * ((1 - logistic_function(np.sum(random_row[0:5] * temp_w) * y))
                                           * ((-1) * y * random_row[i]) + regularization_rate * w[i])
    return w


def coefficients_batchgd(train, learning_rate, n_epoch, learning_class, regularization_rate):
    """Calculates coefficients for separting vector (which will be used in logistic regression function)
       Use batch gradient algorithm: calculates average gradient for all samples and change separating vector.

       train -- train data
       learning_rate --  Used to limit the amount each coefficient is corrected each time it is updated.
       n_epoch -- count of learning epochs
       learning_class -- class which will be learned (there is tree classes for iris data)
       regularization_rate -- parameter which will be used for l2 regularization
       """
    w = np.array([0, 0, 0, 0, 0], dtype=np.float32)
    for epoch in range(n_epoch):
        w_accumulate = np.array([0, 0, 0, 0, 0], dtype=np.float32)
        temp_w = copy.copy(w)
        for row in train:
            if learning_class == row[5]:
                y = 1
            else:
                y = -1
            for i in range(0, 5):
                w_accumulate[i] += ((1 - logistic_function(np.sum(row[0:5] * temp_w) * y))
                                    * ((-1) * y * row[i]) + regularization_rate * w[i])
        for i in range(0, 5):
            w[i] = w[i] - learning_rate * w_accumulate[i] / train.__len__()
    return w


# w = coefficients_sgd(training_data, 0.01, 1000, 2, 0.4)
w = coefficients_batchgd(training_data, 0.01, 100, 2, 0.1)

def test_prediction(sample, w, learning_class):
    """Predict result on sample and returns result according learning class"""
    prediction = logistic_function(np.sum(sample[0:5] * w))
    if prediction >= 0.5 and learning_class == sample[5] or prediction < 0.5 and learning_class != sample[5]:
        return 1
    else:
        return 0


def test_solotion(test_data, w, learning_class):
    """Calculates percentage of correctly predicted objects"""
    correctly_predicted = 0
    for test_row in test_data:
        correctly_predicted += test_prediction(test_row, w, learning_class)
    return correctly_predicted / test_data.__len__()


def evaluate_algorithm(algorithm, dataset, n_epochs, learning_rate, learning_class, regularization_rate, n_folds=5):
    folds = cross_validation_split(dataset, n_folds)
    scores = []
    for test_set in folds:
        train_set = [item for item in dataset if item not in test_set]
        w = algorithm(train_set, learning_rate, n_epochs, learning_class, regularization_rate)
        accuracy = test_solotion(test_set, w, learning_class)
        scores.append(accuracy)
    return sum(scores)/scores.__len__()


# print evaluate_algorithm(coefficients_sgd, training_data, 100, 0.01, 1, 0.2)
# print evaluate_algorithm(coefficients_sgd, training_data, 100, 0.01, 2, 0.2)
# print evaluate_algorithm(coefficients_sgd, training_data, 100, 0.01, 3, 0.2)


best_solution = 0
best_params = None
#for n_epochs in range(100, 500, 100):
for n_epochs in [100]:
    print "!!!"
    for learning_rate in np.arange(0.0, 1.001, 0.01):
        for regularization_rate in np.arange(0.0, 1.1, 0.01):
            class1_accurasity = evaluate_algorithm(coefficients_sgd, training_data, n_epochs, learning_rate, 1, regularization_rate)
            class2_accurasity = evaluate_algorithm(coefficients_sgd, training_data, n_epochs, learning_rate, 2, regularization_rate)
            class3_accurasity =  evaluate_algorithm(coefficients_sgd, training_data, n_epochs, learning_rate, 3, regularization_rate)
            print class1_accurasity
            print class2_accurasity
            print class3_accurasity

            if class1_accurasity + class2_accurasity + class3_accurasity > best_solution:
                best_solution = class1_accurasity + class2_accurasity + class3_accurasity
                best_params = (n_epochs, learning_rate, regularization_rate)
            print "-----"
print best_params
print best_solution

# (490, 0.01, 0.01)

# print evaluate_algorithm(coefficients_sgd, training_data, 480, 1, 1, 0)
# print evaluate_algorithm(coefficients_sgd, training_data, 480, 1, 2, 0)
# print evaluate_algorithm(coefficients_sgd, training_data, 480, 1, 3, 0)

#
# for learning_rate in np.arange(0.0, 1.1, 0.1):
#     print learning_rate