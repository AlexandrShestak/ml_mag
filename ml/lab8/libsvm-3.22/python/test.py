from svmutil import *
import subprocess

shuffled_file = open('spambase.data.shuffled', 'r')

X_ORIGINAL = []
Y_ORIGINAL = []
with open("spambase.data.shuffled", "r") as shuffled_file:
    for line in shuffled_file:
        values = line.split(',')
        Y_ORIGINAL.append(int(values[-1]))
        x_temp = []
        for i in range(0, values.__len__() - 1):
            x_temp.append(float(values[i]))
        X_ORIGINAL.append(x_temp)

x_train = X_ORIGINAL[: 3450]
y_train = Y_ORIGINAL[: 3450]
x_test = X_ORIGINAL[3450:]
y_test = Y_ORIGINAL[3450:]


def cross_validation_split(x, y, n_folds):
    """ Split data set for n folds. Need for k-fold validation"""
    dataset_split = []
    fold_size = x.__len__() / n_folds
    for i in range(n_folds):
        x_test = x[i * fold_size:(i + 1) * fold_size]
        y_test = y[i * fold_size:(i + 1) * fold_size]
        x_train = x[0: max(i, 0) * fold_size]
        x_train.extend(x[min(i + 1, 10) * fold_size: 10 * fold_size])
        y_train = y[0: max(i , 0) * fold_size]
        y_train.extend(y[min(i + 1, 10) * fold_size: 10 * fold_size])
        print x_test.__len__()
        print y_test.__len__()
        print x_train.__len__()
        print y_train.__len__()
        dataset_split.append(((x_train, y_train), (x_test, y_test)))
    return dataset_split


def k_fold(x_train, y_train):
    dataset_split = cross_validation_split(x_train, y_train, 10)
    for d in [1, 2, 3, 4]:
        for k in range(-10, 11, 1):
            c = 2 ** k
            scores = []
            for dataset in dataset_split:
                print dataset[0][1].__len__()
                print dataset[0][0].__len__()
                model = svm_train(dataset[0][1], dataset[0][0], "-s 0 -t 1 -d %d -c %f" % (d, c))
                p_label, p_acc, p_val = svm_predict(dataset[1][1], dataset[1][0], model)
                scores.append(p_acc[0])
            score = sum(scores) / scores.__len__()
            print "d: %d, c: %f, score: %f" % (d, c, score)

y_train_scaled, x_train_scaled = svm_read_problem('data.libsvmformat.train.scaled')
k_fold(x_train_scaled, y_train_scaled)

