import numpy as np
import pandas as pd
import graphic

import svm as svm
import math as math

C = [0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]
gauss_parameters = [1, 2, 3, 4, 5]
polynomial_parameters = [2, 3, 4, 5]


def read(data):
    targets = np.array(data['class'].map(lambda c: 1 if c == 'P' else -1))
    data = data.drop('class', 1)
    return data.to_numpy(), targets


def polynomial(d):
    return lambda x, y: (x.dot(y) + 1) ** d


def gaussian(b):
    return lambda x, y: math.e ** (-b * (x - y).dot(x - y))


def linear():
    return lambda x, y: x.dot(y)


def accuracy(dataset, target, predictor):
    correct = 0
    for i in range(len(target)):
        if predictor.predict(dataset[i]) == target[i]:
            correct += 1
    return correct / len(target)


def find_best_C(dataset, targets, kernel):
    best_c = -1
    best_accuracy = -1
    for c in C:
        predictor = svm.SVM(dataset, targets, kernel, c)
        acc = accuracy(dataset, targets, predictor)
        if best_accuracy < acc:
            best_accuracy = acc
            best_c = c
    return best_c, best_accuracy


def show(prefix, names, values):
    for i in range(len(names)):
        print(prefix, ' ', names[i], ' ', values[i])


def find_best(file):
    data = pd.read_csv(file)
    dataset, targets = read(data)

    best_linear_c, best_linear_accuracy = find_best_C(dataset, targets, linear())
    print("Lin end")
    best_gaussian_c = -1
    best_gaussian_b = -1
    best_gaussian_accuracy = -1
    for b in gauss_parameters:
        c, acc = find_best_C(dataset, targets, gaussian(b))
        if acc > best_gaussian_accuracy:
            best_gaussian_b = b
            best_gaussian_c = c
            best_gaussian_accuracy = acc
    print("Gauss end")
    best_poly_c = -1
    best_poly_d = -1
    best_poly_accuracy = -1
    for d in polynomial_parameters:
        c, acc = find_best_C(dataset, targets, polynomial(d))
        if acc > best_poly_accuracy:
            best_poly_c = c
            best_poly_d = d
            best_poly_accuracy = acc
    print("Poly end")
    bests = ([best_linear_accuracy, best_linear_c], [best_gaussian_accuracy, best_gaussian_c, best_gaussian_b],
             [best_poly_accuracy, best_poly_c, best_poly_d])
    show("Linear", ["acc", "c"], bests[0])
    show("Gauss", ["acc", "c", "b"], bests[1])
    show("Poly", ["acc", "c", "d"], bests[2])
    return bests


file = 'geyser.csv'
# bests = find_best(file)
data = pd.read_csv(file)
dataset, targets = read(data)
predictor = svm.SVM(dataset, targets, polynomial(2), 0.05)
graphic.draw_graphic(dataset, targets, predictor, 'polynomial', 'polynomial_geyser')
print(accuracy(dataset, targets, predictor))


# chips
# Linear   acc   0.559322033898305
# Linear   c   1.0
# Gauss   acc   0.864406779661017
# Gauss   c   10.0
# Gauss   b   3
# Poly   acc   0.0.8389830508474576
# Poly   c   10.0
# Poly   d   3

# geysers
# Linear   acc   0.9099099099099099
# Linear   c   0.05
# Gauss   acc   0.9279279279279279
# Gauss   c   5.0
# Gauss   b   3
# Poly   acc   0.9009009009009009
# Poly   c   0.05
# Poly   d   2
