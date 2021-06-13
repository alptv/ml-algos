import pandas as pd
import numpy as np
import math as math


def computeMinmax(marks):
    minmax = []
    for j in range(len(marks[0])):
        column = marks[:, j]
        minmax.append((column.min(), column.max()))
    return minmax


def normalize(marks):
    minmax = computeMinmax(marks)
    for i in range(len(marks)):
        for j in range(len(marks[i])):
            marks[i][j] = (marks[i][j] - minmax[j][0]) / (minmax[j][1] - minmax[j][0])


def labelEncoding(targets):
    classes = np.unique(targets)
    classes = {classes[i]: float(i + 1) for i in range(len(classes))}
    encoding = []
    for target in targets:
        encoding.append(classes[target])
    return encoding


def oneHotEncoding(targets):
    classes = np.unique(targets)
    classes = {classes[i]: [float((i - j == 0)) for j in range(len(classes))] for i in range(len(classes))}
    encoding = []
    for target in targets:
        encoding.append(np.array(classes[target]))
    return encoding


def fixedWindowRegression(mark, window, kernel, metric, marks, targets, skip=-1):
    # not empty marks
    weightDistanceSum = 0
    weightSum = 0
    average = 0
    averageInMark = 0
    countInMark = 0
    for i in range(len(marks)):
        if i == skip:
            continue
        average += targets[i]
        distance = metric(mark, marks[i])
        if window != 0:
            weight = kernel(distance / window)
            weightDistanceSum += targets[i] * weight
            weightSum += weight
        elif distance == 0:
            averageInMark += targets[i]
            countInMark += 1
    if window == 0 and countInMark == 0:
        return average / len(marks)
    if window == 0:
        return averageInMark / countInMark
    if weightSum == 0:
        return average / len(marks)
    return weightDistanceSum / weightSum


def variableWindowRegression(mark, k, kernel, metric, marks, targets, skip=-1):
    # k cant' be >= dataset size
    sortedMarks = sorted(marks, key=lambda other: metric(mark, other))
   # print(metric(mark, sortedMarks[k]), sep=' ')
    return fixedWindowRegression(mark, metric(mark, sortedMarks[k]), kernel, metric, marks, targets, skip)


def predict(mark, w, kernel, metric, marks, targets, encoding, fixed, skip=-1):
    enc = encoding(targets)
    if fixed:
        return fixedWindowRegression(mark, w, kernel, metric, marks, enc, skip)
    else:
        return variableWindowRegression(mark, w, kernel, metric, marks, enc, skip)


def getLabeledClass(clazz):
    ansFloor = math.floor(clazz)
    ansCeil = math.ceil(clazz)
    if abs(ansFloor - clazz) < abs(ansCeil - clazz):
        return int(ansFloor - 1)
    return int(ansCeil - 1)


def getOneHotClass(vector):
    maximum = -1
    index = -1
    for i in range(len(vector)):
        if vector[i] > maximum:
            maximum = vector[i]
            index = i
    return index


def labelFixed(mark, h, kernel, metric, marks, targets, skip=-1):
    return getLabeledClass(predict(mark, h, kernel, metric, marks, targets, labelEncoding, True, skip))


def oneHotFixed(mark, h, kernel, metric, marks, targets, skip=-1):
    return getOneHotClass(predict(mark, h, kernel, metric, marks, targets, oneHotEncoding, True, skip))


def labelVariable(mark, k, kernel, metric, marks, targets, skip=-1):
    return getLabeledClass(predict(mark, k, kernel, metric, marks, targets, labelEncoding, False, skip))


def oneHotVariable(mark, k, kernel, metric, marks, targets, skip=-1):
    return getOneHotClass(predict(mark, k, kernel, metric, marks, targets, oneHotEncoding, False, skip))
