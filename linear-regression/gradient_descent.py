import numpy as np
import util as util

STEP = 2000
EPS = 0.05
BATCH_SIZE = 1
RISK_PARAMETER = 0.7
MAX_ITERATION_COUNT = 2000
RISK_INF = 100

REGULARIZATION_PARAMETER = 0
REGULARIZATION_VARIABLES = [0, 1e-32, 1e-31, 1e-30, 1e-29, 1e-28, 1e-27, 1e-26, 1e-25, 1e-24, 1e-23, 1e-22, 1e-21,
                            1e-20, 1e-19, 1e-18, 1e-17, 1e-16, 1e-15, 1e-14, 1e-13, 1e-12, 1e-11, 1e-10, 1e-9, 1e-8,
                            1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 5e-2, 6e-2, 1e-1, 2e-1]


def setRegularizationParameter(tau):
    global REGULARIZATION_PARAMETER
    REGULARIZATION_PARAMETER = tau


def initWeights(length):
    return np.zeros(length)


def predict(mark, weights):
    return np.dot(mark, weights)


def gradient(marks, targets, weights, iteration):
    grad = np.zeros(len(weights))
    objectCount = len(marks)
    squareSum = 0
    normalization = max(targets) - min(targets)
    for i in range(BATCH_SIZE):
        index = (BATCH_SIZE * iteration + i) % objectCount
        predictTarget = predict(marks[index], weights)
        grad += (predictTarget - targets[index]) * marks[index]
        squareSum += (predictTarget - targets[index]) ** 2
    return weights * REGULARIZATION_PARAMETER + grad / (np.sqrt(squareSum * objectCount) * normalization)


def isPrimitive(targets):
    return max(targets) == min(targets)


def primitiveLine(length, targets):
    line = [0] * length
    line[0] = -targets[0]
    return line


def isWeightsClose(weights, previousWeights):
    for i in range(len(weights)):
        if abs(weights[i] - previousWeights[i]) >= EPS:
            return False
    return True


def isRiskClose(risk, previousRisk):
    return abs(risk - previousRisk) <= EPS


def empiricalRisk(marks, targets, weights, iteration):
    objectCount = len(targets)
    risk = 0
    for i in range(BATCH_SIZE):
        index = (BATCH_SIZE * iteration + i) % objectCount
        predictTarget = predict(marks[index], weights)
        risk += (predictTarget - targets[i]) ** 2
    risk /= objectCount
    risk **= 1 / 2
    risk /= max(targets) - min(targets)
    return risk


def lineEquation(marks, targets):
    if isPrimitive(targets):
        return primitiveLine(len(marks[0]) + 1, targets)
    formatMarks = util.changeFormat(marks)
    weights = initWeights(len(formatMarks[0]))
    risk = RISK_INF
    iteration = 1
    while True:
        previousWeights = np.copy(weights)
        previousRisk = risk
        weights = weights - (STEP / iteration) * gradient(formatMarks, targets, weights, iteration)
        risk = (1 - RISK_PARAMETER) * risk + RISK_PARAMETER * empiricalRisk(formatMarks, targets, weights, iteration)
        iteration += 1
        if (isWeightsClose(weights, previousWeights) and isRiskClose(risk,
                                                                     previousRisk)) or iteration > MAX_ITERATION_COUNT:
            break
    return weights
