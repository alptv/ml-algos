import numpy as np


class Statistic:
    def __init__(self, tp, tn, fp, fn):
        self.tp = tp
        self.tn = tn
        self.fp = fp
        self.fn = fn

    def computePrecision(self):
        if (self.tp + self.fp) == 0:
            return 0
        return self.tp / (self.tp + self.fp)

    def computeRecall(self):
        if (self.tp + self.fn) == 0:
            return 0
        return self.tp / (self.tp + self.fn)

    def computeFScore(self):
        if self.tp + self.tp + self.fp + self.fn == 0:
            return 1
        return 2 * self.tp / (self.tp + self.tp + self.fp + self.fn)


def computeStatistic(confusionMatrix):
    matrixSize = len(confusionMatrix)
    rowSums = [0] * matrixSize
    columnSums = [0] * matrixSize
    s = 0
    for i in range(matrixSize):
        for j in range(matrixSize):
            rowSums[i] += confusionMatrix[i][j]
            columnSums[i] += confusionMatrix[j][i]
            s += confusionMatrix[i][j]

    statistic = [0] * matrixSize
    for i in range(matrixSize):
        statistic[i] = Statistic(confusionMatrix[i][i], s - rowSums[i] - columnSums[i] + confusionMatrix[i][i],
                                 columnSums[i] - confusionMatrix[i][i], rowSums[i] - confusionMatrix[i][i])
    return statistic


def microFScore(confusionMatrix):
    statistic = computeStatistic(confusionMatrix)
    precision = 0
    recall = 0
    count = 0

    for s in statistic:
        classSize = s.tp + s.fn
        precision += classSize * s.computePrecision()
        recall += classSize * s.computeRecall()
        count += classSize

    recall /= count
    precision /= count

    if recall + precision == 0:
        return 0
    return (2 * recall * precision) / (recall + precision)


def macroFScore(confusionMatrix):
    statistics = computeStatistic(confusionMatrix)
    macroFScore = 0
    count = 0

    for s in statistics:
        classSize = s.tp + s.fn
        macroFScore += classSize * s.computeFScore()
        count += classSize

    return macroFScore / count


def computeConfusionMatrix(w, kernel, metric, marks, targets, predict):
    count = len(np.unique(targets))
    confusionMatrix = []
    for i in range(count):
        confusionMatrix.append([0] * count)
    for i in range(len(marks)):
        predictedTarget = predict(marks[i], w, kernel, metric, marks, targets, i)
        confusionMatrix[targets[i] - 1][predictedTarget] += 1
    return confusionMatrix
