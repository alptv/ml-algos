import numpy as np
import matplotlib.pyplot as plt
import gradient_descent as gd
import least_square_method as lsd


def predict(line, objectMarks):
    prediction = -line[0]
    for i in range(len(objectMarks)):
        prediction += objectMarks[i] * line[i + 1]
    return prediction


def score(line, marks, targets):
    nrmse = 0
    targetCount = len(targets)
    for i in range(targetCount):
        prediction = predict(line, marks[i])
        nrmse += (prediction - targets[i]) ** 2
    maxTarget = max(targets)
    minTarget = min(targets)
    if maxTarget == minTarget:
        return float(nrmse == 0)
    else:
        nrmse /= targetCount
        nrmse **= 1 / 2
        nrmse /= maxTarget - minTarget
    return nrmse


def read(data, m):
    n = int(data.readline())
    mark = np.empty([n, m])
    target = np.empty([n])
    for i in range(n):
        marksAndTarget = list(map(int, data.readline().split()))
        target[i] = marksAndTarget[m]
        mark[i] = np.array(marksAndTarget[:-1])
    return mark, target


def graphic(methodType):
    x = []
    y = []
    for tau in methodType.REGULARIZATION_VARIABLES:
        methodType.setRegularizationParameter(tau)
        line = methodType.lineEquation(marks, targets)
        x.append(tau)
        y.append(score(line, marks, targets))
    plt.plot(x, y)
    plt.show()


with open("data.txt") as data:
    m = int(data.readline())
    marks, targets = read(data, m)
    lineDescent = gd.lineEquation(marks, targets)
    lineSquare = lsd.lineEquation(marks, targets)
    marks, targets = read(data, m)
    graphic(gd)
    graphic(lsd)
