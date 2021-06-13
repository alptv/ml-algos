import numpy as np
import util as util

REGULARIZATION_PARAMETER = 0
REGULARIZATION_VARIABLES = [0, 1e-32, 1e-31, 1e-30, 1e-29, 1e-28, 1e-27, 1e-26, 1e-25, 1e-24, 1e-23, 1e-22, 1e-21]


def setRegularizationParameter(tau):
    global REGULARIZATION_PARAMETER
    REGULARIZATION_PARAMETER = tau


def lineEquation(marks, target):
    formatMarks = util.changeFormat(marks)
    V, D, Uh = np.linalg.svd(formatMarks)
    ans = np.zeros(len(formatMarks[0]))
    for i in range(len(formatMarks[0])):
        coefficient = D[i] / (D[i] ** 2 + REGULARIZATION_PARAMETER)
        coefficient *= np.dot(V[:, i], target)
        ans = ans + coefficient * Uh[i]
    return ans






