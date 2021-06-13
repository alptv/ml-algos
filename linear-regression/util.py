import numpy as np


def formatObjectMarks(objectMarks):
    length = len(objectMarks) + 1
    result = np.full(length, -1)
    for i in range(1, length):
        result[i] = objectMarks[i - 1]
    return result


def changeFormat(marks):
    objectCount = len(marks)
    markCount = len(marks[0]) + 1
    formatMarks = np.empty([objectCount, markCount])
    for i in range(objectCount):
        formatMarks[i] = formatObjectMarks(marks[i])
    return formatMarks
