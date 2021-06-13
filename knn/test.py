import knn as knn
import numpy as np
import fscore as fscore
import math as math
import pandas as pd
import matplotlib.pyplot as plt

metrics = {'manhattan': lambda x, y: sum(abs(xi - yi) for xi, yi in zip(x, y)),
           'euclidean': lambda x, y: (sum((xi - yi) ** 2 for xi, yi in zip(x, y))) ** 0.5,
           'chebyshev': lambda x, y: max(abs(xi - yi) for xi, yi in zip(x, y))
           }

kernels = {'uniform': lambda d: 1 / 2 if abs(d) < 1 else 0,
           'triangular': lambda d: 1 - abs(d) if abs(d) < 1 else 0,
           'epanechnikov': lambda d: (3 / 4) * (1 - d ** 2) if abs(d) < 1 else 0,
           'quartic': lambda d: (15 / 16) * (1 - d ** 2) ** 2 if abs(d) < 1 else 0,
           'triweight': lambda d: (35 / 32) * (1 - d ** 2) ** 3 if abs(d) < 1 else 0,
           'tricube': lambda d: (70 / 81) * (1 - abs(d) ** 3) ** 3 if abs(d) < 1 else 0,
           'cosine': lambda d: (math.pi / 4) * math.cos(math.pi * d / 2) if abs(d) < 1 else 0,
           'gaussian': lambda d: (2 * math.pi) ** -0.5 * math.e ** (-0.5 * d ** 2),
           'logistic': lambda d: (math.e ** d + math.e ** (-d) + 2) ** -1,
           'sigmoid': lambda d: (2 / math.pi) * (math.e ** d + math.e ** (-d)) ** -1
           }
predicts = {
    'labelFixed': knn.labelFixed,
    'oneHotFixed': knn.oneHotFixed,
    'labelVariable': knn.labelVariable,
    'oneHotVariable': knn.oneHotVariable,
}


def getBest(mode):
    if mode == 0:
        return ['oneHotFixed', 'tricube', 'chebyshev', 0.2, 0.947580651444431]
    bestPredict = None
    bestKernel = None
    bestMetric = None
    bestWindow = -1
    bestFScore = -1
    for kernel in kernels.keys():
        for metric in metrics.keys():
            for predict in predicts.keys():
                print('next')
                windows = np.linspace(0.1, 1, 10) if (predict == 'labelFixed' or predict == 'oneHotFixed') else range(4,
                                                                                                                      100,
                                                                                                                      10)
                for window in windows:
                    matrix = fscore.computeConfusionMatrix(window, kernels[kernel], metrics[metric], marks, targets,
                                                           predicts[predict])
                    score = fscore.microFScore(matrix)
                    if score >= bestFScore:
                        bestKernel = kernel
                        bestMetric = metric
                        bestWindow = window
                        bestPredict = predict
                        bestFScore = score
    return [bestPredict, bestKernel, bestMetric, bestWindow, bestFScore]


filename = "seeds.csv"
dataset = pd.read_csv(filename)
marks = dataset.to_numpy()[:, :-1:]
targets = dataset.to_numpy()[:, -1]
targets = targets.astype('int64')
knn.normalize(marks)

bestPredict, bestKernel, bestMetric, bestWindow, bestFScore = getBest(0)
# 0.947580651444431
# tricube
# chebyshev
# oneHotFixed
# 0.2
k = []
h = []
fk = []
fh = []
kpredict = knn.oneHotVariable if bestPredict == 'oneHotVariable' or bestPredict == 'oneHotFixed' else knn.labelVariable
hpredict = knn.oneHotFixed if bestPredict == 'oneHotVariable' or bestPredict == 'oneHotFixed' else knn.labelFixed
j = 0
for i in range(1, 210, 3):
    print(j, end=' ')
    j += 1
    k.append(i)
    fk.append(fscore.microFScore(
        fscore.computeConfusionMatrix(i, kernels[bestKernel], metrics[bestMetric], marks, targets, kpredict)))
print("")
j = 0
for i in np.linspace(0.1, 1, 100):
    print(j, end=' ')
    j += 1
    h.append(i)
    fh.append(fscore.microFScore(
        fscore.computeConfusionMatrix(i, kernels[bestKernel], metrics[bestMetric], marks, targets, hpredict)))
plt.plot(k, fk)
plt.show()
plt.plot(h, fh)
plt.show()
