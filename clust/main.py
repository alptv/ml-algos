import matplotlib.pyplot as plt
import pandas as pd
from util import *
from kmeans import *


def draw_classes(signs, targets, name, file, cls, mark):
    reduced_signs = pca(signs, 2)
    x1 = []
    y1 = []
    x2 = []
    y2 = []
    x3 = []
    y3 = []
    for i in range(len(reduced_signs)):
        if targets[i] == 0:
            x1.append(reduced_signs[i][0])
            y1.append(reduced_signs[i][1])
        if targets[i] == 1:
            x2.append(reduced_signs[i][0])
            y2.append(reduced_signs[i][1])
        if targets[i] == 2:
            x3.append(reduced_signs[i][0])
            y3.append(reduced_signs[i][1])
    plt.scatter(x1, y1, c=cls[0], marker=mark)
    plt.scatter(x2, y2, c=cls[1], marker=mark)
    plt.scatter(x3, y3, c=cls[2], marker=mark)
    plt.title(name)
    plt.savefig(file)
    plt.show()

def draw_plot(signs, targets):
    centers_cnt = []
    rand_val = []
    sil_val = []
    for center_cnt in range(2, 12):
        print(center_cnt)
        clust = KMeans(signs, center_cnt, metrics[1])
        clust.train()
        centers_cnt.append(center_cnt)
        rand_val.append(rand(clust.targets, targets))
        sil_val.append(silhouette(clust.targets, signs, metrics[1], center_cnt))
    plt.plot(centers_cnt, rand_val)
    plt.title('Rand')
    plt.savefig('/home/alexander/Projects/Python/clust/rand')
    plt.show()
    plt.plot(centers_cnt, sil_val)
    plt.title('Sil')
    plt.savefig('/home/alexander/Projects/Python/clust/sil')
    plt.show()


metrics = [
    lambda x, y: (sum((xi - yi) ** 2 for xi, yi in zip(x, y))) ** 0.5,
    lambda x, y: sum(abs(xi - yi) for xi, yi in zip(x, y)),
    lambda x, y: max(abs(xi - yi) for xi, yi in zip(x, y))
]
metrics_str = ["euclidean", "manhattan", "chebyshev"]

data = pd.read_csv('wine.csv')
signs = normalize(data.to_numpy()[:, 1::])

targets = np.array(list(map(lambda x: int(x) - 1, data.to_numpy()[:, 0])))

for i in range(len(metrics)):
    clust = KMeans(signs, 3, metrics[i])
    clust.train()
    rand_score = rand(clust.targets, targets)
    sil_score = silhouette(clust.targets, signs, metrics[i], 3)
    print("Metric: " + metrics_str[i], end=',')
    print("Rand: " + str(rand_score), end=', ')
    print("Sil: " + str(sil_score))

# Metric: euclidean,Rand: 0.924585793182251, Sil: 0.30958530110035176
# Metric: manhattan,Rand: 0.9542944201104552, Sil: 0.3368255798273835
# Metric: chebyshev,Rand: 0.8371103916714276, Sil: 0.249747103958116
draw_plot(signs, targets)