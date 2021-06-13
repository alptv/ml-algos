import numpy as np


class KMeans:
    EPS = 0.01
    INF = 1e9

    def __init__(self, signs, class_count, metric):
        self.signs = signs
        self.size = len(signs)
        self.class_count = class_count
        self.metric = metric
        self.targets = None

    def _dist(self, i, j):
        return self.metric(self.signs[i], self.signs[j])

    def _find_centers(self):
        centers = np.empty([1, len(self.signs[0])])
        centers[0] = self.signs[np.random.randint(0, self.size)]
        while len(centers) != self.class_count:
            min_dist_from_centers = np.full(self.size, KMeans.INF)
            for i in range(self.size):
                for center in centers:
                    min_dist_from_centers[i] = min(min_dist_from_centers[i], self.metric(self.signs[i], center))
            centers = np.vstack([centers, self._find_center(min_dist_from_centers)])
        return centers

    def _find_center(self, min_dist_from_centers):
        sum_dist = sum(min_dist_from_centers)
        probabilities = min_dist_from_centers / sum_dist
        selector = np.random.uniform()
        left = 0
        for i in range(len(probabilities)):
            p = probabilities[i]
            right = left + p
            if left <= selector <= right:
                return self.signs[i]
            left = right

    def train(self):
        current_centers = self._find_centers()
        while True:
            obj_centres = np.zeros(self.size, dtype=int)
            for i in range(self.size):
                for j in range(len(current_centers)):
                    if self.metric(current_centers[j], self.signs[i]) < self.metric(current_centers[obj_centres[i]],
                                                                                    self.signs[i]):
                        obj_centres[i] = j
            next_centres = np.zeros([self.class_count, len(self.signs[0])])
            class_size = np.zeros(self.class_count, dtype=int)
            for i in range(self.size):
                next_centres[obj_centres[i]] += self.signs[i]
                class_size[obj_centres[i]] += 1
            for i in range(self.class_count):
                next_centres[i] = next_centres[i] / class_size[i]
            if self._is_close(next_centres, current_centers):
                self.targets = obj_centres
                break
            current_centers = next_centres

    def _is_close(self, next_centres, current_centers):
        for i in range(self.class_count):
            close = True
            for j in range(len(next_centres)):
                if abs(next_centres[i][j] - current_centers[i][j]) >= KMeans.EPS:
                    close = False
            if not close:
                return False
        return True
