import numpy as np
from tree import Tree


class Forest:

    def __init__(self, dataset, targets, class_count, size):
        self.dataset = dataset
        self.targets = targets
        self.class_count = class_count
        self.size = size
        self.trees = np.empty(size, dtype=object)

    def build(self):
        for i in range(self.size):
            data_subset, targets_subset = self._data_subset()
            t = Tree(data_subset, targets_subset, self.class_count, Tree.INF_HEIGHT,
                     Tree.random_label_generator)
            t.build()
            self.trees[i] = t

    def predict(self, labels):
        counts = np.zeros(self.class_count, dtype=int)
        for i in range(self.size):
            predict_target = self.trees[i].predict(labels)
            counts[predict_target] += 1
        return np.argmax(counts)

    def _data_subset(self):
        data_size = len(self.dataset)
        labels_count = len(self.dataset[0])
        data_subset = np.ndarray([data_size, labels_count])
        targets_subset = np.empty(data_size, dtype=int)
        for i in range(data_size):
            cur = np.random.randint(0, data_size)
            data_subset[i] = self.dataset[cur]
            targets_subset[i] = self.targets[i]
        return data_subset, targets_subset
