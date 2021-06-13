import math
import numpy as np


class Tree:
    INF_HEIGHT = int(1e9)

    class Node:
        def __init__(self, label_number, predicate_value, left, right):
            self.label_number = label_number
            self.predicate_value = predicate_value
            self.left = left
            self.right = right

        def isLeaf(self):
            return False

    class Leaf:
        def __init__(self, class_number):
            self.class_number = class_number

        def isLeaf(self):
            return True

    class SplitInfo:

        def __init__(self):
            self.score = 0
            self.label = 0
            self.separator_index = 0
            self.predicative_value = 0
            self.indexes = None
            self.empty = True

        def update_info(self, info):
            self.update_values(info.score, info.label, info.separator_index, info.predicative_value, info.indexes)

        def update_values(self, score, label, separator_index, predicative_value, indexes):
            if self.empty or score < self.score:
                self.score = score
                self.label = label
                self.separator_index = separator_index
                self.predicative_value = predicative_value
                self.indexes = indexes
                self.empty = False

    def random_label_generator(label_count):
        used_labels_count = int(np.sqrt(label_count))
        x = np.random.choice(label_count, used_labels_count, False)
        return x

    def default_label_generator(label_count):
        return range(label_count)

    def __init__(self, dataset, targets, class_count, max_height=INF_HEIGHT, label_generator=default_label_generator):
        self.dataset = dataset
        self.targets = targets
        self.class_count = class_count
        self.label_count = len(dataset[0])
        self.max_height = max_height
        self.label_generator = label_generator
        self.root = None

    def build(self):
        self.root = self._build_recursively(self.dataset, self.targets, 0, self._entropy_split)

    def build_random(self):
        self.root = self._build_recursively(self.dataset, self.targets, 0, self._random_split)

    def _build_recursively(self, dataset, targets, height, splitter):
        if height == self.max_height:
            return Tree.Leaf(self._most_probable_target(targets))
        if self._is_one_class(targets):
            return Tree.Leaf(targets[0])

        split_info, dataset_l, targets_l, dataset_r, targets_r = splitter(
            dataset, targets)

        if len(dataset_l) == 0 or len(dataset_r) == 0:
            return Tree.Leaf(self._most_probable_target(targets))

        left = self._build_recursively(dataset_l, targets_l, height + 1, splitter)
        right = self._build_recursively(dataset_r, targets_r, height + 1, splitter)
        return Tree.Node(split_info.label, split_info.predicative_value, left, right)

    def _random_split(self, dataset, targets):
        label_number = np.random.randint(0, self.label_count)
        labels, object_indexes = self._sort_with_label(dataset, label_number)
        separator_index = np.random.randint(0, len(dataset))

        split_info = Tree.SplitInfo()
        split_info.update_values(0, label_number, separator_index,
                                 labels[separator_index], object_indexes)
        dataset_l, targets_l, dataset_r, targets_r = self._split_data(dataset, targets, split_info)
        return split_info, dataset_l, targets_l, dataset_r, targets_r

    def _entropy_split(self, dataset, targets):
        split_info = Tree.SplitInfo()
        for label_number in self.label_generator(self.label_count):
            split_info.update_info(self._label_entropy_split(label_number, dataset, targets))

        dataset_l, targets_l, dataset_r, targets_r = self._split_data(dataset, targets, split_info)

        return split_info, dataset_l, targets_l, dataset_r, targets_r

    def _label_entropy_split(self, label_number, dataset, targets):
        best_split_info = Tree.SplitInfo()
        dataset_size = len(dataset)
        labels, object_indexes = self._sort_with_label(dataset, label_number)

        left_counts, right_counts = self._init_counts(targets)
        left_score, right_score = self._init_scores(right_counts, dataset_size)
        for i in range(0, dataset_size):
            score = left_score + right_score
            best_split_info.update_values(score, label_number, i, labels[i], object_indexes)
            moving_target = targets[object_indexes[i]]
            left_score = self._update_score(left_score, left_counts[moving_target], i, 1)
            right_score = self._update_score(right_score, right_counts[moving_target], dataset_size - i, -1)
            left_counts[moving_target] += 1
            right_counts[moving_target] -= 1
        return best_split_info

    def _update_score(self, prev, count, size, direction):
        return prev - self._score(count) + self._score(count + direction) + self._score(size) - self._score(
            size + direction)

    def _score(self, value):
        if value == 0:
            return 0
        return -value * math.log2(value)

    def _init_counts(self, targets):
        count_right = np.zeros(self.class_count)
        for target in targets:
            count_right[target] += 1
        return np.zeros(self.class_count), count_right

    def _init_scores(self, right_counts, dataset_size):
        right_score = 0
        for count in right_counts:
            if count != 0:
                right_score -= count * np.log2(count / dataset_size)
        return 0, right_score

    def _is_one_class(self, targets):
        for i in range(1, len(targets)):
            if targets[i] != targets[i - 1]:
                return False
        return True

    def _sort_with_label(self, dataset, number):
        dataset_size = len(dataset)
        temp = [(dataset[i][number], i) for i in range(dataset_size)]
        temp.sort()
        labels = [t[0] for t in temp]
        object_indexes = [t[1] for t in temp]
        return labels, object_indexes

    def _split_data(self, dataset, targets, split_info):
        separator_index = split_info.separator_index
        indexes = split_info.indexes
        dataset_l, targets_l = self._split_on_part(dataset, targets, indexes, 0, separator_index)
        dataset_r, targets_r = self._split_on_part(dataset, targets, indexes, separator_index, len(dataset))
        return dataset_l, targets_l, dataset_r, targets_r

    def _split_on_part(self, dataset, targets, object_indexes, start, end):
        new_dataset = np.empty(end - start, dtype=object)
        new_targets = np.empty(end - start, dtype=int)
        for i in range(start, end):
            object_index = object_indexes[i]
            new_dataset[i - start] = dataset[object_index]
            new_targets[i - start] = targets[object_index]
        return new_dataset, new_targets

    def _most_probable_target(self, targets):
        counts = np.zeros(self.class_count)
        for i in range(len(targets)):
            counts[targets[i]] += 1
        return np.argmax(counts)

    def predict(self, labels):
        return self._predict_recursively(labels, self.root)

    def _predict_recursively(self, labels, node):
        if node.isLeaf():
            return node.class_number
        if labels[node.label_number] < node.predicate_value:
            return self._predict_recursively(labels, node.left)
        else:
            return self._predict_recursively(labels, node.right)
