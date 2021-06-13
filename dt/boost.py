import numpy as np
from tree import Tree


class AdaptiveBoost:
    MAX_TREE_HEIGHT = 5
    TREE_COUNT = 1000

    def __init__(self, dataset, targets, iteration_count):
        self.dataset = dataset
        self.targets = targets
        self.iteration_count = iteration_count
        self.coefficients = None
        self.trees = None

    def train(self):
        weights = self._init_weights()
        self.trees = np.zeros(self.iteration_count, dtype=object)
        trees_spaces = self._init_trees()
        self.coefficients = np.zeros(self.iteration_count)
        for i in range(self.iteration_count):
            best_tree, error, correct_prediction = self._pick_best_tree(weights, trees_spaces)
            self.trees[i] = best_tree
            coefficient = 0.5 * np.log((1 - error) / error)
            self.coefficients[i] = coefficient
            self._recalculate_weights(weights, error, correct_prediction, coefficient)
            self._normallize(weights)

    def predict(self, labels):
        result = 0
        for i in range(self.iteration_count):
            p = 1 if self.trees[i].predict(labels) == 1 else -1
            result += self.coefficients[i] * p
        if result < 0:
            return 0
        return 1

    def _init_trees(self):
        trees = np.empty(0, dtype=object)
        for i in range(AdaptiveBoost.TREE_COUNT):
            trees = np.append(trees, self._new_tree(np.random.randint(1, AdaptiveBoost.MAX_TREE_HEIGHT + 1)))
        return trees

    def _new_tree(self, max_height):
        tree = Tree(self.dataset, self.targets, 2, max_height)
        tree.build_random()
        return tree

    def _init_weights(self):
        dataset_size = len(self.dataset)
        weights = np.empty(dataset_size)
        for i in range(dataset_size):
            weights[i] = 1 / dataset_size
        return weights

    def _pick_best_tree(self, weights, trees):
        least_error = 2  # more than 1
        best_tree = None
        best_correct_prediction = None
        for tree in trees:
            error, correct_prediction = self._compute_error(tree, weights)
            if error < least_error:
                least_error = error
                best_tree = tree
                best_correct_prediction = correct_prediction
        return best_tree, least_error, best_correct_prediction

    def _compute_error(self, tree, weights):
        error = 0
        correct_prediction = np.full(len(self.dataset), True)
        for i in range(len(self.dataset)):
            if tree.predict(self.dataset[i]) != self.targets[i]:
                error += weights[i]
                correct_prediction[i] = False
        return error, correct_prediction

    def _recalculate_weights(self, weights, error, correct_prediction, c):
        positive_coef = np.sqrt(error / (1 - error))
        negative_coef = np.sqrt((1 - error) / error)
        for i in range(len(weights)):
            if correct_prediction[i]:
                weights[i] = weights[i] * positive_coef
            else:
                weights[i] = weights[i] * negative_coef

    def _normallize(self, weights):
        s = sum(weights)
        for i in range(len(weights)):
            weights[i] /= s
