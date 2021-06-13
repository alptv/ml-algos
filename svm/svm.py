import numpy as np
import random as rand


class SVM:
    MAX_ITERATION_COUNT = 20

    def __init__(self, dataset, targets, kernel, C):
        self.dataset = dataset
        self.targets = targets
        self.kernel = kernel
        self.data_size = len(targets)
        self.sign_count = len(dataset[0])
        self.C = C
        self.scalar_products = self._new_scalar_products(dataset)
        self.coefficients = np.zeros(self.data_size)
        self._learn()
        self.free_term = self._compute_free_term()

    def _compute_free_term(self):
        self.free_term = 0
        free_term = 0
        for i in range(self.data_size):
            free_term += self._value(self.dataset[i]) - self.targets[i]
        return free_term / self.data_size

    def _new_scalar_products(self, dataset):
        scalar_products = np.empty([self.data_size, self.data_size])
        for i in range(self.data_size):
            for j in range(self.data_size):
                scalar_products[i][j] = self.kernel(dataset[i], dataset[j])
        return scalar_products

    def _learn(self):
        iteration = 1
        while iteration <= SVM.MAX_ITERATION_COUNT:
            for i in range(self.data_size):
                first = i
                second = rand.randint(0, self.data_size - 1)
                if first != second:
                    self._make_step(first, second)
            iteration += 1

    def _make_step(self, first, second):
        denominator = 2 * self.scalar_products[first][second] - self.scalar_products[first][first] - \
                      self.scalar_products[second][second]

        if denominator < 0:
            line_coefficient = self.targets[second] * self.targets[first]

            first_error = self._predict(first) - self.targets[first]
            second_error = self._predict(second) - self.targets[second]

            delta_second = self.targets[second] * (second_error - first_error) / denominator

            first_value = self.coefficients[first]
            second_value = self.coefficients[second]

            self.coefficients[second] = self._recalculate_value(first_value + line_coefficient * second_value,
                                                                line_coefficient,
                                                                delta_second + second_value)

            self.coefficients[first] = first_value - line_coefficient * (self.coefficients[second] - second_value)

    def _predict(self, index):
        target = 0
        for i in range(len(self.coefficients)):
            target += self.targets[i] * self.coefficients[i] * self.scalar_products[i][index]
        return target

    def _recalculate_value(self, free_term, line_coefficient, try_value):
        upper_bound = 0
        lower_bound = 0
        if line_coefficient == 1 and free_term < self.C:
            upper_bound = free_term
            lower_bound = 0
        if line_coefficient == 1 and free_term >= self.C:
            upper_bound = self.C
            lower_bound = free_term - self.C
        if line_coefficient == -1 and free_term < 0:
            upper_bound = self.C
            lower_bound = -free_term
        if line_coefficient == -1 and free_term >= 0:
            upper_bound = self.C - free_term
            lower_bound = 0
        if try_value > upper_bound:
            return upper_bound
        if try_value < lower_bound:
            return lower_bound
        return try_value

    def _value(self, sign):
        target = 0
        for i in range(len(self.coefficients)):
            target += self.targets[i] * self.coefficients[i] * self.kernel(self.dataset[i], sign)
        return target - self.free_term

    def predict(self, sign):
        return np.sign(self._value(sign))
