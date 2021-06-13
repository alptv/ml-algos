import numpy as np


class NaiveSpamBayes:
    SPAM_MARK = 0
    LEGIT_MARK = 1
    MARK_COUNT = 2
    LOG_EPS = 1e-200

    def __init__(self, dataset, targets, lambdas, alpha):
        self.dataset = dataset
        self.targets = targets
        self.lambdas = lambdas
        self.alpha = alpha
        self.class_sizes = self._compute_class_sizes(targets)
        self.sign_probabilities = self._compute_sign_probabilities(dataset, targets, alpha)

    def _compute_class_sizes(self, targets):
        class_sizes = np.zeros(NaiveSpamBayes.MARK_COUNT)
        for target in targets:
            class_sizes[target] += 1
        return class_sizes

    def _compute_sign_probabilities(self, dataset, targets, alpha):
        sign_count = len(dataset[0])
        sign_probabilities = np.zeros([sign_count, NaiveSpamBayes.MARK_COUNT])

        for i in range(len(dataset)):
            for j in range(sign_count):
                sign_probabilities[j][targets[i]] += dataset[i][j]

        for i in range(NaiveSpamBayes.MARK_COUNT):
            for j in range(sign_count):
                sign_probabilities[j][i] = (sign_probabilities[j][i] + alpha) / (self.class_sizes[i] + 2 * alpha)
        return sign_probabilities

    def predict(self, sign):
        spam_weight = self._compute_weight(sign, NaiveSpamBayes.SPAM_MARK)
        legit_weight = self._compute_weight(sign, NaiveSpamBayes.LEGIT_MARK)
        if legit_weight > spam_weight:
            return NaiveSpamBayes.LEGIT_MARK
        return NaiveSpamBayes.SPAM_MARK

    def legit_probability(self, sign):
        spam_weight = self._compute_weight(sign, NaiveSpamBayes.SPAM_MARK)
        legit_weight = self._compute_weight(sign, NaiveSpamBayes.LEGIT_MARK)
        return legit_weight / (spam_weight + legit_weight)

    def _compute_weight(self, prediction_signs, mark):
        weight = self.lambdas[mark] * self.class_sizes[mark]
        for i in range(len(prediction_signs)):
            sign_probability = self.sign_probabilities[i][mark]
            if prediction_signs[i] == 0:
                weight *= 1 - sign_probability
            else:
                weight += sign_probability
        return weight

    def _ln(self, x):
        if x < NaiveSpamBayes.LOG_EPS:
            x = NaiveSpamBayes.LOG_EPS
        return np.log(x)

    def _exp(self, x):
        return np.exp(x)
