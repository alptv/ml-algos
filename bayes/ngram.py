import numpy as np


class NGram:

    def __init__(self, words):
        self.words = np.array(words, dtype='int64')

    def __getitem__(self, i):
        return self.words[i]

    def __len__(self):
        return len(self.words)

    def __str__(self):
        return str(self.words)

    def __hash__(self):
        return hash(str(self))

    def __eq__(self, other):
        for i in range(len(other)):
            if self[i] != other[i]:
                return False
        return True

    def __lt__(self, other):
        for i in range(len(other)):
            if self[i] >= other[i]:
                return False
        return True


def split(message, n):
    ngram_count = max(len(message) - n + 1, 0)
    ngrams = np.empty(ngram_count, dtype=object)
    for i in range(ngram_count):
        ngrams[i] = NGram(message[i:i + n])
    return np.unique(ngrams)


def to_vector(message_ngr, ngrams):
    vector = np.zeros(len(ngrams), dtype='int8')
    message_ngrams = set(message_ngr)
    for i in range(len(ngrams)):
        if ngrams[i] in message_ngrams:
            vector[i] = 1
    return vector
