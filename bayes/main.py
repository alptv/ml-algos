import os
import numpy as np
import bayes
import ngram
import matplotlib.pyplot as plt

NGRAMS = [1]
ALPHA = [0.01, 0.1, 0.2, 0.3, 0.6, 1]
SPAM_C = 1
LEGIT_C = [0.2, 0.5, 1, 1.5, 2, 3]


def get_path(*parts):
    return '/'.join(parts)


def get_parts_dirs():
    parts_dirs = os.listdir('messages')
    parts_dirs.sort()
    parts_dirs = list(map(lambda part: get_path('messages', part), parts_dirs))
    return parts_dirs


def read_data():
    parts_dirs = get_parts_dirs()
    parts = np.empty(len(parts_dirs), dtype=object)
    for i in range(len(parts_dirs)):
        files = os.listdir(parts_dirs[i])
        parts[i] = np.empty(len(files), dtype=object)
        for j in range(len(files)):
            with open(get_path(parts_dirs[i], files[j])) as data:
                is_legit = int('legit' in files[j])
                subject = np.array(list(map(int, data.readline().split()[1:])))
                data.readline()
                message = np.array(list(map(int, data.readline().split())))
                parts[i][j] = (subject, message, is_legit)
    return parts


def to_ngrams(data, n):
    ngrams = np.empty(len(data), dtype=object)
    for i in range(len(data)):
        ngrams[i] = np.empty(len(data[i]), dtype=object)
        for j in range(len(data[i])):
            subject = ngram.split(data[i][j][0], n)
            message = ngram.split(data[i][j][1], n)
            ngrams[i][j] = (np.unique(np.append(subject, message)), data[i][j][2])
    return ngrams


def make_signs_ngrams(ngrams, i):
    res = np.empty(0)
    for j in range(len(ngrams)):
        if j != i:
            res = np.append(res, ngrams[i][j][0])
    return np.unique(res)


def make_test(test_part, signs_ngrams):
    test_targ = np.empty(len(test_part), dtype='int8')
    test_data = np.empty(len(test_part), dtype=object)
    for i in range(len(test_part)):
        test_targ[i] = test_part[i][1]
        test_data[i] = ngram.to_vector(test_part[i][0], signs_ngrams)
    return test_data, test_targ


def make_learn_data(ngrams, signs_ngrams, skip):
    learn_data = np.empty(0, dtype=object)
    learn_targ = np.empty(0, dtype='int8')
    for i in range(len(ngrams)):
        if i == skip:
            continue
        shift = len(learn_data)
        learn_data = np.append(learn_data, np.empty(len(ngrams[i]), dtype=object))
        learn_targ = np.append(learn_targ, np.empty(len(ngrams[i]), dtype='int8'))
        for j in range(len(ngrams[i])):
            learn_data[shift + j] = ngram.to_vector(ngrams[i][j][0], signs_ngrams)
            learn_targ[shift + j] = ngrams[i][j][1]
    return learn_data, learn_targ


def process():
    data = read_data()
    bests = {}
    for n in NGRAMS:
        ngrams = to_ngrams(data, n)
        for i in range(len(ngrams)):
            signs_ngrams = make_signs_ngrams(ngrams, i)
            test_data, test_targ = make_test(ngrams[i], signs_ngrams)
            learn_data, learn_targ = make_learn_data(ngrams, signs_ngrams, i)
            for a in ALPHA:
                for leg_c in LEGIT_C:
                    predictor = bayes.NaiveSpamBayes(learn_data, learn_targ, [SPAM_C, leg_c], a)
                    t = 0
                    f = 0
                    legit_error = 0
                    for j in range(len(test_targ)):
                        if predictor.predict(test_data[j]) == test_targ[j]:
                            t += 1
                        else:
                            f += 1
                            if test_targ[j] == 1:
                                legit_error += 1
                    p = (n, a, leg_c)
                    print(p)
                    print(t / (t + f))
                    if p in bests:
                        bests[p] += t / (t + f)
                    else:
                        bests[p] = t / (t + f)
    for p in bests.keys():
        bests[p] /= 10
    return bests


def draw_roc(n, i, leg_c, a):
    data = read_data()
    ngrams = to_ngrams(data, n)
    signs_ngrams = make_signs_ngrams(ngrams, i)
    test_data, test_targ = make_test(ngrams[i], signs_ngrams)
    learn_data, learn_targ = make_learn_data(ngrams, signs_ngrams, i)
    predictor = bayes.NaiveSpamBayes(learn_data, learn_targ, [SPAM_C, leg_c], a)
    roc_data = []
    neg = 0
    pos = 0
    for i in range(len(test_targ)):
        prob = (1 - predictor.legit_probability(test_data[i]))
        roc_data.append((prob, test_targ[i]))
        neg += 1 - test_targ[i]
        pos += test_targ[i]
    roc_data.sort()
    print(roc_data)
    roc_x = [0]
    roc_y = [0]
    prev_x = 0
    prev_y = 0
    for i in range(len(roc_data)):
        if roc_data[i][1] == 1:
            prev_y += 1 / pos
        else:
            prev_x += 1 / neg
        roc_x.append(prev_x)
        roc_y.append(prev_y)
    plt.plot(roc_x, roc_y)
    plt.title('roc curve')
    plt.savefig('roc-curve')
    plt.show()

#best params
i = 0
leg_c = 1
a = 0.01
draw_roc(2, i, leg_c, a)
