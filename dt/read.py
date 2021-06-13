import pandas as pd


def split(data):
    dataset = data.to_numpy()[:, :-1:]
    targets = data.to_numpy()[:, -1]
    return dataset, targets, max(targets)


def read_data(number):
    number_str = str(number) if number >= 10 else "0" + str(number)
    data_train = pd.read_csv("/home/alexander/Projects/Python/dt/data/" + number_str + "_train.csv")
    tr_data, tr_targ, tr_count = split(data_train)
    data_test = pd.read_csv("/home/alexander/Projects/Python/dt/data/" + number_str + "_test.csv")
    t_data, t_targ, t_count = split(data_test)
    return tr_data, tr_targ, tr_count, t_data, t_targ, t_count


def accuracy(dataset, targets, predictor):
    res = 0
    for i in range(len(dataset)):
        pred = predictor.predict(dataset[i])
        if pred == targets[i]:
            res += 1
    return res / len(dataset)
