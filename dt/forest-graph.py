from forest import Forest
from read import accuracy, read_data


for i in range(1, 22):
    tr_data, tr_targ, tr_count, t_data, t_targ, t_count = read_data(i)
    forest = Forest(tr_data, tr_targ, tr_count + 1, 30)
    forest.build()
    print("Dataset:", i, "Accuracy:", accuracy(t_data, t_targ, forest))