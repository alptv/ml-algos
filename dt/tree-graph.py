from tree import Tree
from read import read_data, accuracy
import numpy as np
import matplotlib.pyplot as plt


def process_dataset(number):
    tr_data, tr_targ, tr_count, t_data, t_targ, t_count = read_data(number)
    best_a = 0
    best_h = 0
    for h in range(1, 11):
        tree = Tree(tr_data, tr_targ, tr_count + 1, h)
        tree.build()
        a = accuracy(t_data, t_targ, tree)
        if a > best_a:
            best_a = a
            best_h = h
    print("Number: ", number)
    print("Acc: ", best_a)
    print("Height: ", best_h)
    print("===============")
    return best_a, best_h


def get_min_max_height():
    min_h = 100
    min_h_a = 0
    max_h = 0
    max_h_a = 0
    for i in range(1, 22):
        best_a, best_h = process_dataset(i)
        if best_h < min_h or (best_h == min_h and best_a > min_h_a):
            min_h_a = best_a
            min_h = best_h
        if best_h > max_h or (best_h == max_h and best_a > max_h_a):
            max_h_a = best_a
            max_h = best_h
    return min_h, min_h_a, max_h, max_h_a


#print(get_min_max_height())


# (1, 1.0, 10, 0.8692113387594723)
# Number:  12
# Acc:  0.8692113387594723
# Height:  10

# Number:  3
# Acc:  1.0
# Height:  1

def draw_plot(number, title=None):
    tr_data, tr_targ, tr_count, t_data, t_targ, t_count = read_data(number)
    h_count = 20
    accs = np.empty(h_count)
    hs = np.empty(h_count)
    for h in range(1, h_count + 1):
        tree = Tree(tr_data, tr_targ, tr_count + 1, h)
        tree.build()
        hs[h - 1] = h
        accs[h - 1] = accuracy(t_data, t_targ, tree)
        print(h, end=' ')
    print()
    plt.plot(hs, accs)
    if title is not None:
        plt.title(title)
        plt.savefig(title)
    plt.show()


draw_plot(12, '12 dataset')
draw_plot(3, '3 dataset')
