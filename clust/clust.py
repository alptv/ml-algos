from enum import Enum
import numpy as np

ObjectType = Enum("ObjectType", "noise root bound unmarked")


class DBSCAN:

    def __init__(self, data, metric, radius, noise_bound):
        self.data = np.copy(data)
        self.data_size = len(data)
        self.metric = metric
        self.radius = radius
        self.noise_bound = noise_bound
        self.object_types = None
        self.classes = None
        self.class_count = 0

    def fit(self):
        self.object_types = np.full(self.data_size, ObjectType.unmarked)
        self.classes = np.full(self.data_size, -1, dtype=int)
        self.class_count = 0
        unmarked_count = self.data_size
        while unmarked_count != 0:
            unmarked_obj_index = self._get_unmarked_obj_index(unmarked_count)
            neighbors_indexes = self._get_neighbors_indexes(unmarked_obj_index)
            if len(neighbors_indexes) < self.noise_bound:
                self.object_types[unmarked_obj_index] = ObjectType.noise
                unmarked_count -= 1
            else:
                unmarked_count = self._add_new_cluster(unmarked_obj_index, neighbors_indexes, unmarked_count)

    def _get_unmarked_obj_index(self, unmarked_count):
        unmarked_index = np.random.randint(0, unmarked_count)
        for i in range(self.data_size):
            if self.object_types[i] == ObjectType.unmarked:
                if unmarked_index == 0:
                    return i
                else:
                    unmarked_index -= 1

    def _get_neighbors_indexes(self, obj_index):
        index = np.empty(0, dtype=int)
        for neighbour_index in range(self.data_size):
            neighbour_type = self.object_types[neighbour_index]
            d = self._dist(obj_index, neighbour_index)
            if (neighbour_type == ObjectType.unmarked or neighbour_type == ObjectType.noise) and self._is_neighbours(
                    obj_index, neighbour_index):
                index = np.append(index, neighbour_index)
        return index

    def _dist(self, i, j):
        return self.metric(self.data[i], self.data[j])

    def _is_neighbours(self, i, j):
        return self._dist(i, j) < self.radius

    def _add_new_cluster(self, obj_index, neighbors_indexes, unmarked_count):
        self.classes[obj_index] = self.class_count
        self.object_types[obj_index] = ObjectType.root
        unmarked_count -= 1

        used = self._start_used(obj_index, neighbors_indexes)
        i = 0
        while i < len(neighbors_indexes):
            neighbour_ind = neighbors_indexes[i]
            self.classes[neighbour_ind] = self.class_count
            if self.object_types[neighbour_ind] == ObjectType.unmarked:
                unmarked_count -= 1
            sub_neighbors_indexes = self._get_neighbors_indexes(neighbour_ind)
            if len(sub_neighbors_indexes) >= self.noise_bound:
                self.object_types[neighbour_ind] = ObjectType.root
                self._merge(neighbors_indexes, sub_neighbors_indexes, used)
            else:
                self.object_types[neighbour_ind] = ObjectType.bound
            i += 1
        self.class_count += 1
        return unmarked_count

    def _start_used(self, obj_index, neighbors_indexes):
        used = np.full(self.data_size, False)
        used[obj_index] = True
        for i in neighbors_indexes:
            used[i] = True
        return used

    def _merge(self, neighbors_indexes, sub_neighbors_indexes, used):
        for i in sub_neighbors_indexes:
            if not used[i]:
                neighbors_indexes = np.append(neighbors_indexes, i)
        for i in sub_neighbors_indexes:
            used[i] = True
