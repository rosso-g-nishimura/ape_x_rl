import numpy as np


class Experiences(object):
    alpha = 0.6
    size = 0

    def __init__(self, max_size, gamma, n_step):
        self.max_size = max_size
        self.memory = SumTree(capacity=self.max_size)
        self.n_gamma = gamma ** n_step

    def sample(self, batch_size):
        total = self.memory.total()
        if total < batch_size:
            return None, None

        batch = []
        index = []
        for rand in np.random.uniform(0, total, batch_size):
            (idx, _, tr) = self.memory.get(rand)
            batch.append(tr)
            index.append(idx)

        return batch, index

    def update(self, idx_list_batch, td_list_batch):
        for i in range(len(idx_list_batch)):
            idx_list = idx_list_batch[i]
            td_list = td_list_batch[i]
            for j in range(len(idx_list)):
                idx = idx_list[j]
                td = td_list[j]
                self.memory.update(idx, td)

    def put(self, n_tr_batch, n_td_err):
        for tr, td_err in zip(n_tr_batch, n_td_err):
            priority = td_err ** self.alpha
            self.memory.add(priority, tr)
        self.size += len(n_td_err)

    @property
    def len(self):
        return self.memory.total()


class SumTree:
    write = 0

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    def add(self, p, data):
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, p)
        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

    def update(self, idx, p):
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]
