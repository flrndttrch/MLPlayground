import random
from collections import namedtuple

import numpy as np

from rl.memory.sumtree import SumTree

transition = namedtuple('transition', ['state', 'action', 'reward', 'next_state', 'mask'])


class BaseMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.position = 0

    def push(self, *args):
        pass

    def sample(self, batch_size):
        pass

    def __len__(self):
        pass


class ReplayBuffer(BaseMemory):
    def __init__(self, capacity):
        super(ReplayBuffer, self).__init__(capacity)
        self.memory = []

    def push(self, *args, error=None):
        if len(self.memory) < self.capacity:
            self.memory.append(None)

        self.memory[self.position] = transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        return _to_dict(batch), None, None

    def __len__(self):
        return len(self.memory)


class PrioritizedReplayBuffer(BaseMemory):
    def __init__(self, capacity, eps=0.01, alpha=0.6, beta=0.4, beta_inc=0.001):
        super(PrioritizedReplayBuffer, self).__init__(capacity)
        self.tree = SumTree(capacity)
        self.eps = eps
        self.alpha = alpha
        self.beta = beta
        self.beta_inc = beta_inc

    def _get_priority(self, error):
        return (error + self.eps) ** self.alpha

    def push(self, *args, error=None):
        sample = transition(*args)
        p = self._get_priority(error)
        self.tree.add(p, sample)

    def sample(self, batch_size):
        batch = []
        idxs = []
        segment = self.tree.total() / batch_size
        priorities = []

        self.beta = np.min([1., self.beta + self.beta_inc])

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            priorities.append(p)
            batch.append(data)
            idxs.append(idx)

        sampling_probs = priorities / self.tree.total()
        is_weights = np.power(self.tree.n_entries * sampling_probs, -self.beta)
        is_weights /= is_weights.max()

        batch_dict = _to_dict(batch)

        return batch_dict, idxs, is_weights

    def update(self, idx, error):
        p = self._get_priority(error)
        self.tree.update(idx, p)

    def __len__(self):
        return self.tree.n_entries


def _to_dict(batch):
    batch_dict = dict(states=[], actions=[], rewards=[], next_states=[], masks=[])
    for b in batch:
        batch_dict["states"].append(b.state)
        batch_dict["actions"].append(b.action)
        batch_dict["rewards"].append(b.reward)
        batch_dict["next_states"].append(b.next_state)
        batch_dict["masks"].append(b.mask)
    return batch_dict
