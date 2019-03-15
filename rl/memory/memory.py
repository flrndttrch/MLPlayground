import random
from collections import namedtuple
from typing import NamedTuple

transition = namedtuple('transition', ['state', 'action', 'reward', 'next_state', 'mask'])

class BaseMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.position = 0

    def push(self, *args):
        pass

    def sample(self, batch_size):
        pass

class ReplayBuffer(BaseMemory):
    def __init__(self, capacity):
        super(ReplayBuffer, self).__init__(capacity)
        self.memory = []

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)

        self.memory[self.position] = transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        return self._to_dict(batch)

    def _to_dict(self, batch):
        batch_dict = dict(states=[], actions=[], rewards=[], next_states=[], masks=[])
        for b in batch:
            batch_dict["states"].append(b.state)
            batch_dict["actions"].append(b.action)
            batch_dict["rewards"].append(b.reward)
            batch_dict["next_states"].append(b.next_state)
            batch_dict["masks"].append(b.mask)
        return batch_dict

    def __len__(self):
        return len(self.memory)

class PrioritizedReplayBuffer():
    def __init__(self, capacity):
        pass