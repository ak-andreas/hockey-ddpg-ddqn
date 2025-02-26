import numpy as np
import random
import torch

# class to store transitions
class Memory():
    def __init__(self, max_size=100000):
        self.transitions = np.asarray([])
        self.size = 0
        self.current_idx = 0
        self.max_size=max_size

    def add_transition(self, transitions_new):
        if self.size == 0:
            blank_buffer = [np.asarray(transitions_new, dtype=object)] * self.max_size
            self.transitions = np.asarray(blank_buffer)

        self.transitions[self.current_idx,:] = np.asarray(transitions_new, dtype=object)
        self.size = min(self.size + 1, self.max_size)
        self.current_idx = (self.current_idx + 1) % self.max_size

    def sample(self, batch=1):
        if batch > self.size:
            batch = self.size
        self.inds=np.random.choice(range(self.size), size=batch, replace=False)
        return self.transitions[self.inds,:]

    def get_all_transitions(self):
        return self.transitions[0:self.size]
    

class PrioritizedMemory():
    def __init__(self, max_size=100000, alpha=0.6, beta=0.4, beta_frames=100000, epsilon=1e-5):
        self.tree = SumTree(max_size)
        self.max_size = max_size
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = (1.0 - beta) / beta_frames
        self.epsilon = epsilon
        self.max_priority = 1.0

    def add_transition(self, transitions_new):
        priority = self.max_priority ** self.alpha
        self.tree.add(priority, np.asarray(transitions_new, dtype=object))

    def sample(self, batch=1):
        batch_data = []
        indices = []
        priorities = []
        segment = self.tree.total_priority() / batch

        for i in range(batch):
            value = random.uniform(i * segment, (i + 1) * segment)
            index, priority, data = self.tree.get_leaf(value)
            batch_data.append(data)
            indices.append(index)
            priorities.append(priority)

        self.beta = min(1.0, self.beta + self.beta_increment)

        probs = np.array(priorities) / self.tree.total_priority()
        weights = (len(self.tree.data) * probs) ** (-self.beta)
        weights /= weights.max()

        return np.array(batch_data), indices, torch.tensor(weights, dtype=torch.float32)
    
    def update_priorities(self, indices, errors):
        priorities = (np.abs(errors) + self.epsilon) ** self.alpha
        for idx, priority in zip(indices, priorities):
            self.tree.update(idx, priority)
            self.max_priority = max(self.max_priority, priority)

    

class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)  # Sum tree
        self.data = np.zeros(capacity, dtype=object)  # Stores (state, action, reward, next_state, done)
        self.size = 0
        self.pointer = 0

    def add(self, priority, data):
        index = self.pointer + self.capacity - 1
        self.data[self.pointer] = data
        self.update(index, priority)

        self.pointer = (self.pointer + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def update(self, index, priority):
        change = priority - self.tree[index]
        self.tree[index] = priority
        while index != 0:
            index = (index - 1) // 2
            self.tree[index] += change

    def get_leaf(self, value):
        index = 0
        while index < self.capacity - 1:
            left = 2 * index + 1
            right = left + 1
            if value <= self.tree[left]:
                index = left
            else:
                value -= self.tree[left]
                index = right
        data_index = index - (self.capacity - 1)
        return index, self.tree[index], self.data[data_index]

    def total_priority(self):
        return self.tree[0]