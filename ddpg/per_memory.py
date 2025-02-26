import torch
import numpy as np

class SumTree:
    """
    A data structure for storing priorities in a segment tree format.
    Each leaf correspond to an index, and internal nodes holds the sum of their children.
    This layout allow searching for a specific prefix-sum in O(log N).
    """
    def __init__(self, capacity):
        """
        Build the tree array. 'capacity' is the maximum number of items.
        The actual array sizes become the next power of two >= capacity.
        """
        self.capacity = capacity
        self.size = 1
        while self.size < capacity:
            self.size *= 2
        self.tree = np.zeros(2 * self.size, dtype=np.float32)

    def update(self, idx, value):
        """
        Set the priority at position 'idx' to 'value' and updates all parent sums
        """
        pos = idx + self.size
        self.tree[pos] = value
        pos //= 2
        while pos >= 1:
            self.tree[pos] = self.tree[2 * pos] + self.tree[2 * pos + 1]
            pos //= 2

    def total(self):
        """
        Returns the sum of all priorities, stored at the root of the tree.
        """
        return self.tree[1]

    def sample(self, prefix_sum):
        """
        Finds the highest index whose cumulative sum is >= prefix_sum.
        Return an index in [0, capacity - 1] such that it is clamped if it exceed capacity - 1.
        """
        pos = 1
        while pos < self.size:
            left = 2 * pos
            if self.tree[left] >= prefix_sum:
                pos = left
            else:
                prefix_sum -= self.tree[left]
                pos = left + 1
        idx = pos - self.size
        if idx >= self.capacity:
            idx = self.capacity - 1
        return idx

class PERMemory:
    """
    A replay buffer that store transitions on the GPU and sample them according to priority.
    It uses a SumTree to handle priority-based sampling in O(log N).
    """
    def __init__(self, obs_dim, act_dim, max_size=100000,
                 alpha=0.6, beta=0.4, beta_increment=1e-5,
                 device="cuda:0"):
        """
        Sets ups the replay buffer's size, priority parameters, and GPU arrays.
        'alpha' controls how strongly priorities affect sampling.
        'beta' controls importance sampling, and grows by 'beta_increment' until it reaches 1.
        """
        self.device = device
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.eps = 1e-5
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        self.states = torch.zeros((max_size, obs_dim), dtype=torch.float32, device=self.device)
        self.actions = torch.zeros((max_size, act_dim), dtype=torch.float32, device=self.device)
        self.rewards = torch.zeros((max_size, 1), dtype=torch.float32, device=self.device)
        self.next_states = torch.zeros((max_size, obs_dim), dtype=torch.float32, device=self.device)
        self.dones = torch.zeros((max_size, 1), dtype=torch.float32, device=self.device)

        self.priorities = np.zeros(max_size, dtype=np.float32)
        self.tree = SumTree(max_size)

    def add_transition(self, transition):
        """
        Adds one transition (state, action, reward, next_state, done) with the highest priority.
        This increses the chance of sampling newly inserted transitions.
        """
        state, action, reward, next_state, done = transition

        self.states[self.ptr] = torch.tensor(state, dtype=torch.float32, device=self.device)
        self.actions[self.ptr] = torch.tensor(action, dtype=torch.float32, device=self.device)
        self.rewards[self.ptr] = torch.tensor([reward], dtype=torch.float32, device=self.device)
        self.next_states[self.ptr] = torch.tensor(next_state, dtype=torch.float32, device=self.device)
        self.dones[self.ptr] = torch.tensor([done], dtype=torch.float32, device=self.device)

        max_priority = self.priorities.max() if self.size > 0 else 1.0
        self.priorities[self.ptr] = max_priority
        self.tree.update(self.ptr, (max_priority + self.eps) ** self.alpha)

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def get_size(self):
        """
        Return the current number of stored transitions.
        """
        return self.size

    def sample(self, batch_size):
        """
        Samples a batch of transitions according to priority.
        Returns (states, actions, rewards, next_states, dones, weights, indices).
        'weights' are for importance-sampling corrections (not the network weights).
        'indices' let user updates priorities later.
        """
        if batch_size > self.size:
            batch_size = self.size
        indices = np.zeros(batch_size, dtype=np.int32)
        segment = self.tree.total() / batch_size
        self.beta = min(1.0, self.beta + self.beta_increment)

        for i in range(batch_size):
            prefix_sum = np.random.uniform(i * segment, (i + 1) * segment)
            idx = self.tree.sample(prefix_sum)
            indices[i] = idx

        probs = (self.priorities[indices] + self.eps) ** self.alpha
        total_p = self.tree.total()
        probs /= (total_p + 1e-8)

        weights = (self.size * probs) ** (-self.beta)
        weights /= weights.max()

        batch_states = self.states[indices]
        batch_actions = self.actions[indices]
        batch_rewards = self.rewards[indices]
        batch_next_states = self.next_states[indices]
        batch_dones = self.dones[indices]
        weights_t = torch.tensor(weights, dtype=torch.float32, device=self.device).unsqueeze(-1)

        return (batch_states, batch_actions, batch_rewards,
                batch_next_states, batch_dones, weights_t, indices)

    def update_priorities(self, indices, td_errors):
        """
        Updates the priorities of sampled transitions using TD-errors.
        'td_errors' should be an array that matches 'indices' otherwise problems arise.
        """
        td_errors = np.abs(td_errors) + self.eps
        self.priorities[indices] = td_errors
        for i, prio in zip(indices, td_errors):
            self.tree.update(i, prio ** self.alpha)
