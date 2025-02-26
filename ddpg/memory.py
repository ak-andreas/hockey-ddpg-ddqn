import torch

class Memory:
    """
    Replay buffer that stores all data directly on the GPU.
    Expects observation and action dimensions (obs_dim, act_dim) to be provided
    when creating the Memory instance.
    """
    def __init__(self, obs_dim, act_dim, max_size=100000, device="cuda:0"):
        """
        Initializes the replay buffer with the given dimensions, maximum size,
        and device for storing tensors.
        """
        self.device = device
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.states = torch.zeros((max_size, obs_dim), dtype=torch.float32, device=self.device)
        self.actions = torch.zeros((max_size, act_dim), dtype=torch.float32, device=self.device)
        self.rewards = torch.zeros((max_size, 1), dtype=torch.float32, device=self.device)
        self.next_states = torch.zeros((max_size, obs_dim), dtype=torch.float32, device=self.device)
        self.dones = torch.zeros((max_size, 1), dtype=torch.float32, device=self.device)

    def add_transition(self, transition):
        """
        Adds a single transition to the replay buffer.
        Expects transition to be a tuple (state, action, reward, next_state, done).
        """
        state, action, reward, next_state, done = transition

        self.states[self.ptr] = torch.tensor(state, dtype=torch.float32, device=self.device)
        self.actions[self.ptr] = torch.tensor(action, dtype=torch.float32, device=self.device)
        self.rewards[self.ptr] = torch.tensor([reward], dtype=torch.float32, device=self.device)
        self.next_states[self.ptr] = torch.tensor(next_state, dtype=torch.float32, device=self.device)
        self.dones[self.ptr] = torch.tensor([done], dtype=torch.float32, device=self.device)

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def get_size(self):
        """
        Returns the current number of stored transitions (up to max_size meaning maximum capacitiy of RB).
        """
        return self.size

    def sample(self, batch_size=1):
        """
        Samples a batch of transitions from the replay buffer and returns GPU tensors
        (states, actions, rewards, next_states, dones). No CPU copies are performed (for better GPU utilization).
        """
        if batch_size > self.size:
            batch_size = self.size
        indices = torch.randint(0, self.size, (batch_size,), device=self.device)
        batch_states = self.states[indices]
        batch_actions = self.actions[indices]
        batch_rewards = self.rewards[indices]
        batch_next_states = self.next_states[indices]
        batch_dones = self.dones[indices]
        return batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones
