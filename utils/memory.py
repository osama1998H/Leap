import numpy as np


class Memory:
    def __init__(self, max_size, state_shape):
        self.max_size = max_size
        self.state_memory = np.zeros((self.max_size, *state_shape))
        self.new_state_memory = np.zeros((self.max_size, *state_shape))
        self.action_memory = np.zeros(self.max_size, dtype=np.int8)
        self.reward_memory = np.zeros(self.max_size)
        self.terminal_memory = np.zeros(self.max_size, dtype=np.bool)
        self.mem_counter = 0

    def store_transition(self, state, action, reward, next_state, done):
        index = self.mem_counter % self.max_size
        self.state_memory[index] = state
        self.new_state_memory[index] = next_state
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.terminal_memory[index] = done
        self.mem_counter += 1

    def __len__(self):
        return min(self.mem_counter, self.max_size)

    def __getitem__(self, index):
        return (self.state_memory[index], self.action_memory[index], self.reward_memory[index],
                self.new_state_memory[index], self.terminal_memory[index])
