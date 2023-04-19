import numpy as np

class Memory:
    def __init__(self, max_size):
        self.mem_counter = 0
        self.mem_size = max_size
        self.state_memory = np.zeros((max_size, 4))
        self.new_state_memory = np.zeros((max_size, 4))
        self.action_memory = np.zeros(max_size, dtype=np.int8)
        self.reward_memory = np.zeros(max_size)
        self.terminal_memory = np.zeros(max_size, dtype=np.bool)

    def store_transition(self, state, action, reward, next_state, done):
        index = self.mem_counter % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = next_state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done
        self.mem_counter += 1
        self.mem_counter = min(self.mem_counter, self.mem_size)




