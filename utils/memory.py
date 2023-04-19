import numpy as np

class Memory:
    def __init__(self, max_size, state_shape):
        self.max_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((max_size, *state_shape))
        self.action_memory = np.zeros(max_size, dtype=np.int8)
        self.reward_memory = np.zeros(max_size)
        self.next_state_memory = np.zeros((max_size, *state_shape))
        self.terminal_memory = np.zeros(max_size, dtype=np.bool)

    def store_transition(self, state, action, reward, next_state, done):
        index = self.mem_cntr % self.max_size
        self.state_memory[index] = state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.next_state_memory[index] = next_state
        self.terminal_memory[index] = done
        self.mem_cntr += 1


