import numpy as np

class ExperianceReplay:
    def __init__(self, capacity, state_dim, action_dim):
        self.capacity = capacity
        self.state_dim = state_dim
        self.action_dim = action_dim
        # memory initiation
        self.state = np.zeros((capacity, state_dim))
        self.action = np.zeros((capacity, action_dim))
        self.reward = np.zeros((capacity, 1))
        self.next_state = np.zeros((capacity, state_dim))
        self.done = np.zeros((capacity, 1))
        self.index = 0
    
    def add(self, capacity, state, action, reward, next_state, done):
        i = self.index
        self.state[i:i+state.__len__()] = state
        self.action[i:i+action.__len__()] = action[:,None]
        self.reward[i:i+reward.size] = reward[:,None]
        self.next_state[i:i+next_state.__len__()] = next_state
        self.done[i:i+done.size] = done[:,None]
        self.index += done.size
        if self.index + state.__len__() >= capacity: 
            self.index = 0
    
    def sample(self, batch_size):
        i = np.random.randint(0, self.index+1, size= batch_size)
        return(
            self.state[i],
            self.action[i],
            self.reward[i],
            self.next_state[i],
            self.done[i]
        )