import numpy as np

# 
# replay buffer
# ref: Deep Reinforcement Learning Hands-On / Maxim Lapan
# 
class ReplayBuffer():
    
    def __init__(self, capacity) -> None:
        self.buffer = [None]*capacity
        self.mem_size = capacity
        self.mem_cntr = 0

    def append(self, experience):
        idx = self.mem_cntr % self.mem_size
        self.buffer[idx] = experience
        self.mem_cntr += 1
    
    def sample(self, batch_size):

        if self.mem_cntr == 0:
            return None
        
        indices = np.random.choice(sum(1 for x in self.buffer if x is not None), batch_size, replace=False)
        states, actions, rewards, dones, next_states = zip(*[self.buffer[idx] for idx in indices])
        
        return np.array(states),\
               np.array(actions),\
               np.array(rewards),\
               np.array(dones),\
               np.array(next_states)
