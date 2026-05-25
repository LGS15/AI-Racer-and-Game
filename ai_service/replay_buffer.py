import random
from collections import deque

import numpy as np

class ReplayBuffer:
    def __init__(self,capacity):
        self.buffer = deque(maxlen=capacity) # rolling buffer basically

    def push(self,state,action_index,reward,next_state,done):
        self.buffer.append((state,action_index,reward,next_state,done))
        
    def sample(self,batch_size):
        batch = random.sample(self.buffer,batch_size)
        state,actions,rewards,next_states,dones = zip(*batch)
        return(
            np.array(state, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32)
        )
    
    def __len__(self):
        return len(self.buffer)
