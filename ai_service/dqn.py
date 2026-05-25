import random

import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    #state vector (18) into qvalues for each action (7)
    def __init__(self,state_size,action_count, hidden=64):
        super().__init__()
        self.fc1 = nn.Linear(state_size,hidden)
        self.fc2 = nn.Linear(hidden,hidden)
        self.out = nn.Linear(hidden,action_count)

    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)
    
class DQNAgent: 
    def __init__(self, state_size, action_count, gamma=0.99, lr=2.5e-4):
        self.action_count = action_count
        self.gamma = gamma
        self.online = QNetwork(state_size, action_count)
        self.target = QNetwork(state_size, action_count)
        self.target.load_state_dict(self.online.state_dict())
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=lr)

    def act(self, state, epsilon):
        if random.random() < epsilon:
            return random.randrange(self.action_count)
        with torch.no_grad():
            q = self.online(torch.as_tensor(state, dtype=torch.float32))
            return int(q.argmax().item())
        
    def train_step(self, batch):
        states, actions, rewards, next_states, dones = (torch.as_tensor(b) for b in batch)

        q = self.online(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q = self.target(next_states).max(dim=1).values
            td_target = rewards + (1.0 - dones) * self.gamma * next_q

        loss = F.smooth_l1_loss(q, td_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return float(loss.item())
    
    def sync_target(self):
        self.target.load_state_dict(self.online.state_dict())

    def save(self, path, step=0):
        torch.save({
            "online": self.online.state_dict(),
            "target": self.target.state_dict(),
            "step": step,
        }, path)

    def load(self, path):
        ckpt = torch.load(path, map_location="cpu")
        self.online.load_state_dict(ckpt["online"])
        self.target.load_state_dict(ckpt["target"])
        return int(ckpt.get("step", 0))   # so epsilon resumes where it left off