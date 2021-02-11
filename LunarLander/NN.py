import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from collections import deque, namedtuple
import numpy as np
import torch.optim as optim

class NN(nn.Module):

    def __init__(self, n_states,n_actions,seed):
        super(NN, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(n_states, 32)
        self.fc2 = nn.Linear(32,64)
        self.fc3 = nn.Linear(64, n_actions)

    def forward(self, state):
        h = nn.functional.relu(self.fc1(state))
        h2 = nn.functional.relu(self.fc2(h))
        a = self.fc3(h2)
        return a

class Replay:
    def __init__(self, buffer_size, batch_size, seed):
        self.batch_size = batch_size
        self.seed = random.seed(seed)
        self.memory = deque(maxlen=buffer_size)
        self.step = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

    def add(self, state, action, reward, next_state, done):
        step = self.step(state, action, reward, next_state, done)
        self.memory.append(step)

    def sample(self):
        samples = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([step.state for step in samples if samples is not None])).float()
        actions = torch.from_numpy(np.vstack([step.action for step in samples if samples is not None])).long()
        rewards = torch.from_numpy(np.vstack([step.reward for step in samples if samples is not None])).float()
        next_states = torch.from_numpy(np.vstack([step.next_state for step in samples if samples is not None])).float()
        dones = torch.from_numpy(np.vstack([step.done for step in samples if samples is not None]).astype(np.uint8)).float()

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory)

class Agent:
    def __init__(self, state_size, action_size, buffersize, batchsize,seed,gamma,alpha,c=4):
        self.state_size = state_size
        self.action_size = action_size
        self.batch_size=batchsize
        self.buffer_size=buffersize
        self.seed = random.seed(seed)
        self.gamma = gamma
        self.alpha = alpha
        self.learn_network = NN(state_size, action_size, seed)
        self.target_network = NN(state_size, action_size, seed)
        self.optimizer = optim.Adam(self.learn_network.parameters())
        self.memory = Replay(buffersize, batchsize, seed)
        self.timestep = 0
        self.c=c

    def step(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)
        self.timestep += 1
        if self.timestep % self.c == 0:
            if len(self.memory) > self.batch_size:
                sampled_experiences = self.memory.sample()
                self.learn(sampled_experiences)

    def learn(self, experiences):
        states, actions, rewards, next_states, dones = experiences
        action_values = self.target_network(next_states).detach()
        max_action_values = action_values.max(1)[0].unsqueeze(1)
        Q_target = rewards + (self.gamma * max_action_values * (1 - dones))
        Q_expected = self.learn_network(states).gather(1, actions)
        loss = F.mse_loss(Q_expected, Q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        for lp, tp in zip(self.learn_network.parameters(), self.target_network.parameters()):
            tp.data.copy_(self.alpha * lp.data + (1.0 - self.alpha) * tp.data)

    def act(self, state, eps=0.0):
        rnd = random.random()
        if rnd < eps:
            return np.random.randint(self.action_size)
        else:
            state = torch.from_numpy(state).float().unsqueeze(0)
            self.learn_network.eval()
            with torch.no_grad():
                action_values = self.learn_network(state)
            self.learn_network.train()
            action = np.argmax(action_values.cpu().data.numpy())
            return action

    def checkpoint(self, filename):
        torch.save(self.learn_network.state_dict(), filename)