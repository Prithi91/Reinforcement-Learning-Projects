import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class nnmodel(nn.Module):
    def __init__(self,lr,nstates,nactions,seed):
        super(nnmodel, self).__init__()
        self.lr = lr
        self.nstates = nstates
        self.nactions = nactions
        self.seed = T.manual_seed(seed)
        self.fc1 = nn.Linear(nstates, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, nactions)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self, state):
        state = T.tensor(state)
        h = nn.functional.relu(self.fc1(state))
        h2 = nn.functional.relu(self.fc2(h))
        a = self.fc3(h2)
        return a

class Agent(object):
    def __init__(self, state_size, action_size,seed,gamma,lr):
        self.gamma = gamma
        self.seed = seed
        self.rewards_list=[]
        self.actions_list=[]
        self.policy = nnmodel(lr, state_size, action_size, seed)

    def act(self, state):
        probs = F.softmax(self.policy.forward(state))
        action_probs = T.distributions.Categorical(probs)
        action = action_probs.sample()
        logs = action_probs.log_prob(action)
        self.actions_list.append(logs)

        return action.item()

    def store_rewards(self, reward):
        self.rewards_list.append(reward)

    def learn(self):
        self.policy.optimizer.zero_grad()
        G = np.zeros_like(self.rewards_list, dtype=np.float64)
        for r in range(len(self.rewards_list)):
            G_t = 0
            gamma =1
            for t in range(r, len(self.rewards_list)):
                G_t += gamma * self.rewards_list[t]
                gamma*= self.gamma
            G[r] = G_t
        mean_rewards = np.mean(G)
        std = np.std(G) if np.std(G) > 0 else 1
        G = (G-mean_rewards)/std

        G= T.tensor(G, dtype=T.float)
        loss = 0
        for g, actionprob in zip(G, self.actions_list):
            loss += -g * actionprob

        loss.backward()
        self.policy.optimizer.step()

        self.rewards_list=[]
        self.actions_list=[]

    def checkpoint(self, filename):
        T.save(self.policy.state_dict(), filename)