import gym
import random
#from Qnet import Agent
from NN import Agent
import numpy as np
from collections import deque
import sys
import matplotlib.pyplot as plt

MAX_TRAIN=2000
MAX_STEPS=1000
EPS=1
EPS_DECAY=0.99
EPS_END=0.01
BUFFER_SIZE = int(1e5)  # Replay memory size
BATCH_SIZE = 64         # Number of experiences to sample from memory
GAMMA = 0.99            # Discount factor
ALPHA = 1e-3              # Soft update parameter for updating fixed q network
LR = 1e-4               # Q Network learning rate
C = 5

if __name__=="__main__":
    env=gym.make('LunarLander-v2')
    env.seed(0)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    dqnagent = Agent(state_size,action_size,BUFFER_SIZE,BATCH_SIZE,0,GAMMA,ALPHA,C)
    rewards_all=[]
    scores_window = deque(maxlen=100)
    perepisoderewards=[]
    solved=-1
    flag=False

    for episode in range(0, MAX_TRAIN):
        state = env.reset()
        rewards=0
        for t in range(MAX_STEPS):
            action = dqnagent.act(state, EPS)
            next_state, reward, done, info = env.step(action)
            dqnagent.step(state, action, reward, next_state, done)
            state = next_state
            rewards += reward
            if done:
                if rewards >= 200 and flag == False:
                    flag = True
                    solved = episode
                break
            EPS = max(EPS * EPS_DECAY, EPS_END)
            if episode % C == 0:
                mean_score = np.mean(scores_window)
                print('\r Progress {}/{}, average score:{:.2f}'.format(episode, MAX_TRAIN, mean_score), end="")
            if rewards >= 200 and flag==False:
                flag = True
                solved=episode
                mean_score = np.mean(scores_window)
                print('\rEnvironment solved in {} episodes, average score: {:.2f}'.format(episode, mean_score), end="")
                sys.stdout.flush()
                dqnagent.checkpoint('model_converged_B55.pth')
                break
        perepisoderewards.append(rewards)
        scores_window.append(rewards)
        rewards_all.append(rewards)
    episodes = np.arange(len(perepisoderewards))
    plt.plot(episodes, perepisoderewards)
    if(solved!=-1):
        plt.plot(solved, perepisoderewards[solved], marker="o",label="Environment Solved")
    plt.title("Gamma=0.99, alpha=0.001, C=5")
    plt.xlabel("# of Epsiodes")
    plt.ylabel("Total Reward per Episode")
    if(solved!=-1):
        plt.legend()
    plt.savefig("Rewardsperstep_C5.png")
    plt.clf()


