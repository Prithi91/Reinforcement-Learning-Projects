#from NN import Agent
from pg_network import Agent
import torch
import gym
from gym import wrappers
import numpy as np
import matplotlib.pyplot as plt

if __name__=="__main__":
    #agent = Agent(8,4,int(1e5),64,0,0.99,1e-3,4)
    env = gym.make('LunarLander-v2')
    env.seed(0)
    #agent.learn_network.load_state_dict(torch.load('model_converged_g0.999.pth'))
    agent = Agent(env.observation_space.shape[0], env.action_space.n,seed=0, lr=1e-3, gamma=0.99)
    agent.policy.load_state_dict(torch.load('model_converged(PG).pth'))

    #env = gym.wrappers.Monitor(gym.make('LunarLander-v2'), directory="videos/lander2/",force=True)

    #env = wrappers.monitor(env, '/videos', )
    perepisoderewards=[]
    for i in range(1000):
        rewards=0
        state = env.reset()
        while True:
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)
            state = next_state
            rewards += reward
            if done:
                break
        perepisoderewards.append(rewards)
        print('episode: {} scored {}'.format(i, rewards))
        if(i >=100):
            rollingaverage = np.mean(perepisoderewards[-100:])
            if(rollingaverage>=200):
                break
    episodes = np.arange(len(perepisoderewards))
    plt.plot(episodes, perepisoderewards,label="Average over the last 100 episodes: {}".format(perepisoderewards[-100]))
    #plt.title("Gamma=0.999, alpha=0.001, C=5")
    plt.xlabel("# of Epsiodes")
    plt.ylabel("Total Reward per Episode")
    plt.legend(loc="lower right")
    plt.savefig("Rewardsperstep_test_PG_256.png")
    plt.clf()

