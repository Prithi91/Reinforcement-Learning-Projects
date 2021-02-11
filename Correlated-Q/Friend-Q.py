from Soccer import Soccer
import random
from cvxopt.modeling import op, variable
from cvxopt.solvers import options
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
options['show_progress'] = False

def friend_qlearning(max_iters, alpha, gamma, epsilon):
    #random.seed(5)
    stats = list()
    state_record = str(3) + str(2) + str(1)
    stateaction_record = (state_record, 2, 4)
    Q = defaultdict(lambda: 0)
    V1 = defaultdict(lambda: 1)
    prev_non0=0
    iter=0

    #start game
    env = Soccer(seed=0)
    curr_state = env.state
    done = False

    while(iter < max_iters):
        prev_q_stat = Q[stateaction_record]
        l=0
        #while(l <100):
        if(done):
            #done = False
            env = Soccer(seed=0)
            curr_state = env.state
            #break
        rnd = random.random()
        if rnd < epsilon:
            a = random.choices([0, 1, 2, 3, 4], k=2)
        else:
            a = []
            q_s = {}
            for key in Q:
                if (key[0] == curr_state):
                    q_s[key] = Q[key]
            if (len(q_s) == 0):
                a = random.choices([0, 1, 2, 3, 4], k=2)
            else:
                qs_a = [0, 0, 0, 0, 0]
                for key in q_s:
                    i = key[1]
                    qs_a[i] += q_s[key]
                a.append(np.argmax(qs_a))
                a.append(np.argmin(qs_a))
                #a.append(random.choice([0, 1, 2, 3, 4]))
        #a = random.choices([0, 1, 2, 3, 4], k=2)
        newstate, r0, r1, done = env.play(curr_state, a[0], a[1])
        maxq= Q[(newstate, 0,0)]
        for a0 in range(5):
            for a1 in range(5):
                maxq = max(maxq, Q[(newstate, a0, a1)])

        V1[newstate] = maxq
        #prev_q = Q[(curr_state, a[0],a[1])]

        Q[(curr_state, a[0],a[1])] = ((1-alpha)* Q[(curr_state, a[0],a[1])]) + (alpha * (r0 + gamma * V1[newstate]))
        curr_state = newstate
        epsilon = max(0.1, epsilon * 0.9999954)

        #new_q = Q[(curr_state, a[0],a[1])]
        new_q_stat = Q[stateaction_record]
        diff = abs(new_q_stat - prev_q_stat)
        # if(abs(new_q_stat-prev_q_stat)>0):
        #     print("new:{}".format(new_q_stat))
        #     print("prev:{}".format(prev_q_stat))
        # print(new_q_stat)
        # print(prev_q_stat)
        if(diff>0):
            prev_non0 = diff
            stats.append((iter, diff))
        else:
            stats.append((iter, prev_non0))
        if iter % 10000 == 0:
            print(iter)
            #print(abs(new_q_stat - prev_q_stat))
        # if ((curr_state, a[0], a[1]) == stateaction_record):
        #     stats.append((iter, abs(new_q - prev_q)))

        alpha = max(0.001, alpha * 0.9999954)
        iter+=1
    print(Q)
    return stats

if __name__=="__main__":
    stats = friend_qlearning(1000000, 0.1, 0.9,1.0)
    for i in range(1500):
        stats = stats[1:]
    stats_disp = range(1500, 1000000)
    #printstats = [stat[1] for stat in stats]
    #print(printstats[0:1000])
    plt.plot(stats_disp, [stat[1] for stat in stats])
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.ylim([0,0.5])
    plt.xticks(np.arange(1,1000000,100000))
    plt.title("Friend-Q")
    plt.xlabel("Simulation Itertion")
    plt.ylabel("Q-Value Difference")
    plt.savefig("FriendQ.png")
    #game = Soccer(seed=10)
