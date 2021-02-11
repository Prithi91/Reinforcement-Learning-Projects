from Soccer import Soccer
import random
from cvxopt.modeling import op, variable
from cvxopt.solvers import options
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
options['show_progress'] = False

def foe_qlearning(max_iters, alpha, gamma, epsilon):
    stats = list()
    stats_state= list()
    state_record = str(3) + str(2) + str(1)
    stateaction_record = (state_record, 2, 4)
    Q = defaultdict(lambda: 1)
    V1 = defaultdict(lambda: 1)
    prev_non0=0
    iter=0

    #start game
    env = Soccer(seed=0)
    curr_state = env.state
    done = False

    while(iter < max_iters):
        prev_q_stat = Q[stateaction_record]
        l = 0
        if(done):
            env = Soccer(seed=0)
            curr_state = env.state
        # rnd = random.random()
        # if rnd < epsilon:
        #     a = random.choices([0, 1, 2, 3, 4], k=2)
        # else:
        #     a = []
        #     q_s={}
        #     for key in Q:
        #         if(key[0] == curr_state):
        #             q_s[key] = Q[key]
        #     if(len(q_s)==0): a = random.choices([0, 1, 2, 3, 4], k=2)
        #     else:
        #         qs_a = [0, 0, 0, 0, 0]
        #         for key in q_s:
        #             i = key[1]
        #             qs_a[i] += q_s[key]
        #         a.append(np.argmax(qs_a))
        #         a.append(random.choice([0, 1, 2, 3, 4]))
                #a.append(np.argmin(qs_a))
        a = random.choices([0, 1, 2, 3, 4], k=2)
        newstate, r0, r1, done = env.play(curr_state, a[0], a[1])

        #action_probs and constraints
        probs = list()
        constraints = list()
        for i in range(5):
            probs.append(variable())
        for i in range(5):
            constraints.append((probs[i] >= 0))

        probs_sum = sum(probs)
        constraints.append((probs_sum==1))

        f = variable()

        for a1 in range(5):
            q =0
            for a0 in range(5):
                q += Q[(newstate, a0, a1)] * probs[a0]
            constraints.append((q >= f))
        #print(constraints)

        lp = op(-f, constraints)
        lp.solve()

        V1[newstate] = f.value[0]
        #prev_Q = Q[(curr_state, a[0], a[1])]

        Q[(curr_state, a[0], a[1])] = ((1- alpha) * Q[(curr_state, a[0], a[1])]) + (alpha * (r0 + gamma * V1[newstate]))
        epsilon = max(0.01, epsilon * 0.99995)
        #l+=1
        #epsilon = max(0.1, epsilon * 0.9999954)
        #new_Q = Q[(curr_state, a[0], a[1])]
        new_q_stat = Q[stateaction_record]
        diff = abs(new_q_stat - prev_q_stat)
        probs_hist=[]
        for i in range(5):
            probs_hist.append(probs[i].value[0])
        # if (diff > 0):
        #     prev_non0 = diff
        #     stats.append((iter, diff,probs_hist))
        # else:
        #     stats.append((iter, prev_non0, probs_hist))
        stats.append((iter, diff, probs_hist))
        if (curr_state == state_record):
            stats_state.append((iter, abs(new_q_stat - prev_q_stat), probs_hist))
        curr_state = newstate
        if iter % 10000 == 0:
            print(iter)

        alpha = max(0.001, alpha * 0.999995)
        iter+=1
    print(Q)
    return stats, stats_state

if __name__=="__main__":
    stats, stats_state = foe_qlearning(1000000,1.0,0.9,0.75)
    #print(stats[-1][-1])
    #print(stats)
    file = open('FoeQ.csv', 'w')
    for iter, diff, probs in stats_state:
        file.write('{},{},{}\n'.format(iter, diff, probs))
    plt.plot(range(len(stats)), [stat[1] for stat in stats])
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.ylim([0,0.5])
    plt.xticks(np.arange(1,1000000,100000))
    plt.title("Foe-Q")
    plt.xlabel("Simulation Itertion")
    plt.ylabel("Q-Value Difference")
    plt.savefig("FoeQ_test.png")