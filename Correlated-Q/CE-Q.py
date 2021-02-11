from Soccer import Soccer
import random
import sys
from cvxopt import solvers, matrix
from cvxopt.modeling import op, variable
from cvxopt.solvers import options
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
options['show_progress'] = False
#solvers.options['show_progress'] = False

def ceq(max_iters, alpha, gamma, epsilon):
    joint_actions = {0:(0,0),1:(0,1),2:(0,2),3:(0,3),4:(0,4),5:(1,0),6:(1,1),7:(1,2),8:(1,3),9:(1,4),10:(2,0),11:(2,1),12:(2,2),13:(2,3),14:(2,4),
                     15:(3,0),16:(3,1),17:(3,2),18:(3,3),19:(3,4),20:(4,0),21:(4,1),22:(4,2),23:(4,3),24:(4,4)}
    stats = list()
    stats_state= list()
    state_record = str(3) + str(2) + str(1)
    stateaction_record = (state_record, 2, 4)
    Q1 = defaultdict(lambda: 1)
    Q2 = defaultdict(lambda: 1)
    V1 = defaultdict(lambda: 1)
    V2 = defaultdict(lambda: 1)
    probs_hist=[]
    prev_non0 = 0
    iter = 0

    # start game
    env = Soccer(seed=0)
    curr_state = env.state
    done = False

    while(iter < max_iters):
        if (done):
            env = Soccer(seed=0)
            curr_state = env.state
        # rnd = random.random()
        # if rnd < epsilon:
        #     a = random.choices([0, 1, 2, 3, 4], k=2)
        # else:
        #     a=[]
        #     if(len(stats)>0):
        #         probs = stats[-1][-1]
        #         for p in range(len(probs)):
        #             if(probs[p] < 0.01): probs[p] = 0
        #             if(probs[p] >0.99): probs[p] = 1
        #         index = range(25)
        #         print(probs)
        #         c = np.random.choice(index, 1, p=probs)
        #         a = joint_actions[c[0]]
        #     else:
        #         a = random.choices([0, 1, 2, 3, 4], k=2)
        a = random.choices([0, 1, 2, 3, 4], k=2)
        newstate, r0, r1, done = env.play(curr_state, a[0], a[1])
        probs_backup={}
        probs = {}
        constraints = []
        sum_prob=0
        for i in range(5):
            for j in range(5):
                probs[(i, j)] = variable()

        for i in range(5):
            for j in range(5):
                constraints.append((probs[(i, j)] >= 0))
        for i in range(5):
            for j in range(5):
                sum_prob += probs[(i,j)]
        constraints.append((sum_prob==1))

        for i in range(5):
            ui = 0
            for j in range(5):
                ui += probs[(i, j)] * Q1[(newstate,i, j)]
            for k in range(5):
                if i != k:
                    uk = 0
                    for l in range(5):
                        uk += probs[(i, l)] * Q1[(newstate, k, l)]
                    constraints.append((ui >= uk))

        for i in range(5):
            ui = 0
            for j in range(5):
                ui += probs[(j, i)] * Q2[(newstate, j, i)]

            for k in range(5):
                if i != k:
                    uk = 0
                    for l in range(5):
                        uk += probs[(l, i)] * Q2[(newstate, l, k)]
                    constraints.append((ui >= uk))

        expected_reward = 0
        f = variable()
        for i in range(5):
            for j in range(5):
                expected_reward += probs[(i, j)] * Q1[(newstate, i, j)]
                expected_reward += probs[(i, j)] * Q2[(newstate, i, j)]
        constraints.append((f==expected_reward))
        lp = op(-f, constraints)
        lp.solve(solver='glpk',options={'glpk':{'msg_lev':'GLP_MSG_OFF', 'tm_lim': 2000}})
        if lp.status == 'optimal':
            allp = [probs[prob].value[0] for prob in probs]
            probs_backup = probs
            v1 = 0
            for i in range(5):
                for j in range(5):
                    v1 += probs[(i, j)].value[0] * Q1[(newstate, i, j)]
            V1[newstate] = v1
            v2 = 0
            for i in range(5):
                for j in range(5):
                    v2 += probs[(i, j)].value[0] * Q2[(newstate, i, j)]
            V2[newstate] = v2
        else:
            probs= probs_backup
        prev_q_stat = Q1[stateaction_record]
        # if((curr_state,a[0],a[1]) == stateaction_record):
        #     print(newstate)
        #     print(V1[newstate])
        Q1[(curr_state, a[0], a[1])] = ((1 - alpha) * Q1[(curr_state, a[0], a[1])] )+ (alpha * (r0 + gamma * V1[newstate]))
        Q2[(curr_state, a[0], a[1])] = ((1 - alpha) * Q2[(curr_state, a[0], a[1])]) + (alpha * (-r0 + gamma * V2[newstate]))
        new_q_stat = Q1[stateaction_record]
        if(lp.status == "optimal"):
            probs_hist=[]
            for i in range(5):
                for j in range(5):
                    probs_hist.append(probs[(i, j)].value[0])
        stats.append((iter, abs(new_q_stat - prev_q_stat), probs_hist))
        if (curr_state == state_record):
            stats_state.append((iter, abs(new_q_stat - prev_q_stat), probs_hist))
        curr_state = newstate
        iter += 1
        alpha = max(0.001, alpha * 0.999995)
        epsilon = max(0.01, epsilon * 0.99995)
        if iter % 10000 == 0:
            print(iter)
    print(Q1)
    print(Q2)
    return stats, stats_state

if __name__=="__main__":
    stats, stats_state = ceq(1000000,0.2, 0.9, 1.0)
    #print(stats[-1][-1])
    file = open('CEQ.csv', 'w')
    for iter, diff, probs in stats_state:
        file.write('{},{},{}\n'.format(iter, diff, probs))
    plt.plot(range(len(stats)), [stat[1] for stat in stats])
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    plt.ylim([0, 0.5])
    plt.xticks(np.arange(1, 1000000, 100000))
    plt.title("Correlated-Q")
    plt.xlabel("Simulation Itertion")
    plt.ylabel("Q-Value Difference")
    plt.savefig("CEQ_test_lr1.png")


