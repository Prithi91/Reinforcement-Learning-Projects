import numpy as np
import random
import matplotlib.pyplot as plt
import math

def generate_seq(n_steps):
    step=3
    sequence =np.zeros(n_steps).astype(int)
    sequence[step]=1
    while(step!= 0 and step !=6):
        seq_i=np.zeros(n_steps).astype(int)
        choice = random.choices([-1, 1], weights=[0.5, 0.5])
        step = step + choice[0]
        seq_i[step]=1
        sequence = np.vstack((sequence, seq_i))
    return sequence

def exp1(seq,w,lmbda,alpha):
    n_steps = seq.shape[0]-1
    lmbda_seq = np.ones(1)
    del_w = np.zeros(5)
    r_final = seq[-1][-1]
    nt_states = seq[:-1,1:-1]
    for step in range(n_steps):
        steps_visited = nt_states[0:step+1]
        if(step== n_steps-1):
            del_v = r_final - np.dot(w,nt_states[step])
        else:
            del_v = np.dot(w,nt_states[step+1]) - np.dot(w,nt_states[step])
        del_w += alpha * del_v * np.sum(steps_visited * lmbda_seq[:,None], axis=0)
        lmbda_seq = np.concatenate((lmbda_seq * lmbda, np.ones(1)))
    return del_w

if __name__=="__main__":
    random.seed(108)
    true_vals= np.asarray([1/6,1/3,1/2,2/3,5/6])
    training=100 * [10 *[None]]
    lmbda_vals=[0,0.1,0.3,0.5,0.7,0.9,1]
    lmbda_vals2=[0,1,0.8,0.3]
    lmbda_vals3 = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
    alpha_vals=[0.0,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6]
    alpha_vals3=[0.0,0.01,0.001,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6]
    RMSE_lmbda=np.zeros(7)
    RMSE_alpha=np.zeros(12)
    ones = []
    zeroes = []
    for set in range(100):
        count_ones=0
        count_zeroes=0
        for i in range(0,10):
            seq = generate_seq(7)
            training[set][i]= seq
            if(seq[-1][-1]==0): count_zeroes+=1
            if(seq[-1][-1]==1): count_ones+=1
        ones.append(count_ones)
        zeroes.append(count_zeroes)
    fig, ax= plt.subplots()
    Sets=['1-25','26-50','51-75','76-100']
    x=np.arange(4)
    no_seqs_1= [np.mean(ones[0:25]), np.mean(ones[25:50]), np.mean(ones[50:75]), np.mean(ones[75:100])]
    no_seqs_0= [np.mean(zeroes[0:25]), np.mean(zeroes[25:50]), np.mean(zeroes[50:75]), np.mean(zeroes[75:100])]
    ax.bar(x- 0.35/2, no_seqs_1,width=0.35,color='g',label='G')
    ax.bar(x+ 0.35/2, no_seqs_0,width=0.35, color='b', label='A')
    ax.set_ylim(0,7)
    plt.xlabel("Training Sets")
    ax.set_xticks(x)
    ax.set_xticklabels(Sets)
    ax.legend()
    plt.ylabel("Average Number of sequences")
    plt.savefig("Distribution")
    plt.clf()
    training_np = np.asarray(training)
    #####################Experiment 1##########################
    iter=0
    for lmbdaval in lmbda_vals:
        w = 0.5 * np.ones(5)
        old_w=np.copy(w)
        del_w = np.zeros(5)
        for repeats in range(1000):
            rmse_set = 0
            for set in range(100):
                for seq in range(10):
                    del_w += exp1(training_np[set][seq], w, lmbdaval,0.001)
                rmse_set += np.sqrt(np.mean((w - true_vals) ** 2))
                w += del_w
                del_w= np.zeros(5)
            dif = np.absolute(old_w-w)
            if(all(dif<1e-6)):
                break
            old_w=np.copy(w)
       # RMSE_lmbda[iter] = np.sqrt(((w - true_vals)**2).mean())
        RMSE_lmbda[iter] = rmse_set/100
        iter+=1
    plt.plot(lmbda_vals, RMSE_lmbda, "-o")
    plt.xticks([0.0,0.1,0.3,0.5,0.7,0.9,1])
    plt.xlabel('λ')
    plt.ylabel('RMSE')
    plt.savefig('graph3.png')
    plt.clf()
    #####################Experiment 2##########################
    for lmbdaval in lmbda_vals2:
        iter = 0
        plt.xticks([0.0,0.1,0.2,0.3,0.4,0.5,0.6])
        plt.xlabel('α')
        plt.ylabel('RMSE')
        for alphaval in alpha_vals:
            rmse_set = 0
            for set in range(100):
                w = 0.5 * np.ones(5)
                for seq in range(10):
                    w+= exp1(training_np[set][seq], w, lmbdaval, alphaval)
                    rmse_set += np.sqrt(((np.round(w,20) - true_vals) ** 2).mean())
            RMSE_alpha[iter] = rmse_set/1000
            iter+=1
        plt.plot(alpha_vals, RMSE_alpha,"-o", label="λ={val}".format(val=lmbdaval))
    plt.legend()
    plt.savefig('graph4.png')
    plt.clf()
    # #####################Experiment 3##########################
    RMSE_bestalpha=np.zeros(len(lmbda_vals3))
    lmbda_iter=0
    for lmbdaval in lmbda_vals3:
        RMSE_alpha = np.zeros(len(alpha_vals3))
        iter = 0
        for alphaval in alpha_vals3:
            rmse_set = 0
            for set in range(100):
                w = 0.5 * np.ones(5)
                for seq in range(10):
                    w+= exp1(training_np[set][seq], w, lmbdaval, alphaval)
                rmse_set += np.sqrt(np.mean((w-true_vals)**2))
            RMSE_alpha[iter] = rmse_set/100
            iter+=1
        best_error = np.amin(RMSE_alpha)
        RMSE_bestalpha[lmbda_iter] = best_error
        lmbda_iter+=1
    plt.plot(lmbda_vals3, RMSE_bestalpha,"-o")
    plt.xticks([0.0,0.2,0.4,0.6,0.8,1])
    plt.xlabel('λ')
    plt.ylabel('Error using Best α')
    plt.savefig('graph5.png')
    plt.clf()

