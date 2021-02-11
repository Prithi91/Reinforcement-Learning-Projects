import random

class Soccer():

    def __init__(self, seed):
        #random.seed(seed)
        self.actions_map = {0: -4, 1: -1, 2: 4, 3: 1, 4:0}
        #a,b = random.sample([2,3,6,7],2)
        # ballpos = -1
        # a = random.sample([1,2,3,4,5,6,7,8],1)
        # if(a[0] in [1,5] or a[0] in [4,8]):
        #     b = random.sample([2,3,6,7],1)
        #     ballpos = 1
        # else:
        #     b = random.sample([1,2,3,4,5,6,7,8],1)
        #     if(b[0] in [1,5] or b[0] in [4,8]):
        #         ballpos = 0
        self.pos1 = 3
        self.pos2 = 2
        #if(ballpos==-1):
        ballpos = random.choice([0,1])
        self.ball = ballpos
        self.state = str(self.pos1) + str(self.pos2) + str(self.ball)

    def play(self, state, action1, action2):
        done = False
        reward0=0
        reward1=0
        first = random.choice([0,1])
        if(first == 0):
            self.pos1 = int(state[0]) + self.actions_map[action1]
            if(self.pos1 < 1 or self.pos1 > 8): self.pos1 = int(state[0])
            elif(self.pos1 == int(state[1])):
                if(int(state[2]) == 0): self.ball = 1
                self.pos1 = int(state[0])
            self.pos2 = int(state[1]) + self.actions_map[action2]
            if(self.pos2 < 1 or self.pos2 > 8): self.pos2 = int(state[1])
            elif(self.pos2 == self.pos1):
                if(self.ball == 1): self.ball = 0
                self.pos2 = int(state[1])
        else:
            self.pos2 = int(state[1]) + self.actions_map[action2]
            if (self.pos2 < 1 or self.pos2 > 8):
                self.pos2 = int(state[1])
            elif (self.pos2 == int(state[0])):
                if (int(state[2]) == 1): self.ball = 0
                self.pos2 = int(state[1])
            self.pos1 = int(state[0]) + self.actions_map[action1]
            if (self.pos1 < 1 or self.pos1 > 8):
                self.pos1 = int(state[0])
            elif (self.pos1 == self.pos2):
                if (self.ball == 0): self.ball = 1
                self.pos1 = int(state[0])
        self.state = str(self.pos1) + str(self.pos2) + str(self.ball)
        if((self.pos1 in [1,5] and self.ball== 0)or(self.pos2 in [1,5] and self.ball==1)):
            reward0 = 100
            reward1 = -100
            done = True
        elif((self.pos1 in [4,8] and self.ball== 0)or(self.pos2 in [4,8] and self.ball==1)):
            reward0 = -100
            reward1 = 100
            done = True
        return (self.state, reward0, reward1, done)

    def reset(self):
        a, b = random.sample([2, 3, 6, 7], 2)
        self.pos1 = a
        self.pos2 = b
        ballpos = random.choice([0, 1])
        self.ball = ballpos
        self.state = str(self.pos1) + str(self.pos2) + str(self.ball)
        return self.state










