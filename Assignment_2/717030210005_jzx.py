from random import randint


class Grid:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.value = 0.0
        self.next_value = 0.0
        self.cnt = 0
        self.next_cnt = 0
        self.sum = 0.0
        self.next_sum = 0.0
        if [self.x, self.y] == [0, 0] or [self.x, self.y] == [3, 3]:
            self.action = []
            self.reward = 0.0
        else:
            self.action = [(1, 0), (0, 1), (-1, 0), (0, -1)]
            self.reward = -1.0
        self.policy = self.action
        self.gamma = 0.5
    
    def updateValue(self):
        self.value = self.next_value
        self.sum = self.next_sum
        self.cnt = self.next_cnt

    def updatePolicy(self, gridworld):
        max_value = -float("inf")
        self.policy = []
        for i in range(len(self.action)):
            next_x = self.x + self.action[i][0]
            next_y = self.y + self.action[i][1]
            if next_x > 3 or next_x < 0 or next_y > 3 or next_y < 0:
                next_x, next_y = self.x, self.y
            if round(gridworld[next_x][next_y].value, 3) == round(max_value, 3):
                self.policy.append(self.action[i])
            elif round(gridworld[next_x][next_y].value, 3) > round(max_value, 3):
                self.policy = [self.action[i]]
                max_value = gridworld[next_x][next_y].value

    def nextGrid(self):
        i = randint(0,len(self.action)-1)
        next_x = self.x + self.action[i][0]
        next_y = self.y + self.action[i][1]
        if next_x > 3 or next_x < 0 or next_y > 3 or next_y < 0:
            next_x, next_y = self.x, self.y
        return next_x,next_y
    

def nextValue(gridworld,episode,idx):
    grid = episode[idx]
    temp_episode = episode[idx+1:]
    for i in range(0,len(temp_episode)):
        grid.next_sum += temp_episode[i].reward * temp_episode[i].gamma**i
        grid.next_cnt += 1
        grid.next_value = grid.next_sum / grid.next_cnt
    



def generateEpisode(gridworld, episode):
    while episode[-1].action:
        next_x,next_y = episode[-1].nextGrid()
        episode.append(gridworld[next_x][next_y])


def firstVisitMC():
    gridworld = [[Grid(i, j) for j in range(4)] for i in range(4)]
    for k in range(100000):
        episode = [gridworld[randint(0, 3)][randint(0, 3)]]
        generateEpisode(gridworld,episode)
        # episode = [gridworld[1][0],gridworld[1][1],gridworld[0][1],gridworld[0][0]]
        # print([(grid.x,grid.y) for grid in episode])
        for i in range(4):
            for j in range(4):
                if gridworld[i][j] in episode:
                    nextValue(gridworld,episode,episode.index(gridworld[i][j]))
        for i in range(4):
            for j in range(4):
                gridworld[i][j].updateValue()
        # print(gridworld[1][0].value,gridworld[2][2].value)
    for i in range(4):
        for j in range(4):
            gridworld[i][j].updatePolicy(gridworld)
    # show gridworld
    for i in range(4):
        print([gridworld[i][j].value for j in range(4)])
    # show policy
    for i in range(4):
        print([gridworld[i][j].policy for j in range(4)])

firstVisitMC()


def everyVisitMC():
    pass


def TD0():
    pass
