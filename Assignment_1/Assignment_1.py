class Grid:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.state = 0.0
        self.next_state = 0.0
        if [self.x, self.y] == [0,0] or [self.x, self.y] == [3,3]:
            self.action = []
        else:
            self.action = [(1,0), (0,1), (-1,0), (0,-1)]
        self.action_p = [0.25, 0.25, 0.25, 0.25]
        self.policy = self.action
        self.reward = -1
    
    def nextState(self, gridworld):
        self.next_state = 0
        for i in range(len(self.action)):
            next_x = self.x + self.action[i][0]
            next_y = self.y + self.action[i][1]
            if next_x > 3 or next_x < 0 or next_y > 3 or next_y < 0:
                next_x, next_y = self.x, self.y
            self.next_state += self.action_p[i] * (self.reward + gridworld[next_x][next_y].state)
    
    def updateState(self):
        self.state = self.next_state
    
    def updatePolicy(self, gridworld):
        max_state = -float("inf")
        self.policy = []
        for i in range(len(self.action)):
            next_x = self.x + self.action[i][0]
            next_y = self.y + self.action[i][1]
            if next_x > 3 or next_x < 0 or next_y > 3 or next_y < 0:
                next_x, next_y = self.x, self.y
            if round(gridworld[next_x][next_y].state, 3) == round(max_state, 3):
                self.policy.append(self.action[i])
            elif round(gridworld[next_x][next_y].state, 3) > round(max_state, 3):
                self.policy = [self.action[i]]
                max_state = gridworld[next_x][next_y].state


# init gridworld
gridworld = [[Grid(i,j) for j in range(4)] for i in range(4)]
stable = False
k = 1

while not stable:
    stable = True
    # calculate nextstate
    for i in range(4):
        for j in range(4):
            gridworld[i][j].nextState(gridworld)
    
    # judge stable
    for i in range(4):
        for j in range(4):
            if abs(gridworld[i][j].next_state - gridworld[i][j].state) > 0.01:
                stable = False

    # update state
    for i in range(4):
        for j in range(4):
            gridworld[i][j].updateState()
    
    # update policy
    for i in range(4):
        for j in range(4):
            gridworld[i][j].updatePolicy(gridworld)

    print("k =",k)
    k += 1

    # show gridworld
    for i in range(4):
        print([gridworld[i][j].state for j in range(4)])
    
    # show policy
    for i in range(4):
        print([gridworld[i][j].policy for j in range(4)])

    # enter to continue
    input()






