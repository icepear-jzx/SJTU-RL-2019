import random


class Grid:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.action = [(1, 0), (0, 1), (-1, 0), (0, -1)]
        if self.x == 3 and 0 < self.y < 11:
            self.reward = -100.0
        else:
            self.reward = -1.0
        self.gamma = 1.0
        self.epsilon = 0.000001
        self.alpha = 0.3
    
    def nextState(self, gridworld, action):
        next_x, next_y = self.x + action[0], self.y + action[1]
        if next_x > 3 or next_x < 0 or next_y > 11 or next_y < 0:
            next_x, next_y = self.x, self.y
        return gridworld[next_x][next_y]


def e_greedy(gridworld, Q, state):
    if state == gridworld[3][11]:
        return (0,0)
    allQ = [Q[state][action] for action in state.action]
    maxQ = max(allQ)
    maxA = state.action[allQ.index(maxQ)]
    pi_maxA = 1 - state.epsilon
    pi_each = state.epsilon/4
    randnum = random.random()
    if randnum < pi_maxA:
        return maxA
    else:
        return state.action[int((randnum - pi_maxA) // pi_each)]


def greedy(gridworld, Q, state):
    if state == gridworld[3][11]:
        return (0,0)
    allQ = [Q[state][action] for action in state.action]
    maxQ = max(allQ)
    maxA = state.action[allQ.index(maxQ)]
    return maxA


def sarsa():
    # init gridworld
    gridworld = [[Grid(i, j) for j in range(12)] for i in range(4)]
    # init Q
    stateList = [gridworld[i][j] for j in range(12) for i in range(4)]
    actionList = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    Q = {state:{action:0.0 for action in actionList} for state in stateList}

    for k1 in range(10000):     # for each episode
        state = gridworld[3][0]
        # get action using e_greedy
        action = e_greedy(gridworld, Q, state)
        while state != gridworld[3][11]:    # for each step
            # take action, get next_state, consider cliff region
            if state.x == 3 and 0 < state.y < 11:
                next_state = gridworld[3][0]
            else:
                next_state = state.nextState(gridworld, action)
            # get next_action using e_greedy
            next_action = e_greedy(gridworld, Q, next_state)
            if next_action != (0,0):
                # update Q using Q's iterative formula
                Q[state][action] += state.alpha * (state.reward + state.gamma * Q[next_state][next_action] - Q[state][action])
            # S <= S', A <= A'
            state = next_state
            action = next_action
    
    # show policy, if failed, print "failed!" five times
    print('Sarsa:')
    state = gridworld[3][0]
    episode = [state]
    while state != gridworld[3][11]:
        allQ = [Q[state][action] for action in state.action]
        maxQ = max(allQ)
        maxA = state.action[allQ.index(maxQ)]
        state = state.nextState(gridworld, maxA)
        if state in episode:
            print('failed!\t'*5)
            break
        episode.append(state)
    if state == gridworld[3][11]:
        for state in episode[:-1]:
            print((state.x,state.y),end='->')
        state = episode[-1]
        print((state.x,state.y))

        


def q_learning():
    # init gridworld
    gridworld = [[Grid(i, j) for j in range(12)] for i in range(4)]
    # init Q
    stateList = [gridworld[i][j] for j in range(12) for i in range(4)]
    actionList = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    Q = {state:{action:0.0 for action in actionList} for state in stateList}

    for k1 in range(10000):     # for each episode
        state = gridworld[3][0]
        while state != gridworld[3][11]:    # for each step
            # choose action using e_greedy
            action = e_greedy(gridworld, Q, state)
            # take action considering cliff region
            if state.x == 3 and 0 < state.y < 11:
                next_state = gridworld[3][0]
            else:
                next_state = state.nextState(gridworld, action)
            # choose next_action using greedy
            next_action = greedy(gridworld, Q, next_state)
            if next_action != (0,0):
                # update Q using Q's iterative formula
                Q[state][action] += state.alpha * (state.reward + state.gamma * Q[next_state][next_action] - Q[state][action])
            state = next_state
    
    # show policy
    print('Q-learning:')
    state = gridworld[3][0]
    print((state.x,state.y),end='->')
    while state != gridworld[3][11]:
        allQ = [Q[state][action] for action in state.action]
        maxQ = max(allQ)
        maxA = state.action[allQ.index(maxQ)]
        state = state.nextState(gridworld, maxA)
        if state == gridworld[3][11]:
            print((state.x,state.y))
        else:
            print((state.x,state.y),end='->')



for k in range(10):
    # Sarsa
    sarsa()

    # Q-learning
    q_learning()