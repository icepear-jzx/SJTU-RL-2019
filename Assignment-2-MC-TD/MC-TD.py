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
            if round(gridworld[next_x][next_y].value, 1) == round(max_value, 1):
                self.policy.append(self.action[i])
            elif round(gridworld[next_x][next_y].value, 1) > round(max_value, 1):
                self.policy = [self.action[i]]
                max_value = gridworld[next_x][next_y].value

    def nextGrid(self):
        i = randint(0, len(self.action)-1)
        next_x = self.x + self.action[i][0]
        next_y = self.y + self.action[i][1]
        if next_x > 3 or next_x < 0 or next_y > 3 or next_y < 0:
            next_x, next_y = self.x, self.y
        return next_x, next_y


def nextValueMC(gridworld, episode, idx):
    grid = episode[idx]
    temp_episode = episode[idx+1:]
    for i in range(0, len(temp_episode)):
        grid.next_sum += temp_episode[i].reward * temp_episode[i].gamma**i
    grid.next_cnt += 1
    grid.next_value = grid.next_sum / grid.next_cnt


def generateEpisode(gridworld, episode):
    while episode[-1].action:
        next_x, next_y = episode[-1].nextGrid()
        episode.append(gridworld[next_x][next_y])


def firstVisitMC():
    # init gridworld
    gridworld = [[Grid(i, j) for j in range(4)] for i in range(4)]

    for k in range(100000):
        # generate episode
        episode = [gridworld[randint(0, 3)][randint(0, 3)]]
        generateEpisode(gridworld, episode)

        # next value
        for i in range(4):
            for j in range(4):
                if gridworld[i][j] in episode:
                    nextValueMC(gridworld, episode,
                                episode.index(gridworld[i][j]))

        # update value
        for i in range(4):
            for j in range(4):
                gridworld[i][j].updateValue()

    # update policy
    for i in range(4):
        for j in range(4):
            gridworld[i][j].updatePolicy(gridworld)

    # show gridworld
    for i in range(4):
        print([round(gridworld[i][j].value, 1) for j in range(4)])

    # show policy
    for i in range(4):
        print([gridworld[i][j].policy for j in range(4)])


def everyVisitMC():
    # init gridworld
    gridworld = [[Grid(i, j) for j in range(4)] for i in range(4)]

    for k in range(100000):
        # generate episode
        episode = [gridworld[randint(0, 3)][randint(0, 3)]]
        generateEpisode(gridworld, episode)

        # next value
        for idx in range(len(episode)):
            nextValueMC(gridworld, episode, idx)

        # update value
        for i in range(4):
            for j in range(4):
                gridworld[i][j].updateValue()

    # update policy
    for i in range(4):
        for j in range(4):
            gridworld[i][j].updatePolicy(gridworld)

    # show gridworld
    for i in range(4):
        print([round(gridworld[i][j].value, 1) for j in range(4)])

    # show policy
    for i in range(4):
        print([gridworld[i][j].policy for j in range(4)])


def nextValueTD(gridworld, episode, idx):
    grid = episode[idx]
    next_grid = episode[idx+1]
    grid.next_value += 0.0001 * \
        (next_grid.reward + next_grid.gamma * next_grid.value - grid.value)


def TD0():
    # init gridworld
    gridworld = [[Grid(i, j) for j in range(4)] for i in range(4)]

    for k in range(100000):
        # generate episode
        episode = [gridworld[randint(0, 3)][randint(0, 3)]]
        generateEpisode(gridworld, episode)

        # next value
        for idx in range(len(episode)-1):
            nextValueTD(gridworld, episode, idx)

        # update value
        for i in range(4):
            for j in range(4):
                gridworld[i][j].updateValue()

    # update policy
    for i in range(4):
        for j in range(4):
            gridworld[i][j].updatePolicy(gridworld)

    # show gridworld
    for i in range(4):
        print([round(gridworld[i][j].value, 1) for j in range(4)])

    # show policy
    for i in range(4):
        print([gridworld[i][j].policy for j in range(4)])


def main():
    print("first-visit MC:")
    firstVisitMC()
    print()
    print("every-visit MC:")
    everyVisitMC()
    print()
    print("TD(0):")
    TD0()
    print()


main()
