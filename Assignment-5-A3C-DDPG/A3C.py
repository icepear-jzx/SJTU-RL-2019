import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import torch.optim as optim
import numpy as np
import gym
import math, os
import matplotlib.pyplot as plt

os.environ["OMP_NUM_THREADS"] = "1"

UPDATE_GLOBAL_ITER = 5
GAMMA = 0.9
MAX_EP = 1000
MAX_EP_STEP = 200
N_CPU = mp.cpu_count()
# N_CPU = 1

env = gym.make('Pendulum-v0')
N_S = env.observation_space.shape[0]
N_A = env.action_space.shape[0]


class SharedAdam(torch.optim.Adam):

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.9), eps=1e-8,
                 weight_decay=0):
        super(SharedAdam, self).__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        # State initialization
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p.data)
                state['exp_avg_sq'] = torch.zeros_like(p.data)

                # share in memory
                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()


class Net(nn.Module):

    def __init__(self, s_dim, a_dim):
        super(Net, self).__init__()
        self.s_dim = s_dim
        self.a_dim = a_dim

        # actor net: 
        # input state
        # output mu and sigma of a normal distribution
        # mu and sigma can describe the possibility of actions
        self.a1 = nn.Linear(s_dim, 200)
        self.mu = nn.Linear(200, a_dim)
        self.sigma = nn.Linear(200, a_dim)

        # critic net:
        # input state
        # output value
        self.c1 = nn.Linear(s_dim, 100)
        self.v = nn.Linear(100, 1)

        # set distribution function
        self.distribution = torch.distributions.Normal

    # input state
    # output mu, sigma, values
    def forward(self, x):
        # x -> a1
        a1 = F.relu6(self.a1(x))
        # a1 -> mu
        mu = 2 * torch.tanh(self.mu(a1))
        # a1 -> sigma
        # + 0.0001 to avoid 0
        sigma = F.softplus(self.sigma(a1)) + 0.0001
        # x -> c1
        c1 = F.relu6(self.c1(x))
        # c1 -> values
        values = self.v(c1)
        return mu, sigma, values

    # choose an action through normal distribution
    def choose_action(self, s):
        self.training = False
        mu, sigma, _ = self.forward(s)
        m = self.distribution(mu, sigma)
        # use clip to limit the action in [-2, 2]
        return m.sample().numpy().clip(-2, 2)

    # calculate loss
    def loss_func(self, s, a, R):
        self.train()

        # critic loss
        mu, sigma, values = self.forward(s)
        advantage = R - values
        c_loss = advantage.pow(2)

        # actor loss
        m = self.distribution(mu, sigma)
        log_prob = m.log_prob(a)
        entropy = 0.5 + 0.5 * math.log(2 * math.pi) + torch.log(m.scale)
        exp_v = log_prob * advantage.detach() + 0.005 * entropy
        a_loss = -exp_v

        total_loss = (a_loss + c_loss).mean()
        return total_loss


class Worker(mp.Process):

    def __init__(self, gnet, global_ep, global_ep_r, res_queue, id):
        super(Worker, self).__init__()
        # the process id
        self.id = id
        # g_ep: global episode count
        # g_ep_r: global average reward of last 10 episodes
        # res_queue: store every g_ep_r
        self.g_ep, self.g_ep_r, self.res_queue = global_ep, global_ep_r, res_queue
        # global net
        self.gnet = gnet
        # optimizer
        self.opt = SharedAdam(gnet.parameters(), lr=0.0002)
        # local net
        self.lnet = Net(N_S, N_A)
        # environment
        self.env = gym.make('Pendulum-v0').unwrapped

    # the main function of this process
    # when process start, this function will be run
    def run(self):
        total_step = 1
        # loop until g_ep >= MAX_EP
        while self.g_ep.value < MAX_EP:
            # reset the env
            s = self.env.reset()
            # buffers to store state, action, reward of every transition
            buffer_s, buffer_a, buffer_r = [], [], []
            # total reward of this episode
            ep_r = 0.0
            for t in range(MAX_EP_STEP):
                # show the movement in process 1
                if self.id == 1:
                    self.env.render()
                # choose an action from local net
                a = self.lnet.choose_action(torch.FloatTensor(s))
                # take the action
                s_, r, done, _ = self.env.step(a)
                # done if already take 200 steps
                if t == MAX_EP_STEP - 1: done = True
                
                # update episode reward
                ep_r += r
                # store acion, state, reward
                buffer_a.append(a)
                buffer_s.append(s)
                buffer_r.append((r+8.1)/8.1)

                if total_step % UPDATE_GLOBAL_ITER == 0 or done:
                    # learn every 5 steps
                    self.learn(s_, buffer_s, buffer_a, buffer_r)
                    # clear buffers
                    buffer_s, buffer_a, buffer_r = [], [], []
                    # if done, store episode reward and print
                    if done:
                        self.record(ep_r)
                        break
                # state <= next_state
                s = s_
                total_step += 1
        # if have finished MAX_EP episodes, return 0 through res_queue
        self.res_queue.put(0)
    
    # update lnet and gnet
    def learn(self, s_, bs, ba, br):
        # R <= 0 for terminal
        # R <= V(s_t) for non_terminal
        # but there is never terminal
        R = self.lnet.forward(torch.Tensor(s_))[-1][0].item()
        # R <= r_i + gamma * R
        buffer_R = []
        for r in br[::-1]:
            R = r + GAMMA * R
            buffer_R.append(R)
        buffer_R.reverse()
        # calculate loss
        loss = self.lnet.loss_func(
            torch.FloatTensor(bs),
            torch.FloatTensor(ba),
            torch.Tensor(buffer_R).view(-1,1))
        # update global net
        self.opt.zero_grad()
        loss.backward()
        for lp, gp in zip(self.lnet.parameters(), self.gnet.parameters()):
            gp._grad = lp.grad
        self.opt.step()
        # update local net
        # copy parameters from global net to local net
        self.lnet.load_state_dict(self.gnet.state_dict())

    # update g_ep, g_ep_r
    # return g_ep_r through res_queue
    # print g_ep and g_ep_r
    def record(self, ep_r):
        with self.g_ep.get_lock():
            self.g_ep.value += 1
        with self.g_ep_r.get_lock():
            if self.g_ep_r.value == 0.0:
                self.g_ep_r.value = ep_r
            else:
                self.g_ep_r.value = self.g_ep_r.value * 0.9 + ep_r * 0.1
        self.res_queue.put(self.g_ep_r.value)
        print("Ep:", self.g_ep.value, "| Ep_r: %d" % self.g_ep_r.value)
        # self.res_queue.put(ep_r)
        # print("Ep:", self.g_ep.value, "| Ep_r: %d" % ep_r)
    
    # when finish training
    # show the result of the policy
    def show(self):
        while True:
            s = self.env.reset()
            for t in range(MAX_EP_STEP):
                self.env.render()
                a = self.gnet.choose_action(torch.FloatTensor(s))
                s, _, _, _ = self.env.step(a)


if __name__ == "__main__":
    # initialize
    gnet = Net(N_S, N_A)
    gnet.share_memory()
    global_ep, global_ep_r, res_queue = mp.Value('i', 0), mp.Value('d', 0.), mp.Queue()
    workers = [Worker(gnet, global_ep, global_ep_r, res_queue, i) for i in range(N_CPU)]
    # start processes
    [w.start() for w in workers]
    res = []
    # get average rewards from res_queue
    while True:
        r = res_queue.get()
        if r:
            res.append(r)
        else:
            break
    [w.join() for w in workers]
    # processes stop
    # draw chart
    plt.plot(res)
    plt.ylabel('average ep reward')
    plt.xlabel('ep')
    plt.show()
    # show the movement of the pendulum using trained policy
    workers[0].show()
    
