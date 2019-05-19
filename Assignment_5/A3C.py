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
MAX_EP = 10000
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
        self.a1 = nn.Linear(s_dim, 200)
        self.mu = nn.Linear(200, a_dim)
        self.sigma = nn.Linear(200, a_dim)
        self.c1 = nn.Linear(s_dim, 100)
        self.v = nn.Linear(100, 1)
        self.distribution = torch.distributions.Normal

    def forward(self, x):
        a1 = F.relu6(self.a1(x))
        mu = 2 * torch.tanh(self.mu(a1))
        sigma = F.softplus(self.sigma(a1)) + 0.0001
        c1 = F.relu6(self.c1(x))
        values = self.v(c1)
        return mu, sigma, values

    def choose_action(self, s):
        self.training = False
        mu, sigma, _ = self.forward(s)
        m = self.distribution(mu, sigma)
        return m.sample().numpy().clip(-2, 2)

    def loss_func(self, s, a, R):
        self.train()
        mu, sigma, values = self.forward(s)
        advantage = R - values
        c_loss = advantage.pow(2)

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
        self.id = id
        self.g_ep, self.g_ep_r, self.res_queue = global_ep, global_ep_r, res_queue
        self.gnet = gnet
        self.opt = SharedAdam(gnet.parameters(), lr=0.0002)
        self.lnet = Net(N_S, N_A)
        self.env = gym.make('Pendulum-v0').unwrapped

    def run(self):
        total_step = 1
        while self.g_ep.value < MAX_EP:
            s = self.env.reset()
            buffer_s, buffer_a, buffer_r = [], [], []
            ep_r = 0.
            for t in range(MAX_EP_STEP):
                # if self.id == 1:
                #     self.env.render()
                a = self.lnet.choose_action(torch.FloatTensor(s))
                s_, r, done, _ = self.env.step(a)
                if t == MAX_EP_STEP - 1: done = True
                
                ep_r += r
                buffer_a.append(a)
                buffer_s.append(s)
                buffer_r.append((r+8.1)/8.1)

                if total_step % UPDATE_GLOBAL_ITER == 0 or done:
                    self.learn(done, s_, buffer_s, buffer_a, buffer_r)
                    buffer_s, buffer_a, buffer_r = [], [], []
                    if done:
                        self.record(ep_r)
                        break
                s = s_
                total_step += 1

        self.res_queue.put(0)
    
    def learn(self, done, s_, bs, ba, br):
        if done:
            R = 0.0
        else:
            R = self.lnet.forward(torch.Tensor(s_))[-1][0].item()

        buffer_R = []
        for r in br[::-1]:
            R = r + GAMMA * R
            buffer_R.append(R)
        buffer_R.reverse()
        loss = self.lnet.loss_func(
            torch.FloatTensor(bs),
            torch.FloatTensor(ba),
            torch.Tensor(buffer_R).view(-1,1))

        self.opt.zero_grad()
        loss.backward()
        for lp, gp in zip(self.lnet.parameters(), self.gnet.parameters()):
            gp._grad = lp.grad
        self.opt.step()

        self.lnet.load_state_dict(self.gnet.state_dict())

    def record(self, ep_r):
        with self.g_ep.get_lock():
            self.g_ep.value += 1
        with self.g_ep_r.get_lock():
            if self.g_ep_r.value == 0.0:
                self.g_ep_r.value = ep_r
            else:
                self.g_ep_r.value = self.g_ep_r.value * 0.99 + ep_r * 0.01
        self.res_queue.put(self.g_ep_r.value)
        print("Ep:", self.g_ep.value, "| Ep_r: %d" % self.g_ep_r.value)
        # self.res_queue.put(ep_r)
        # print("Ep:", self.g_ep.value, "| Ep_r: %d" % ep_r)
    
    def show(self):
        while True:
            s = self.env.reset()
            for t in range(MAX_EP_STEP):
                self.env.render()
                a = self.gnet.choose_action(torch.FloatTensor(s))
                s, _, _, _ = self.env.step(a)


if __name__ == "__main__":

    gnet = Net(N_S, N_A)
    gnet.share_memory()
    global_ep, global_ep_r, res_queue = mp.Value('i', 0), mp.Value('d', 0.), mp.Queue()

    workers = [Worker(gnet, global_ep, global_ep_r, res_queue, i) for i in range(N_CPU)]
    [w.start() for w in workers]
    res = []
    while True:
        r = res_queue.get()
        if r:
            res.append(r)
        else:
            break
    [w.join() for w in workers]
    plt.plot(res)
    plt.ylabel('average ep reward')
    plt.xlabel('ep')
    plt.show()
    workers[0].show()
    
