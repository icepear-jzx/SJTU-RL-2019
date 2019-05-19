import gym
import numpy as np
import torch
from torch.autograd import Variable
import torch.multiprocessing as mp
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import os
import random
import matplotlib.pyplot as plt

# It seems that this line can speed up the process.
os.environ["OMP_NUM_THREADS"] = "1"

# some parameters
MAX_EPISODES = 200
MAX_STEPS = 200
env = gym.make('Pendulum-v0')
S_DIM = env.observation_space.shape[0]
A_DIM = env.action_space.shape[0]
A_MAX = env.action_space.high[0]


# return a noise
# Based on http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab
class OrnsteinUhlenbeckActionNoise:

	def __init__(self, mu = 0, theta = 0.15, sigma = 0.2):
		self.action_dim = A_DIM
		self.mu = mu
		self.theta = theta
		self.sigma = sigma
		self.X = np.ones(self.action_dim) * self.mu

	def reset(self):
		self.X = np.ones(self.action_dim) * self.mu

	def sample(self):
		dx = self.theta * (self.mu - self.X)
		dx = dx + self.sigma * np.random.randn(len(self.X))
		self.X = self.X + dx
		return self.X


# memory to store past transitions
class Memory:

	def __init__(self):
		self.memory = []
		self.len = 0
		self.capacity = 10000
		self.batch_size = 32

	# get a batch of transitions from memory
	def sample(self):
		batch = random.sample(self.memory, self.batch_size)
		[s, a, r, s_] = zip(*batch)
		s = torch.Tensor(s)
		a = torch.Tensor(a)
		r = torch.Tensor(r)
		s_ = torch.Tensor(s_)
		return s, a, r, s_

	# push a transition into memory
	def push(self, s, a, r, s1):
		transition = (s,a,r,s1)	
		if self.len >= self.capacity:
			self.len = 0
			self.memory[self.len] = transition
		else:
			self.len += 1
			self.memory.append(transition)


class Critic(nn.Module):

	def __init__(self):
		super(Critic, self).__init__()
		self.state_dim = S_DIM
		self.action_dim = A_DIM

		self.fcs1 = nn.Linear(self.state_dim,64)
		self.fcs2 = nn.Linear(64,32)
		self.fca1 = nn.Linear(self.action_dim,32)
		self.fc2 = nn.Linear(64,32)
		self.fc3 = nn.Linear(32,1)

	def forward(self, state, action):
		s1 = F.relu(self.fcs1(state))
		s2 = F.relu(self.fcs2(s1))
		a1 = F.relu(self.fca1(action))
		x = torch.cat((s2,a1),dim=1)
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		return x


class Actor(nn.Module):

	def __init__(self):
		super(Actor, self).__init__()
		self.state_dim = S_DIM
		self.action_dim = A_DIM
		self.action_lim = A_MAX

		self.fc1 = nn.Linear(self.state_dim,64)
		self.fc2 = nn.Linear(64,32)
		self.fc3 = nn.Linear(32,self.action_dim)

	def forward(self, state):
		x = F.relu(self.fc1(state))
		x = F.relu(self.fc2(x))
		action = torch.tanh(self.fc3(x))
		action = action * self.action_lim
		return action


class Agent:

	def __init__(self, memory):
		# some parameters
		self.state_dim = S_DIM
		self.action_dim = A_DIM
		self.action_lim = A_MAX
		self.tau = 0.001
		self.lr = 0.001
		self.gamma = 0.99
		self.memory = memory

		self.noise = OrnsteinUhlenbeckActionNoise()

		self.actor = Actor()
		self.target_actor = Actor()
		self.actor_optimizer = optim.Adam(self.actor.parameters(),lr=self.lr)

		self.critic = Critic()
		self.target_critic = Critic()
		self.critic_optimizer = optim.Adam(self.critic.parameters(),lr=self.lr)

		# copy parameters from actor to target_actor
		self.hard_update(self.target_actor, self.actor)
		# copy parameters from critic to target_critic
		self.hard_update(self.target_critic, self.critic)

	def get_exploration_action(self, state):
		state = torch.Tensor(state)
		action = self.actor.forward(state).detach()
		new_action = action + torch.Tensor(self.noise.sample() * self.action_lim)
		return new_action.numpy()

	def hard_update(self, target, source):
		for target_param, param in zip(target.parameters(), source.parameters()):
			target_param.data.copy_(param.data)
	
	def soft_update(self, target, source):
		for target_param, param in zip(target.parameters(), source.parameters()):
			target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

	def learn(self):
		s1,a1,r1,s2 = self.memory.sample()
		s1 = Variable(s1)
		a1 = Variable(a1)
		r1 = Variable(r1)
		s2 = Variable(s2)

		# ---------------------- optimize critic ----------------------
		a2 = self.target_actor.forward(s2).detach()
		next_val = self.target_critic.forward(s2, a2).detach().view(-1)
		y_expected = r1 + self.gamma * next_val
		y_predicted = self.critic.forward(s1, a1).view(-1)
		
		loss_critic = F.smooth_l1_loss(y_predicted, y_expected)
		
		self.critic_optimizer.zero_grad()
		loss_critic.backward()
		self.critic_optimizer.step()

		# ---------------------- optimize actor ----------------------
		pred_a1 = self.actor.forward(s1)
		loss_actor = -1 * torch.sum(self.critic.forward(s1, pred_a1))
		self.actor_optimizer.zero_grad()
		loss_actor.backward()
		self.actor_optimizer.step()

		self.soft_update(self.target_actor, self.actor)
		self.soft_update(self.target_critic, self.critic)



def main():
	memory = Memory()
	agent = Agent(memory)
	res = []
	avg_res = []
	for episodes in range(MAX_EPISODES):
		state = env.reset()
		ep_r = 0
		for steps in range(MAX_STEPS):
			# if episodes % 10 == 0:
			# 	env.render()

			action = agent.get_exploration_action(state)

			next_state, reward, _, _ = env.step(action)

			memory.push(state, action, reward, next_state)

			if episodes > 2:
				agent.learn()
			
			ep_r += reward
			state = next_state
		
		res.append(ep_r)
		if avg_res:
			avg_res.append(avg_res[-1] * 0.9 + ep_r * 0.1)
		else:
			avg_res.append(ep_r)
		print("Ep:", episodes, "| Ep_r: %d" % ep_r)

	plt.ion()
	plt.figure()
	plt.plot(res)
	plt.plot(avg_res)
	plt.ylabel("ep reward")
	plt.xlabel("ep")
	plt.ioff()
	plt.show()


if __name__ == "__main__":
	p = mp.Process(target=main)
	p.start()
	p.join()