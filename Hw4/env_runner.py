import torch
import gym
import numpy as np
from collections import deque

# Compute discounted return
def compute_discounted_return(rewards, dones, last_values, last_dones, gamma=0.99):
	returns = np.zeros_like(rewards)
	n_step  = len(rewards)

	for t in reversed(range(n_step)):
		if t == n_step - 1:
			returns[t] = rewards[t] + gamma * last_values * (1.0 - last_dones)
		else:
			returns[t] = rewards[t] + gamma * returns[t+1] * (1.0 - dones[t+1])

	return returns

# Compute gae
def compute_gae(rewards, values, dones, last_values, last_dones, gamma=0.99, lamb=0.95):
	#rewards    : (n_step, n_env)
	#values     : (n_step, n_env)
	#dones      : (n_step, n_env)
	#advs       : (n_step, n_env)
	#last_values: (n_env)
	#last_dones : (n_env)
	advs         = np.zeros_like(rewards)
	n_step       = len(rewards)
	last_gae_lam = 0.0

	for t in reversed(range(n_step)):
		if t == n_step - 1:
			next_nonterminal = 1.0 - last_dones
			next_values = last_values
		else:
			next_nonterminal = 1.0 - dones[t+1]
			next_values = values[t+1]

		delta   = rewards[t] + gamma*next_values*next_nonterminal - values[t]
		advs[t] = last_gae_lam = delta + gamma*lamb*next_nonterminal*last_gae_lam

	return advs + values

#Runner for multiple environment
class EnvRunner:
	# Constructor
	def __init__(self, env, s_dim, a_dim, n_step=5, gamma=0.99, lamb=0.95, device='cpu'):
		self.env    = env
		self.n_env  = env.n_env
		self.s_dim  = s_dim
		self.a_dim  = a_dim
		self.n_step = n_step
		self.gamma  = gamma
		self.lamb   = lamb
		self.device = device

		#last states: (n_env, s_dim)
		#last dones : (n_env)
		self.states = self.env.reset()
		self.dones  = np.ones((self.n_env), dtype=np.bool)

		#Storages (state, action, value, reward, a_logp, done)
		self.mb_states  = np.zeros((self.n_step, self.n_env, self.s_dim), dtype=np.float32)
		self.mb_actions = np.zeros((self.n_step, self.n_env, self.a_dim), dtype=np.float32)
		self.mb_values  = np.zeros((self.n_step, self.n_env), dtype=np.float32)
		self.mb_rewards = np.zeros((self.n_step, self.n_env), dtype=np.float32)
		self.mb_a_logps = np.zeros((self.n_step, self.n_env), dtype=np.float32)
		self.mb_dones   = np.zeros((self.n_step, self.n_env), dtype=np.bool)
		

		#Reward & length recorder
		self.total_rewards = np.zeros((self.n_env), dtype=np.float32)
		self.total_len = np.zeros((self.n_env), dtype=np.int32)
		self.reward_buf = deque(maxlen=100)
		self.len_buf = deque(maxlen=100)

	# Run n steps to get a batch
	def run(self, policy_net, value_net):
		#1. Run n steps
		#-------------------------------------
		for step in range(self.n_step):
			#self.states : (n_env, s_dim)
			#actions     : (n_env, a_dim)
			#self.dones  : (n_env)
			#a_logps     : (n_env)
			#values      : (n_env)
			#rewards     : (n_env)
			#TODO 3: Run a step to collect data
			'''
			self.mb_states[step, :]  = ...
			self.mb_dones[step, :]   = ...
			self.mb_actions[step, :] = ...
			self.mb_a_logps[step, :] = ...
			self.mb_values[step, :]  = ...
			self.states, rewards, self.dones, info = self.env.step(actions)
			self.mb_rewards[step, :] = ...
			'''

		last_values = value_net(torch.from_numpy(self.states).float().to(self.device)).cpu().numpy()
		self.record()

		#2. Compute returns
		#-------------------------------------
		mb_returns = compute_gae(self.mb_rewards, self.mb_values, self.mb_dones, last_values, self.dones, self.gamma, self.lamb)

		#mb_states : (n_step*n_env, s_dim)
		#mb_actions: (n_step*n_env) or (n_env*n_step, a_dim)
		#mb_a_logps: (n_step*n_env)
		#mb_values : (n_step*n_env)
		#mb_returns: (n_step*n_env)
		return self.mb_states.reshape(self.n_step*self.n_env, self.s_dim), \
				self.mb_actions.reshape(self.n_step*self.n_env, self.a_dim), \
				self.mb_a_logps.flatten(), \
				self.mb_values.flatten(), \
				mb_returns.flatten()

	# Record return & length
	def record(self):
		for i in range(self.n_step):
			for j in range(self.n_env):
				if self.mb_dones[i, j]:
					self.reward_buf.append(self.total_rewards[j] + self.mb_rewards[i, j])
					self.len_buf.append(self.total_len[j] + 1)
					self.total_rewards[j] = 0
					self.total_len[j] = 0
				else:
					self.total_rewards[j] += self.mb_rewards[i, j]
					self.total_len[j] += 1

	# Get performance
	def get_performance(self):
		if len(self.reward_buf) == 0:
			mean_return = 0
			std_return  = 0
		else:
			mean_return = np.mean(self.reward_buf)
			std_return  = np.std(self.reward_buf)

		if len(self.len_buf) == 0:
			mean_len = 0
		else:
			mean_len = np.mean(self.len_buf)

		return mean_return, std_return, mean_len
