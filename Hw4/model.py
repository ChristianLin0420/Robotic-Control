import numpy as np
import torch
import torch.nn as nn

# Weight initialization
def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module

#Normal distribution module with fixed mean and std.
class FixedNormal(torch.distributions.Normal):
	# Log-probability
	def log_probs(self, actions):
		return super().log_prob(actions).sum(-1)

	# Entropy
	def entropy(self):
		return super().entropy().sum(-1)

	# Mode
	def mode(self):
		return self.mean

#Diagonal Gaussian distribution
class DiagGaussian(nn.Module):
	# Constructor
	def __init__(self, inp_dim, out_dim, std=0.5):
		super(DiagGaussian, self).__init__()

		init_ = lambda m: init(
			m,
			nn.init.orthogonal_,
			lambda x: nn.init.constant_(x, 0)
		)
		self.fc_mean = init_(nn.Linear(inp_dim, out_dim))
		self.std = torch.full((out_dim,), std)

	# Forward
	def forward(self, x):
		mean = self.fc_mean(x)
		return FixedNormal(mean, self.std.to(x.device))

#Policy network
class PolicyNet(nn.Module):
	# Constructor
	def __init__(self, s_dim, a_dim, std=0.5):
		super(PolicyNet, self).__init__()

		init_ = lambda m: init(
			m,
			nn.init.orthogonal_,
			lambda x: nn.init.constant_(x, 0),
			nn.init.calculate_gain('relu')
		)
		#TODO 1: policy network architecture
		'''
		self.main = ...
		self.dist = ...
		'''

	# Forward
	def forward(self, state, deterministic=False):
		feature = self.main(state)
		dist    = self.dist(feature)

		if deterministic:
			action = dist.mode()
		else:
			action = dist.sample()

		return action, dist.log_probs(action)

	# Output action
	def action_step(self, state, deterministic=True):
		feature = self.main(state)
		dist    = self.dist(feature)

		if deterministic:
			action = dist.mode()
		else:
			action = dist.sample()

		return action

	# Evaluate log-probs & entropy
	def evaluate(self, state, action):
		feature = self.main(state)
		dist    = self.dist(feature)
		return dist.log_probs(action), dist.entropy()

#Value network
class ValueNet(nn.Module):
	# Constructor
	def __init__(self, s_dim):
		super(ValueNet, self).__init__()

		init_ = lambda m: init(
			m,
			nn.init.orthogonal_,
			lambda x: nn.init.constant_(x, 0),
			nn.init.calculate_gain('relu')
		)
		#TODO 2: value network architecture
		'''
		self.main = ...
		'''

	# Forward
	def forward(self, state):
		return self.main(state)[:, 0]
