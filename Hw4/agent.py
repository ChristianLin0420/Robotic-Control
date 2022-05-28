import torch
import torch.nn as nn
import numpy as np

#PPO Agent Class
class PPO:
	#Constructor
	def __init__(
		self,
		policy_net,
		value_net,
		lr=1e-4,
		max_grad_norm=0.5,
		clip_val=0.2,
		sample_n_epoch=4,
		sample_mb_size=64,
		mb_size=1024,
		device='cpu'
	):
		self.policy_net     = policy_net
		self.value_net      = value_net
		self.opt_actor      = torch.optim.Adam(self.policy_net.parameters(), lr)
		self.opt_critic     = torch.optim.Adam(self.value_net.parameters(), lr)
		self.device         = device
		self.lr             = lr
		self.max_grad_norm  = max_grad_norm
		self.clip_val       = clip_val
		self.sample_n_epoch = sample_n_epoch
		self.sample_mb_size = sample_mb_size
		self.sample_n_mb    = mb_size // sample_mb_size
		self.rand_idx       = np.arange(mb_size)

	#Train PPO
	def train(self, mb_states, mb_actions, mb_old_values, mb_advs, mb_returns, mb_old_a_logps):
		mb_states      = torch.from_numpy(mb_states).to(self.device)
		mb_actions     = torch.from_numpy(mb_actions).to(self.device)
		mb_old_values  = torch.from_numpy(mb_old_values).to(self.device)
		mb_advs        = torch.from_numpy(mb_advs).to(self.device)
		mb_returns     = torch.from_numpy(mb_returns).to(self.device)
		mb_old_a_logps = torch.from_numpy(mb_old_a_logps).to(self.device)

		for i in range(self.sample_n_epoch):
			np.random.shuffle(self.rand_idx)

			for j in range(self.sample_n_mb):
				sample_idx         = self.rand_idx[j*self.sample_mb_size : (j+1)*self.sample_mb_size]
				sample_states      = mb_states[sample_idx]
				sample_actions     = mb_actions[sample_idx]
				sample_old_values  = mb_old_values[sample_idx]
				sample_advs        = mb_advs[sample_idx]
				sample_returns     = mb_returns[sample_idx]
				sample_old_a_logps = mb_old_a_logps[sample_idx]

				sample_a_logps, sample_ents = self.policy_net.evaluate(sample_states, sample_actions)
				sample_values = self.value_net(sample_states)

				#PPO loss
				v_pred_clip = sample_old_values + torch.clamp(sample_values - sample_old_values, -self.clip_val, self.clip_val)
				v_loss1     = (sample_returns - sample_values).pow(2)
				v_loss2     = (sample_returns - v_pred_clip).pow(2)
				v_loss      = torch.max(v_loss1, v_loss2).mean()

				#TODO 4: Policy gradient loss for PPO
				'''
				pg_loss  = ...
				'''

				#Train actor
				self.opt_actor.zero_grad()
				pg_loss.backward()
				nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.max_grad_norm)
				self.opt_actor.step()

				#Train critic
				self.opt_critic.zero_grad()
				v_loss.backward()
				nn.utils.clip_grad_norm_(self.value_net.parameters(), self.max_grad_norm)
				self.opt_critic.step()

		return pg_loss.item(), v_loss.item()

	#Linear learning rate decay
	def linear_lr_decay(self, opt, it, n_it, initial_lr):
		lr = initial_lr - (initial_lr * (it / float(n_it)))

		for param_group in opt.param_groups:
			param_group['lr'] = lr

	#Learning rate decay
	def lr_decay(self, it, n_it):
		self.linear_lr_decay(self.opt_actor, it, n_it, self.lr)
		self.linear_lr_decay(self.opt_critic, it, n_it, self.lr)
