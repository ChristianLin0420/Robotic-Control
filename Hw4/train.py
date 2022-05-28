from multi_env import MultiEnv, make_env
from env_runner import EnvRunner
from model import PolicyNet, ValueNet
from agent import PPO
import torch
import os
import time
import numpy as np

def main():
	#TODO 5: Adjust these parameters if needed
	#Parameters that can be modified
	#----------------------------
	n_env          = 8
	n_step         = 128
	sample_mb_size = 64
	sample_n_epoch = 4
	a_std          = 0.5
	lamb           = 0.95
	gamma          = 0.99
	clip_val       = 0.2
	lr             = 1e-4
	n_iter         = 30000
	device         = 'cpu'

	#Parameters that are fixed
	#----------------------------
	s_dim          = 14
	a_dim          = 1
	mb_size        = n_env*n_step
	max_grad_norm  = 0.5
	disp_step      = 20
	save_step      = 100
	check_step     = 500
	save_dir       = './save'

	#Create multiple environments
	#----------------------------
	env    = MultiEnv([make_env(i, rand_seed=int(time.time())) for i in range(n_env)])
	runner = EnvRunner(
		env,
		s_dim,
		a_dim,
		n_step,
		gamma,
		lamb,
		device=device
	)

	#Create model
	#----------------------------
	policy_net = PolicyNet(s_dim, a_dim, a_std).to(device)
	value_net  = ValueNet(s_dim).to(device)
	agent      = PPO(
		policy_net,
		value_net,
		lr,
		max_grad_norm,
		clip_val,
		sample_n_epoch,
		sample_mb_size,
		mb_size,
		device=device
	)

	#Load model
	#----------------------------
	if not os.path.exists(save_dir):
		os.mkdir(save_dir)

	if os.path.exists(os.path.join(save_dir, "model.pt")):
		print("Loading the model ... ", end="")
		state_dict = torch.load(os.path.join(save_dir, "model.pt"))
		policy_net.load_state_dict(state_dict["PolicyNet"])
		value_net.load_state_dict(state_dict["ValueNet"])
		start_it = state_dict["it"]
		print("Done.")
	else:
		start_it = 0

	#Start training
	#----------------------------
	t_start = time.time()
	policy_net.train()
	value_net.train()

	for it in range(start_it, n_iter):
		#Run the environment
		with torch.no_grad():
			mb_obs, mb_actions, mb_old_a_logps, mb_values, mb_returns = runner.run(policy_net, value_net)
			mb_advs = mb_returns - mb_values
			mb_advs = (mb_advs - mb_advs.mean()) / (mb_advs.std() + 1e-6)

		#Train
		pg_loss, v_loss = agent.train(
			mb_obs,
			mb_actions,
			mb_values,
			mb_advs,
			mb_returns,
			mb_old_a_logps
		)

		#Print the result
		if it % disp_step == 0:
			agent.lr_decay(it, n_iter)
			n_sec = time.time() - t_start
			fps = int((it - start_it)*n_env*n_step / n_sec)
			mean_return, std_return, mean_len = runner.get_performance()

			print("[{:5d} / {:5d}]".format(it, n_iter))
			print("----------------------------------")
			print("Timesteps    = {:d}".format((it - start_it) * mb_size))
			print("Elapsed time = {:.2f} sec".format(n_sec))
			print("FPS          = {:d}".format(fps))
			print("actor loss   = {:.6f}".format(pg_loss))
			print("critic loss  = {:.6f}".format(v_loss))
			print("mean return  = {:.6f}".format(mean_return))
			print("mean length  = {:.2f}".format(mean_len))
			print()

		#Save model
		if it % save_step == 0:
			print("Saving the model ... ", end="")
			torch.save({
				"it": it,
				"PolicyNet": policy_net.state_dict(),
				"ValueNet": value_net.state_dict()
			}, os.path.join(save_dir, "model.pt"))
			print("Done.")
			print()

			if it - start_it >= save_step:
				with open(os.path.join(save_dir, "return.txt"), "a") as file:
					file.write("{:d},{:.4f},{:.4f}\n".format(it, mean_return, std_return))

		#Save checkpoint
		if it % check_step == 0:
			print("Saving the checkpoint ... ", end="")
			torch.save({
				"it": it,
				"PolicyNet": policy_net.state_dict(),
				"ValueNet": value_net.state_dict()
			}, os.path.join(save_dir, "model-{:05d}.pt".format(it)))
			print("Done.")
			print()

	env.close()

if __name__ == '__main__':
	main()
