from model import PolicyNet
import cv2
import torch
import numpy as np
import os
import wrapper
import argparse

def main():
	#Parse arguments
	#----------------------------
	parser = argparse.ArgumentParser()
	parser.add_argument("--stoch", action="store_true", help='Use a stochastic policy (default is deterministic)')
	args = parser.parse_args()

	#Parameters that are fixed
	#----------------------------
	s_dim    = 14
	a_dim    = 1
	save_dir = './save'
	device   = 'cpu'

	#Create environment & model
	#----------------------------
	env = wrapper.PathTrackingEnv()
	policy_net = PolicyNet(s_dim, a_dim).to(device)

	#Load model
	#----------------------------
	if os.path.exists(os.path.join(save_dir, "model.pt")):
		print("Loading the model ... ", end="")
		checkpoint = torch.load(os.path.join(save_dir, "model.pt"))
		policy_net.load_state_dict(checkpoint["PolicyNet"])
		print("Done.")
	else:
		print("Error: No model saved")

	#Start playing
	#----------------------------
	policy_net.eval()

	for it in range(3):
		ob, _ = env.reset()
		total_reward = 0
		length = 0

		while True:
			#Render
			if cv2.waitKey(1) == 27: break
			cv2.imshow("PPO Play", env.render())

			#Step
			state_tensor = torch.tensor(np.expand_dims(ob, axis=0), dtype=torch.float32, device=device)
			action = policy_net.action_step(state_tensor, deterministic=not args.stoch).cpu().detach().numpy()
			ob, reward, done, info = env.step(action[0])
			total_reward += reward
			length += 1
			if done: break

		print("Total reward = {:.6f}, length = {:d}".format(total_reward, length), flush=True)

	cv2.destroyWindow("PPO Play")

if __name__ == '__main__':
	main()
