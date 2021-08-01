
import gym
import os
from torch._C import device
from rl_env.adv_spd_env import AdvSpdEnv

import numpy as np

from stable_baselines3 import PPO, SAC, DDPG, A2C, DQN, TD3
from stable_baselines3.common.callbacks import CheckpointCallback
# scenario = np.ones(shape=(1, 200, 40))
# scenario[0, 100, 20:] = 1
import pickle
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--cuda', default='0', type=str)
parser.add_argument('--model', default='DDPG', type=str)
parser.add_argument('--coef-vel', default=1, type=float)
parser.add_argument('--coef-shock', default=1, type=float)
parser.add_argument('--coef-jerk', default=1, type=float)
parser.add_argument('--coef-power', default=0.01, type=float)
parser.add_argument('--coef-tt', default=0, type=float)
args = parser.parse_args()

cuda = args.cuda
os.environ["CUDA_VISIBLE_DEVICES"] = cuda

env = AdvSpdEnv(reward_coef=[args.coef_vel,
                             args.coef_shock,
                             args.coef_jerk,
                             args.coef_power,
                             args.coef_tt,
                             1]  # coef_signal_violation
                )

env = gym.wrappers.TimeLimit(env, max_episode_steps=2400)

model = args.model
checkpoint_callback = CheckpointCallback(save_freq=100000, save_path=f"./params/{model}{int(cuda)}", name_prefix=f"AdvSpdRL_{model}")

directory = f'params/{model}{int(cuda)}'
if not os.path.exists(directory):
    os.mkdir(directory)

pickle.dump(env, open(f"params/{model}{int(cuda)}/env.pkl", 'wb'))

model = globals()[model]("MlpPolicy", env, verbose=1, tensorboard_log=f"log/{model}/", device='cuda')
# model.save("params/AdvSpdRL")
model.learn(total_timesteps=1000000000, callback=checkpoint_callback)
model.save("params/AdvSpdRL_PPO")

# checkpoint_callback = CheckpointCallback(save_freq=10000, save_path="./params/SAC", name_prefix="AdvSpdRL_SAC")

# model = SAC("MlpPolicy", env, verbose=1, tensorboard_log="log/SAC/", device='cuda')
# # model.save("params/AdvSpdRL")
# model.learn(total_timesteps=10000000, callback=checkpoint_callback, log_interval=10)
# model.save("params/AdvSpdRL_SAC")


# ob = env.reset()
# episode_over = False
# ob_list = []
# while not episode_over:
#     # action = env.get_random_action()
#     action , _ = model.predict(ob, deterministic=True)
#     ob, reward, episode_over, info = env.step(action)
#     ob_list.append(ob)
#     env.render()
