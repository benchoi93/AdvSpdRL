
import os
from torch._C import device
from rl_env.adv_spd_env import AdvSpdEnv

import numpy as np

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import CheckpointCallback
# scenario = np.ones(shape=(1, 200, 40))
# scenario[0, 100, 20:] = 1

env = AdvSpdEnv()
cuda = '0'
os.environ["CUDA_VISIBLE_DEVICES"] = cuda

checkpoint_callback = CheckpointCallback(save_freq=100000, save_path=f"./params/PPO{int(cuda)}", name_prefix="AdvSpdRL_PPO")

model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="log/PPO/", device='cuda')
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
