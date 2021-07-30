
from torch._C import device
from rl_env.adv_spd_env import AdvSpdEnv

import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
# scenario = np.ones(shape=(1, 200, 40))
# scenario[0, 100, 20:] = 1

env = AdvSpdEnv()
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"

checkpoint_callback = CheckpointCallback(save_freq = 1000000 , save_path = "./params/", name_prefix= "AdvSpdRL")

model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="log/PPO/",device='cuda')
model.save("params/AdvSpdRL")
model.learn(total_timesteps=10000000, callback= checkpoint_callback)
model.save("params/AdvSpdRL")

# ob = env.reset()
# episode_over = False
# ob_list = []
# while not episode_over:
#     # action = env.get_random_action()
#     action , _ = model.predict(ob, deterministic=True)
#     ob, reward, episode_over, info = env.step(action)
#     ob_list.append(ob)
#     env.render()
