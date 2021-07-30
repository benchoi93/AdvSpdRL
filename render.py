import time
from rl_env.adv_spd_env import AdvSpdEnv

import numpy as np

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import CheckpointCallback
# scenario = np.ones(shape=(1, 200, 40))
# scenario[0, 100, 20:] = 1

env = AdvSpdEnv()

model = SAC("MlpPolicy", env, verbose=1)
model = model.load("params/SAC/AdvSpdRL_SAC_2470000_steps")
# self = env

ob = env.reset()
episode_over = False
ob_list = []
while not episode_over:
    # action = np.array(env.get_random_action())
    action, _ = model.predict(ob)
    # print(action)
    ob, reward, episode_over, info = env.step(action)
    ob_list.append(ob)
    env.render()
    # input()
env.viewer.close()
