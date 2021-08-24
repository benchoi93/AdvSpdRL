import os
import glob
import pickle
import matplotlib.pyplot as plt
import time
from rl_env.adv_spd_env_road import AdvSpdEnvRoad
from PIL import Image
import numpy as np

from stable_baselines3 import PPO, SAC, DDPG, A2C, DQN, TD3
from stable_baselines3.common.callbacks import CheckpointCallback
# scenario = np.ones(shape=(1, 200, 40))
# scenario[0, 100, 20:] = 1

import time
start = time.time()

modelname = 'PPO'
cuda = '0'
# coef_power = 0.01
# param = 'AdvSpdRL_DDPG_3500000_steps'

# try:
env: AdvSpdEnvRoad = pickle.load(open(os.path.join('params', f'{modelname}{cuda}', 'env.pkl'), 'rb'))
# env = AdvSpdEnvRoad(reward_coef=env.reward_coef)

model = globals()[modelname]("MlpPolicy", env, verbose=1, device='cpu')
list_of_files = glob.glob(os.path.join('params', f'{modelname}{cuda}/*'))
latest_file = max(list_of_files, key=os.path.getmtime)
model = model.load(latest_file, device='cpu')
# env = env

ob = env.reset()
episode_over = False
combine = True

env.car_moving(env.ob_list, startorfinish=True, combine=combine)

while not episode_over:
    t = time.time()
    # action = np.array(env.get_random_action())
    action, _ = model.predict(ob)
    # action = env.get_random_action()
    # print("action space: ", action)
    # action = np.array(min(5, (50/3.6 - ob[1]) / env.dt))
    # print(action)
    ob, reward, episode_over, info = env.step(action)

    # info[0]: vehicle position / info[1]: vehicle velocity / info[2]: vehicle acceleration / info[3]: timestep
    # ob_list.append([info[0], info[1], info[2], info[3], reward])
    # env.render(visible=True)
    # env.car_moving(env.ob_list, startorfinish=0, combine=combine)
    # print('-------------------------------------')
    # input()

env.car_moving(env.ob_list, startorfinish=True, combine=combine)
env.info_graph(env.ob_list, check_finish=True)
env.make_gif(path=f'simulate_gif/{modelname}{cuda}/simulate.gif')

print('-------------------------------------')
print("reward coef: ", env.reward_coef)
print("reward: {}".format(sum([x[4] for x in env.ob_list])))
print("execution time: {}".format(np.round(time.time()-start), 5))
print("# of episodes: {}".format(len(env.ob_list)))
print("execution time per episode: {}".format((time.time()-start)/len(env.ob_list)))
