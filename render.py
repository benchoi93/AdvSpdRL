import os
import glob
import pickle
import matplotlib.pyplot as plt
import time
from rl_env.adv_spd_env import AdvSpdEnv
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
env = pickle.load(open(os.path.join('params', f'{modelname}{cuda}', 'env.pkl'), 'rb'))
# except:
#     print("cannot load predefined Env, loading default env")
#     env = AdvSpdEnv()

model = globals()[modelname]("MlpPolicy", env, verbose=1)
list_of_files = glob.glob(os.path.join('params', f'{modelname}{cuda}/*'))
latest_file = max(list_of_files, key=os.path.getmtime)
# model = model.load(latest_file)
# env = env

ob = env.reset()
episode_over = False
ob_list = []

print(env.reward_coef)
check_start = 1
check_finish = 0
while not episode_over:
    t = time.time()
    # action = np.array(env.get_random_action())
    action, _ = model.predict(ob)
    # action = np.array(min(5, (50/3.6 - ob[1]) / env.dt))
    # print(action)
    ob, reward, episode_over, info = env.step(action)
    ob_list.append([env.vehicle.position, env.vehicle.velocity, env.vehicle.acceleration, env.timestep, reward])
    # env.render(visible=True)
    env.car_moving(ob_list, check_start, check_finish, combine=False)
    print("one episode: {}".format(time.time()-t))
    print('-------------------------------------')
    # input()

check_finish = 1
# print(ob_list)
print(sum([x[4] for x in ob_list]))
# env.car_moving(ob_list)
env.car_moving(ob_list, check_start, check_finish)
env.info_graph(ob_list)
env.make_gif()
# input()
# env.viewer.close()

print("execution time: {}".format(np.round(time.time()-start), 5))



