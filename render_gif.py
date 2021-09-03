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

modelname = 'PPO'
cuda = '1'
# coef_power = 0.01
# param = 'AdvSpdRL_DDPG_3500000_steps'

# try:
env = pickle.load(open(os.path.join('params', f'{modelname}{cuda}', 'env.pkl'), 'rb'))
# except:
#     print("cannot load predefined Env, loading default env")
#     env = AdvSpdEnv()

model = globals()[modelname]("MlpPolicy", env, verbose=1, device='cpu')
list_of_files = glob.glob(os.path.join('params', f'{modelname}{cuda}/*.zip'))
# latest_file = max(list_of_files, key=os.path.getmtime)


def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


ensure_dir('simulate_gif')
ensure_dir(os.path.join('simulate_gif', f'{modelname}{cuda}'))

list_of_files = sorted(list_of_files, key=lambda x: int(x.split("_")[2]))

for file in list_of_files:
    start = time.time()

    model = model.load(file, device='cpu')
    # env = env

    ob = env.reset()
    episode_over = False
    ob_list = []

    combine = False

    ob_list.append([0, 0, 0, 0, 0])
    env.car_moving(ob_list, startorfinish=1, combine=combine)
    print('-------------------------------------')

    while not episode_over:
        t = time.time()
        # action = np.array(env.get_random_action())
        action, _ = model.predict(ob)
        # action = np.array(min(5, (50/3.6 - ob[1]) / env.dt))
        # print(action)
        ob, reward, episode_over, info = env.step(action)
        ob_list.append([env.vehicle.position, env.vehicle.velocity, env.vehicle.acceleration, env.timestep, reward])
        # env.render(visible=True)
        env.car_moving(ob_list, startorfinish=0, combine=combine)
        check_start = 0
        # input()

    env.car_moving(ob_list, startorfinish=1, combine=combine)
    env.info_graph(ob_list, check_finish=1)
    env.make_gif(path=os.path.join('simulate_gif', f'{modelname}{cuda}', f'{file.split("/")[2].split(".")[0]}.gif'))
    print('-------------------------------------')
    print("reward coef: ", env.reward_coef)
    print("reward: {}".format(sum([x[4] for x in ob_list])))
    print("execution time: {}".format(np.round(time.time()-start), 5))
    print("# of episodes: {}".format(len(ob_list)))
    print("execution time per episode: {}".format((time.time()-start)/len(ob_list)))
