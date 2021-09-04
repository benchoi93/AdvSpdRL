import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import glob
import pickle
import matplotlib.pyplot as plt
import time
from rl_env.adv_spd_env_road_multi import AdvSpdEnvRoadMulti
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
# env: AdvSpdEnvRoadMulti = pickle.load(open(os.path.join('params', f'{modelname}{cuda}', 'env.pkl'), 'rb'))
env = AdvSpdEnvRoadMulti(num_signal=3, num_action_unit=3)

model = globals()[modelname]("MlpPolicy", env, verbose=1, device='cpu')
list_of_files = glob.glob(os.path.join('params', f'{modelname}{cuda}/*'))
latest_file = max(list_of_files, key=os.path.getmtime)
# model = model.load(latest_file, device='cpu')
# env = env

ob = env.reset()
episode_over = False
combine = True

while not episode_over:
    t = time.time()
    # action = np.array(env.get_random_action())
    # action, _ = model.predict(ob)
    action = env.get_random_action()
    print(action)
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

print(env.timestep/10)
print("-------------------------------------")

env.car_moving(env.vehicle.veh_info[:1], startorfinish=True, combine=combine)

for i in range(env.timestep):
    env.car_moving(env.vehicle.veh_info[:i+2], startorfinish=False, combine=True)

env.car_moving(env.vehicle.veh_info[:env.timestep+1], startorfinish=True, combine=combine)
env.info_graph(env.vehicle.veh_info[:env.timestep+1], check_finish=True)

finish = time.time()

env.make_gif(path=f'simulate_gif/{modelname}{cuda}/simulate.gif')

print('-------------------------------------')
print("reward coef: ", env.reward_coef)
print("reward: {}".format((env.vehicle.veh_info[:, 4]).sum()))
print("execution time: {}".format(np.round(finish-start), 5))
print("# of episodes: {}".format(env.timestep))
print("execution time per episode: {}".format((finish-start)/env.timestep))
