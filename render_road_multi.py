from pathlib import Path
from util.plotutil import info_graph
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3 import PPO, SAC, DDPG, A2C, DQN, TD3
import numpy as np
from PIL import Image
from rl_env.adv_spd_env_road_multi import AdvSpdEnvRoadMulti
import time
import matplotlib.pyplot as plt
import pickle
import glob
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# scenario = np.ones(shape=(1, 200, 40))
# scenario[0, 100, 20:] = 1

start = time.time()

modelname = 'PPO'
cuda = '0'
i = 0

# coef_power = 0.01
# param = 'AdvSpdRL_DDPG_3500000_steps'
# for cuda in range(4):
#     cuda = str(cuda)
#     for i in range(100):
# try:
env: AdvSpdEnvRoadMulti = pickle.load(open(os.path.join('params', f'{modelname}{cuda}', 'env.pkl'), 'rb'))
# env = AdvSpdEnvRoadMulti(num_signal=3, num_action_unit=3)

model = globals()[modelname]("MlpPolicy", env, verbose=1, device='cpu')
list_of_files = glob.glob(os.path.join('params', f'{modelname}{cuda}/*'))
latest_file = max(list_of_files, key=os.path.getmtime)
model = model.load(latest_file, device='cpu')
# env = env
path = Path(f'simulate_gif/{modelname}{cuda}/')
if not path.exists():
    path.mkdir()

ob = env.reset()
pickle.dump(env, open(f'simulate_gif/{modelname}{cuda}/env_{i}.pkl', 'wb'))

episode_over = False
combine = True

cnt = 0
print("-------------------------------------")

while not episode_over:
    t = time.time()
    # action = np.array(env.get_random_action())
    action, _ = model.predict(ob)
    # action = env.get_random_action()
    # print(action)
    print(f"{cnt=} || {action=}")
    cnt += 1
    # print("action space: ", action)
    # action = np.array([9, 9, 9])
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

# env.car_moving(env.vehicle.veh_info[:1], startorfinish=True, combine=combine)

# for i in range(env.timestep):
#     env.car_moving(env.vehicle.veh_info[:i+2], startorfinish=False, combine=True)

# env.car_moving(env.vehicle.veh_info[:env.timestep+1], startorfinish=True, combine=combine)

# info_graph(env, env.vehicle.veh_info[:env.timestep+1], check_finish=True, path=f'simulate_gif/{modelname}{cuda}/infograph_{i}.png')

env1 = env

finish = time.time()

env = pickle.load(open(f'simulate_gif/{modelname}{cuda}/env_{i}.pkl', 'rb'))

episode_over = False
combine = True

cnt = 0
print("-------------------------------------")

while not episode_over:
    action = np.array([9])
    print(f"{cnt=} || {action=}")
    cnt += 1
    ob, reward, episode_over, info = env.step(action)


print(env.timestep/10)
print("-------------------------------------")

# info_graph(env, env.vehicle.veh_info[:env.timestep+1], check_finish=True, path=f'simulate_gif/{modelname}{cuda}/infograph_base_{i}.png')

env2 = env

env_list = [env1, env2]

# veh_info_list = []
# for env in env_list:
#     veh_info_list.append(env.vehicle.veh_info)

# timestep = np.max([env.timestep for env in env_list])
# print(timestep)

info_graph(env_list, [env.vehicle.veh_info[:env.timestep+1] for env in env_list], check_finish=True, path=f'simulate_gif/{modelname}{cuda}/infograph_base_{i}.png')

# env.make_gif(path=f'simulate_gif/{modelname}{cuda}/simulate.gif')

env_num = 0
for env in env_list:
    print('-------------------------------------')
    print("env{}".format(env_num))
    env_num += 1
    print("reward coef: ", env.reward_coef)
    print("reward: {}".format((env.vehicle.veh_info[:, 4]).sum()))
    print("execution time: {}".format(np.round(finish-start), 5))
    print("# of episodes: {}".format(env.timestep))
    print("execution time per episode: {}".format((finish-start)/env.timestep))
