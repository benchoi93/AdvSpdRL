from pathlib import Path
from util.plotutil import info_graph, info_graph_separate, make_gif, info_graph_detail
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3 import PPO, SAC, DDPG, A2C, DQN, TD3
import numpy as np
from PIL import Image
from rl_env.adv_spd_env_road_multi_fromsrc import AdvSpdEnvRoadMulti_SRC
import time
import matplotlib.pyplot as plt
import pickle
import glob
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

outpath = 'simulate_gif_fromsrc'
modelname = 'PPO'
cuda = '1'
i = 0

# coef_power = 0.01
# param = 'AdvSpdRL_DDPG_3500000_steps'
#     cuda = str(cuda)
#     for i in range(100):
# try:
# env: AdvSpdEnvRoadMulti_SRC = pickle.load(open(os.path.join('params', f'{modelname}{cuda}', 'env.pkl'), 'rb'))
env = AdvSpdEnvRoadMulti_SRC(src="rl_env/data/brt1001_signal_offset.xlsx", timelimit=20000,
                             reward_coef=[1, 1, 1, 1, 0, 0, 0, 0],
                             unit_length=25,
                             unit_speed=5)

model = globals()[modelname]("MlpPolicy", env, verbose=1, device='cpu')
# list_of_files = glob.glob(os.path.join('params', f'{modelname}{cuda}/best_model.zip'))
# latest_file = max(list_of_files, key=os.path.getmtime)
# model = model.load(latest_file, device='cpu')

path = Path(f'{outpath}/{modelname}{cuda}/')
if not path.exists():
    path.mkdir(parents=True)

ob = env.reset()
pickle.dump(env, open(f'{outpath}/{modelname}{cuda}/env_{i}.pkl', 'wb'))

episode_over = False
combine = True

cnt = 0
print("-------------------------------------")
while not episode_over:
    action = env.get_random_action()
    # print(f"{cnt=} || {action=}")
    cnt += 1
    ob, reward, episode_over, info = env.step(action)
env1 = env
print(env.timestep/10)


env = pickle.load(open(f'{outpath}/{modelname}{cuda}/env_{i}.pkl', 'rb'))

episode_over = False
combine = True

cnt = 0
print("-------------------------------------")

while not episode_over:
    action = np.array([9])
    # print(f"{cnt=} || {action=}")
    cnt += 1
    ob, reward, episode_over, info = env.step(action)
env2 = env
print(env.timestep/10)

print("-------------------------------------")

# env_list = [env1]
env_list = [env1, env2]


info_graph(env_list, [env.vehicle.veh_info[:env.timestep+1] for env in env_list],
           check_finish=True, path=f'simulate_gif_fromsrc/{modelname}{cuda}/infograph_base_{i}.png')
# info_graph_separate(env_list, [env.vehicle.veh_info[:env.timestep+1] for env in env_list], path=f'simulate_gif/{modelname}{cuda}/infograph_separate_{i}.png')
info_graph_detail(env_list, [env.vehicle.veh_info[:env.timestep+1] for env in env_list], True,
                  path=f'simulate_gif_fromsrc/{modelname}{cuda}/infograph_detail_separate_{i}.png')
info_graph_detail(env_list, [env.vehicle.veh_info[:env.timestep+1] for env in env_list], False,
                  path=f'simulate_gif_fromsrc/{modelname}{cuda}/infograph_detail_nonseparate_{i}.png')
# render_gif = True
# if render_gif == True:
#     env_num = 0
#     for env in env_list:
#         print("@ env{}/ Making Car-moving gif".format(env_num))
#         make_gif(env, env_num)
#         env_num += 1
# print("-------------------------------------")

# env_num = 0
# for env in env_list:
#     print("@ env{}/ Test information".format(env_num))
#     env_num += 1
#     print("reward coef: ", env.reward_coef)
#     print("reward: {}".format(np.round((env.vehicle.veh_info[:, 4]).sum(), 3)))
#     # print("execution time: {}".format(np.round(finish-start), 5))
#     print("# of episodes: {}".format(env.timestep))
#     # print("execution time per episode: {}".format((finish-start)/env.timestep))
#     print('-------------------------------------')
