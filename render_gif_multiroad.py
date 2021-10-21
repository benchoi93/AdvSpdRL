import pandas as pd
import pickle
from rl_env.adv_spd_env_road_multi import AdvSpdEnvRoadMulti
from util.plotutil import car_moving

import numpy as np
from tqdm import tqdm
from itertools import chain
from stable_baselines3 import PPO, SAC, DDPG, A2C, DQN, TD3
import glob
import os
from util.plotutil import info_graph, info_graph_separate, info_graph_detail, make_gif
from pathlib import Path

gif_targets = [{"cuda": 0, "scn": [28, 50, 67, 71, 82]},
               {"cuda": 1, "scn": [27, 58]},
               {"cuda": 2, "scn": [19, 44, 77]},
               {"cuda": 3, "scn": [65, 78]}
               ]
outpath = 'gifout'
modelname = 'PPO'
compare_out = []
for cudascn in gif_targets:
    cuda = cudascn['cuda']
    for i in cudascn['scn']:

        # for cuda in [0, 1, 2, 3]:
        #     for i in range(100):

        # modelname = "PPO"
        # cuda = 2
        # i = 2
        path = "simulate_gif"
        outpathcheck = Path(f'{outpath}/{modelname}{cuda}/')
        if not outpathcheck.exists():
            outpathcheck.mkdir()

        env: AdvSpdEnvRoadMulti = pickle.load(open(f"{path}/PPO{cuda}/env_{i}.pkl", "rb"))

        model = PPO("MlpPolicy", env, verbose=1, device='cpu')
        list_of_files = glob.glob(os.path.join('params', f'{modelname}{cuda}/*.zip'))
        latest_file = max(list_of_files, key=os.path.getmtime)
        model = model.load(latest_file, device='cpu')

        ob = env.get_state()
        # pickle.dump(env, open(f'{path}/{modelname}{cuda}/env_{i}.pkl', 'wb'))

        episode_over = False
        combine = True

        cnt = 0
        print("-------------------------------------")
        while not episode_over:
            action, _ = model.predict(ob)
            print(f"{cnt=} || {action=}")
            cnt += 1
            ob, reward, episode_over, info = env.step(action)
        env1 = env
        print(env.timestep/10)

        env = pickle.load(open(f'{path}/{modelname}{cuda}/env_{i}.pkl', 'rb'))

        episode_over = False
        combine = True

        cnt = 0
        print("-------------------------------------")

        while not episode_over:
            action = np.array([9])
            print(f"{cnt=} || {action=}")
            cnt += 1
            ob, reward, episode_over, info = env.step(action)
        env2 = env
        print(env.timestep/10)

        print("-------------------------------------")

        # env_list = [env1]
        env_list = [env1, env2]

        info_graph_detail(env_list, [env.vehicle.veh_info[:env.timestep+1] for env in env_list],
                          True, path=f'{outpath}/{modelname}{cuda}/infograph2_separate{i}.png')
        info_graph_detail(env_list, [env.vehicle.veh_info[:env.timestep+1] for env in env_list],
                          False, path=f'{outpath}/{modelname}{cuda}/infograph2_nonseparate{i}.png')

        for j in range(len(env_list)):
            env = env_list[j]
            env.png_list = []
            start = car_moving(env, env.vehicle.veh_info[:1], startorfinish=True, combine=True)
            mid = [car_moving(env, env.vehicle.veh_info[:i+2], startorfinish=False, combine=True) for i in tqdm(range(env.timestep))]
            end = car_moving(env, env.vehicle.veh_info[:env.timestep+1], startorfinish=True, combine=True)

            env.png_list = start + list(chain(*mid)) + end
            env.png_list[0].save(f"{outpath}/{modelname}{cuda}/scn_{i}_carmoving_{j}.gif", save_all=True,
                                 append_images=env.png_list[1:], optimize=False, duration=30, loop=1)

        env1_reward_per_unit = pd.DataFrame(np.array([x[1] for x in env1.reward_per_unitlen]))
        env2_reward_per_unit = pd.DataFrame(np.array([x[1] for x in env2.reward_per_unitlen]))
# [-reward_norm_velocity, -reward_shock, -reward_jerk, -reward_power, -penalty_travel_time, -# penalty_signal_violation, -reward_remaining_distance, -reward_action_gap]
        rewards_columns = ["norm_velocity", "shock", "jerk", "power", "tt", "sig_violation", "remaining_distance", "actiongap", "total"]
        env1_reward_per_unit.columns = rewards_columns
        env2_reward_per_unit.columns = rewards_columns

        compare = env1_reward_per_unit.sum(0) / env2_reward_per_unit.sum(0)
        compare['cuda'] = cuda
        compare['i'] = i
        compare_out.append(compare)

        del env1
        del env2


pd.DataFrame(compare_out).to_csv(f"{outpath}/compare.csv")
