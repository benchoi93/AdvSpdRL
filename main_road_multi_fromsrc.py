
import gym
import os
from stable_baselines3.common.monitor import Monitor
from torch import nn
from torch._C import device
# from rl_env.adv_spd_env import AdvSpdEnv
from rl_env.adv_spd_env_road_multi_fromsrc import AdvSpdEnvRoadMulti_SRC
import numpy as np

from stable_baselines3 import PPO, SAC, DDPG, A2C, DQN, TD3
from stable_baselines3.common.callbacks import CheckpointCallback
# scenario = np.ones(shape=(1, 200, 40))
# scenario[0, 100, 20:] = 1
import pickle
import argparse

from util.custom_callback_road import SaveOnBestTrainingRewardCallback
# from rl_env.policy import MlpExtractor_AdvSpdRL


parser = argparse.ArgumentParser()
parser.add_argument('--cuda', default='0', type=str)
parser.add_argument('--model', default='PPO', type=str)
parser.add_argument('--coef-vel', default=1, type=float, help="add penalty to difference between current speed and speed limit")
parser.add_argument('--coef-shock', default=1, type=float, help="add penalty to exceeding speed limit")
parser.add_argument('--coef-jerk', default=1, type=float, help="add penalty to large jerk")
parser.add_argument('--coef-power', default=1, type=float, help="add penalty to large power consumption")
parser.add_argument('--coef-tt', default=0, type=float, help="add penalty to traveltime")
parser.add_argument('--coef-signal', default=100, type=float, help="add penalty to traveltime")
parser.add_argument('--coef-distance', default=0, type=float, help="add penalty to remaining travel distance")
parser.add_argument('--coef-actiongap', default=1, type=float, help="add penalty to gap between calculated action and applied action")
parser.add_argument('--max-episode-steps', default=20000, type=int, help="maximum number of steps in one episode")
parser.add_argument('--activation', default='relu', type=str, choices=['relu', 'tanh'], help="activation function of policy networks")
parser.add_argument('--unit-length', default=25, type=int, help="")
parser.add_argument('--unit-speed', default=5, type=int, help="")
parser.add_argument('--action-dt', default=1, type=int, help="")
parser.add_argument('--stochastic', default=False, type=bool)
parser.add_argument('--entropy', default=1e-1, type=float, help="")
args = parser.parse_args()

print(args)
cuda = args.cuda
os.environ["CUDA_VISIBLE_DEVICES"] = cuda

if args.activation == 'relu':
    activation_fn = nn.ReLU
elif args.activation == 'tanh':
    activation_fn = nn.Tanh
else:
    print("wrong input")

# env = AdvSpdEnvRoadMulti_SRC(src="rl_env/data/brt1001_signal_offset.xlsx", timelimit=20000,
#                              reward_coef=[1, 1, 1, 1, 0, 0, 0, 0],
#                              unit_length=25,
#                              unit_speed=5)

env = AdvSpdEnvRoadMulti_SRC(src="rl_env/data/brt1001_signal_offset.csv",
                             reward_coef=[args.coef_vel,
                                          args.coef_shock,
                                          args.coef_jerk,
                                          args.coef_power,
                                          args.coef_tt,
                                          args.coef_signal,
                                          args.coef_distance,
                                          args.coef_actiongap],  # coef_signal_violation
                             timelimit=args.max_episode_steps,
                             unit_length=args.unit_length,
                             unit_speed=args.unit_speed,
                             action_dt=args.action_dt,
                             stochastic=args.stochastic
                             )

env = gym.wrappers.TimeLimit(env, max_episode_steps=args.max_episode_steps)

modelname = args.model
directory = f'params/{modelname}{int(cuda)}'
if not os.path.exists(directory):
    os.mkdir(directory)

env = Monitor(env, f"./params/{modelname}{int(cuda)}/")
checkpoint_callback = CheckpointCallback(save_freq=10000, save_path=f"./params/{modelname}{int(cuda)}", name_prefix=f"AdvSpdRL_{modelname}")
best_callback = SaveOnBestTrainingRewardCallback(check_freq=10000, log_dir=f"./params/{modelname}{int(cuda)}")
# gif_callback = GIFCallback(env=env, save_freq=100000, save_path=os.path.join('simulate_gif', f'{model}{int(cuda)}'), name_prefix=f"AdvSpdRL_{model}")


pickle.dump(env.env, open(f"params/{modelname}{int(cuda)}/env.pkl", 'wb'))

model = PPO("MlpPolicy",
            env,
            verbose=1,
            tensorboard_log=f"log/{modelname}/",
            device='cuda',
            policy_kwargs={"activation_fn": activation_fn},
            ent_coef=args.entropy,
            n_steps=10240,
            batch_size=256
            )
# model.save("params/AdvSpdRL")
model.learn(total_timesteps=1000000000, callback=[checkpoint_callback, best_callback])
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
