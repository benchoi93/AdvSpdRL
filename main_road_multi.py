
import gym
import os
from torch import nn
from torch._C import device
# from rl_env.adv_spd_env import AdvSpdEnv
from rl_env.adv_spd_env_road_multi import AdvSpdEnvRoadMulti
import numpy as np

from stable_baselines3 import PPO, SAC, DDPG, A2C, DQN, TD3
from stable_baselines3.common.callbacks import CheckpointCallback
# scenario = np.ones(shape=(1, 200, 40))
# scenario[0, 100, 20:] = 1
import pickle
import argparse

from util.custom_callback_road import GIFCallback
# from rl_env.policy import MlpExtractor_AdvSpdRL


parser = argparse.ArgumentParser()
parser.add_argument('--cuda', default='0', type=str)
parser.add_argument('--model', default='PPO', type=str)
parser.add_argument('--coef-vel', default=1, type=float, help="add penalty to difference between current speed and speed limit")
parser.add_argument('--coef-shock', default=1, type=float, help="add penalty to exceeding speed limit")
parser.add_argument('--coef-jerk', default=0, type=float, help="add penalty to large jerk")
parser.add_argument('--coef-power', default=0, type=float, help="add penalty to large power consumption")
parser.add_argument('--coef-tt', default=0, type=float, help="add penalty to traveltime")
parser.add_argument('--coef-signal', default=100, type=float, help="add penalty to traveltime")
parser.add_argument('--coef-distance', default=0, type=float, help="add penalty to remaining travel distance")
parser.add_argument('--coef-actiongap', default=0, type=float, help="add penalty to gap between calculated action and applied action")
parser.add_argument('--max-episode-steps', default=7500, type=int, help="maximum number of steps in one episode")
parser.add_argument('--activation', default='relu', type=str, choices=['relu', 'tanh'], help="activation function of policy networks")
parser.add_argument('--unit-length', default=25, type=int, help="")
parser.add_argument('--unit-speed', default=5, type=int, help="")
parser.add_argument('--action-dt', default=1, type=int, help="")
parser.add_argument('--stochastic', default=True, type=bool)
parser.add_argument('--entropy', default=0, type=float, help="")
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


env = AdvSpdEnvRoadMulti(num_signal=3,
                         num_action_unit=3,
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

model = args.model
checkpoint_callback = CheckpointCallback(save_freq=100000, save_path=f"./params/{model}{int(cuda)}", name_prefix=f"AdvSpdRL_{model}")
# gif_callback = GIFCallback(env=env, save_freq=100000, save_path=os.path.join('simulate_gif', f'{model}{int(cuda)}'), name_prefix=f"AdvSpdRL_{model}")


directory = f'params/{model}{int(cuda)}'
if not os.path.exists(directory):
    os.mkdir(directory)

pickle.dump(env, open(f"params/{model}{int(cuda)}/env.pkl", 'wb'))

model = globals()[model]("MlpPolicy",
                         env,
                         verbose=1,
                         tensorboard_log=f"log/{model}/",
                         device='cuda',
                         policy_kwargs={"activation_fn": activation_fn},
                         ent_coef=args.entropy
                         )
# model.save("params/AdvSpdRL")
model.learn(total_timesteps=1000000000, callback=[checkpoint_callback])
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
