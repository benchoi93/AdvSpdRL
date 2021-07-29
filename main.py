
from rl_env.adv_spd_env import AdvSpdEnv

import numpy as np

# scenario = np.ones(shape=(1, 200, 40))
# scenario[0, 100, 20:] = 1

env = AdvSpdEnv()

print(env.reset())
episode_over = False
while not episode_over:
    action = env.get_random_action()
    ob, reward, episode_over, info = env.step(action)
    env.render()
