from gym.spaces import space
import numpy as np
import gym
from gym import spaces

track_length = 500
unit_length = 100
track_length//unit_length+1

max_speed = 50
unit_speed = 5

action_space = spaces.Tuple(([spaces.Discrete(max_speed//unit_speed+1) for i in range(track_length//unit_length+1)]))

random_action = action_space.sample()

section_max_speed = tuple(unit_speed*i for i in random_action)

print(section_max_speed)