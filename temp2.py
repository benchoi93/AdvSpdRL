from gym.spaces import space
import numpy as np
import gym
from gym import spaces

# track_length = 500
# unit_length = 100
# track_length//unit_length+1

# max_speed = 50
# unit_speed = 5

# action_space = spaces.Tuple(([spaces.Discrete(max_speed//unit_speed+1) for i in range(track_length//unit_length+1)]))

# random_action = action_space.sample()

# section_max_speed = tuple(unit_speed*i for i in random_action)

# print(section_max_speed)


# aa = np.zeros((2,4))
# print(aa)
# aa[0] = [1, 2, 3, 4]
# aa[1] = [5, 6, 7, 8]

# print(aa)

# print(aa[:, 0])

# reward_list = [[1,2,4], [1,3,9], [1,4,16]]
# reward_coef = [1,2,3]

# reward = np.array(np.array(reward_list).sum()).dot(np.array(reward_coef))
# reward0 = np.array(np.array(reward_list).sum(0)).dot(np.array(reward_coef))
# reward1 = np.array(np.array(reward_list).sum(1)).dot(np.array(reward_coef))

# print(np.array(np.array(reward_list)).dot(np.array(reward_coef)))
# print("---------------------")
# print(reward)
# print(reward0)
# print(reward1)
# print(np.array(reward_list).sum(0))
# print(np.array(reward_list).sum(1))
# print("----------------")

# i = 0
# print(i)
# print(np.array((1,2,3)).sum())

# # import time
# # len = 2000
# # t1 = time.time()
# # list1 = []

# # for i in range(len):
# #     list1.append([3,3,3,3,3])

# # print("t1: ", (time.time()-t1)*1000)

# # t2 = time.time()
# # list2 = [0]*len

# # for i in range(len):
# #     list2[i] = [3,3,3,3,3]

# # print("t2: ", (time.time()-t2)*1000)

# # t3 = time.time()
# # arr1 = np.zeros((len, 5))

# # for i in range(len):
# #     arr1[i] = [3,3,3,3,3]

# # print("t3: ", (time.time()-t3)*1000)
# print("--------------------")
# arr = np.zeros((10,5))
# for i in range(len(arr)):
#     print(i)


aa = np.zeros((8, 5))

aa[0] = np.array([1,2,3,4,0])
aa[1] = np.array([5,6,7,8,0])
aa[2] = np.array([9,10,11,12,0])
aa[3] = np.array([13,14,15,16,0])
aa[4] = np.array([17,18,19,20,1])
aa[5] = np.array([21,22,23,24,1])
aa[6] = np.array([25,26,27,28,1])
aa[7] = np.array([29,30,31,32,1])

i = 4
import math

print(aa[aa[:,1] < 14])

# import matplotlib.pyplot as plt

# plt.plot([0,1], [0,1])
# plt.show()

aa = []

for i in range(5):
    aa.append([i, [i, i*2, i*3, i*4]])

print(aa)

bb = np.linspace(0,10,101)
print(bb)
print(np.round(bb, 0))
print(np.trunc(bb))