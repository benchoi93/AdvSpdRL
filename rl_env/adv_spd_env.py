import gym
import numpy as np
from gym import spaces


class Vehicle(object):
    def __init__(self):
        max_speed = 50 / 3.6
        self.position = 0
        self.velocity = np.random.rand() * max_speed


class TrafficSignal(object):
    def __init__(self):
        min_location = 50
        max_location = 450

        phase_length = 10

        self.location = min_location + np.random.rand() * (max_location - min_location)
        self.timetable = np.zeros(shape=[40])
        offset = np.random.randint(0, 10)

        running_offset = offset
        cursignal = True
        while running_offset + phase_length < 40:
            self.timetable[running_offset:(running_offset+phase_length)] = cursignal
            cursignal = not cursignal
            running_offset += phase_length

    def get_greentime(self, timestep):
        timestep = int(3.7)

        greentime_start = 0
        greentime_end = 0

        for i in range(timestep, self.timetable.shape[0]):
            if greentime_start == 0:
                if self.timetable[i] == 1:
                    greentime_start = i
            else:
                if self.timetable[i] == 0:
                    greentime_end = i
                    break
        return greentime_start, greentime_end


class AdvSpdEnv(gym.Env):
    def __init__(self):

        # num_observations = 2
        self.dt = 0.1

        self.action_space = spaces.Box(low=-5.0, high=5.0, shape=[1, ])

        min_states = np.array([0.0,  # position
                               0.0,  # velocity
                               0.0,  # signal location
                               0.0,  # green time start
                               0.0  # green time end
                               ])
        max_states = np.array([500.0,  # position
                               50.0/3.6,  # velocity
                               500.0,   # signal location
                               40.0,   # green time start
                               40.0   # green time end
                               ])

        self.observation_space = spaces.Box(low=min_states,
                                            high=max_states)

        # self.scenario = scenario

        pass

    def get_random_action(self):
        return -5.0 + (np.random.rand()) * 10

    def step(self, action):
        """

        Parameters
        ----------
        action :

        Returns
        -------
        ob, reward, episode_over, info : tuple
            ob (object) :
                an environment-specific object representing your observation of
                the environment.
            reward (float) :
                amount of reward achieved by the previous action. The scale
                varies between environments, but the goal is always to increase
                your total reward.
            episode_over (bool) :
                whether it's time to reset the environment again. Most (but not
                all) tasks are divided up into well-defined episodes, and done
                being True indicates the episode has terminated. (For example,
                perhaps the pole tipped too far, or you lost your last life.)
            info (dict) :
                 diagnostic information useful for debugging. It can sometimes
                 be useful for learning (for example, it might contain the raw
                 probabilities behind the environment's last state change).
                 However, official evaluations of your agent are not allowed to
                 use this for learning.
        """
        self._take_action(action)
        self.timestep += 1

        reward = self._get_reward()
        ob = self._get_state()

        episode_over = self.vehicle.position > 500

        return ob, reward, episode_over, {}

    def _get_state(self):
        state = [self.vehicle.position,
                 self.vehicle.velocity,
                 self.signal.location,
                 self.signal.get_greentime(self.timestep)[0],
                 self.signal.get_greentime(self.timestep)[1]]
        return state

    def reset(self):

        self.vehicle = Vehicle()
        self.signal = TrafficSignal()

        self.timestep = 0

        self.state = [self.vehicle.position,
                      self.vehicle.velocity,
                      self.signal.location,
                      self.signal.get_greentime(self.timestep)[0],
                      self.signal.get_greentime(self.timestep)[1]]

        self.reward = 0
        self.done = False
        self.info = {}

        return self.state

    def render(self, mode='human', close=False):
        # pass
        print(f"position = {self.vehicle.position:.3f} || speed = {self.vehicle.velocity:.3f}")

    def _take_action(self, action):
        self.vehicle.velocity = self.vehicle.velocity + action * self.dt
        self.vehicle.position = self.vehicle.position + self.vehicle.velocity * self.dt

    def _get_reward(self):
        return (self.vehicle.velocity) / (50/3.6) - 1
