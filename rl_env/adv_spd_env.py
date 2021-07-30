import gym
import numpy as np
from gym import spaces


class Vehicle(object):
    def __init__(self):
        max_speed = 50 / 3.6
        self.position = 0
        self.velocity = np.random.rand() * max_speed
        self.acceleration = 0


class TrafficSignal(object):
    def __init__(self):
        min_location = 250
        max_location = 450

        self.phase_length = 60
        self.cycle_length = 120

        self.location = min_location + np.random.rand() * (max_location - min_location)
        self.timetable = np.zeros(shape=[self.cycle_length])
        offset = np.random.randint(0, 10)

        running_offset = offset
        cursignal = True
        while running_offset + self.phase_length < self.cycle_length:
            self.timetable[running_offset:(running_offset+self.phase_length)] = cursignal
            cursignal = not cursignal
            running_offset += self.phase_length
        pass

    def get_greentime(self, timestep):
        greentime_start = 0
        greentime_end = 0

        while greentime_end == 0:
            if greentime_start == 0:
                if self.timetable[timestep % self.cycle_length] == 1:
                    greentime_start = timestep
            else:
                if self.timetable[timestep % self.cycle_length] == 0:
                    greentime_end = timestep
            timestep += 1

        return greentime_start, greentime_end


class AdvSpdEnv(gym.Env):
    def __init__(self):

        # num_observations = 2
        self.dt = 0.1
        self.track_length = 500.0

        self.action_space = spaces.Box(low=-5.0, high=5.0, shape=[1, ])

        min_states = np.array([0.0,  # position
                               0.0,  # velocity
                               0.0,  # signal location
                               0.0,  # green time start
                               0.0  # green time end
                               ])
        max_states = np.array([self.track_length,  # position
                               50.0/3.6,  # velocity
                               self.track_length,   # signal location
                               40.0,   # green time start
                               40.0   # green time end
                               ])

        self.observation_space = spaces.Box(low=min_states,
                                            high=max_states)

        # self.scenario = scenario
        self.viewer = None
        self.reset()
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
        prev_ob = self.state

        self._take_action(action.item())
        self.timestep += 1

        ob = self._get_state()

        if prev_ob[0] <= self.signal.location:
            if ob[0] > self.signal.location:
                if not self.signal.timetable[int(self.timestep * self.dt) % self.signal.timetable.shape[0]]:
                    self.violation = True

        reward = self._get_reward()

        episode_over = self.vehicle.position > self.track_length

        return ob, reward, episode_over, {}

    def _get_state(self):
        self.state = [self.vehicle.position,
                      self.vehicle.velocity,
                      self.signal.location,
                      self.signal.get_greentime(self.timestep)[0],
                      self.signal.get_greentime(self.timestep)[1]]
        return self.state

    def reset(self):

        self.vehicle = Vehicle()
        self.signal = TrafficSignal()

        self.timestep = 0
        self.violation = False

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
        screen_width = 1000
        screen_height = 450

        clearance_x = 80
        clearance_y = 10

        zero_x = 0.25 * screen_width
        visible_track_length = 500
        scale_x = screen_width / visible_track_length

        if self.viewer is None:
            import matplotlib.pyplot as plt
            import seaborn as sns
            from util import rendering
            self.viewer = rendering.Viewer(width=screen_width,
                                           height=screen_height)

            import os
            rel_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'util', 'assets', 'track', 'img')
            fname = os.path.join(rel_dir, 'start_finish_30x100.png')
            start = rendering.Image(fname,
                                    rel_anchor_y=0,
                                    batch=self.viewer.batch,
                                    group=self.viewer.background)
            start.position = (zero_x, clearance_y)
            self.viewer.components['start'] = start

            finish = rendering.Image(fname,
                                     rel_anchor_y=0,
                                     batch=self.viewer.batch,
                                     group=self.viewer.background)
            finish.position = (zero_x + scale_x * 500,
                               clearance_y)
            self.viewer.components['finish'] = finish
            fname = os.path.join(rel_dir, 'car_80x40.png')
            car = rendering.Image(fname,
                                  rel_anchor_x=1,
                                  batch=self.viewer.batch,
                                  group=self.viewer.foreground)
            car.position = (zero_x, 50 + clearance_y)
            self.viewer.components['car'] = car

            # signal
            fname = os.path.join(rel_dir, 'sign_60x94.png')
            signal = rendering.Image(fname,
                                     rel_anchor_x=1,
                                     batch=self.viewer.batch,
                                     group=self.viewer.foreground)
            signal.position = (zero_x + scale_x * (self.signal.location - self.vehicle.position),
                               100 + clearance_y)
            self.viewer.components['signal'] = signal

            # info figures
            self.viewer.history['velocity'] = []
            self.viewer.history['speed_limit'] = []
            self.viewer.history['position'] = []
            self.viewer.history['acceleration'] = []
            sns.set_style('whitegrid')
            self.fig = plt.Figure((960 / 80, 200 / 80))
            info = rendering.Figure(self.fig,
                                    rel_anchor_x=0,
                                    rel_anchor_y=0,
                                    batch=self.viewer.batch,
                                    group=self.viewer.background)
            info.position = (clearance_x - 40, 225 + clearance_y)
            self.viewer.components['info'] = info

        if self.signal.timetable[self.timestep % self.signal.timetable.shape[0]]:
            self.viewer.components['signal'].color = (0, 255, 0)
        else:
            self.viewer.components['signal'].color = (255, 0, 0)
        self.viewer.components['signal'].position = (zero_x + scale_x * (self.signal.location - self.vehicle.position), 100 + clearance_y)

        # info figures
        self.viewer.history['velocity'].append(self.vehicle.velocity * 3.6)
        # self.viewer.history['speed_limit'].append(self.current_speed_limit *3.6)
        self.viewer.history['position'].append(self.vehicle.position)
        self.viewer.history['acceleration'].append(self.vehicle.acceleration)
        # self.viewer.components['info'].visible = False

        self.viewer.components['info'].visible = True
        self.fig.clf()
        ax = self.fig.add_subplot(131)
        ax.plot(self.viewer.history['position'],
                self.viewer.history['velocity'],
                lw=2,
                color='k')
        # ax.plot(self.viewer.history['position'],
        #         self.viewer.history['speed_limit'],
        #         lw=1.5,
        #         ls='--',
        #         color='r')
        ax.set_xlabel('Position in m')
        ax.set_ylabel('Velocity in km/h')
        ax.set_xlim(
            (0.0, max(500, self.vehicle.position + (500 - self.vehicle.position) % 500)))
        ax.set_ylim((0.0, 130))

        ax2 = self.fig.add_subplot(132)
        ax2.plot(self.viewer.history['position'],
                 self.viewer.history['acceleration'],
                 lw=2,
                 color='k')
        ax2.set_xlabel('Position in m')
        ax2.set_ylabel('Acceleration in m/s²')
        ax2.set_xlim(
            (0.0, max(500, self.vehicle.position + (500 - self.vehicle.position) % 500)))
        ax2.set_ylim((-5.0, 5.0))

        ax3 = self.fig.add_subplot(133)
        ax3.plot(self.viewer.history['position'],
                 self.viewer.history['acceleration'],
                 lw=2,
                 color='k')
        ax3.set_xlabel('Time in s')
        ax3.set_ylabel('Position in m')
        ax3.set_xlim(
            (0.0, max(500, self.vehicle.position + (500 - self.vehicle.position) % 500)))
        ax3.set_ylim((-5.0, 5.0))

        self.fig.tight_layout()
        self.viewer.components['info'].figure = self.fig
        # updates
        self.viewer.components['start'].position = (zero_x + scale_x *
                                                    (0 - self.vehicle.position),
                                                    clearance_y)
        self.viewer.components['finish'].position = (
            zero_x + scale_x * (self.track_length - self.vehicle.position),
            clearance_y)
        mode = None
        # self.viewer.activate()
        return self.viewer.render(return_rgb_array=mode == 'rgb_array')
        # import time
        # input()
        # self.viewer.close()

        # pass
        # print(f"position = {self.vehicle.position:.3f} || speed = {self.vehicle.velocity:.3f}")

    def _take_action(self, action):
        self.vehicle.acceleration = action
        self.vehicle.velocity = self.vehicle.velocity + action * self.dt
        self.vehicle.position = self.vehicle.position + self.vehicle.velocity * self.dt

    def _get_reward(self):
        reward_norm_velocity = (self.vehicle.velocity) / (50/3.6)
        # penalty_travel_time = -1
        penalty_signal_violation = -100 if self.violation else 0

        return -reward_norm_velocity - penalty_signal_violation
        # return reward_norm_velocity
