import gym
import numpy as np
from gym import spaces


class Vehicle(object):
    def __init__(self):
        max_speed = 50 / 3.6
        self.position = 0
        # self.velocity = np.random.rand() * max_speed
        self.velocity = 40/3.6
        self.acceleration = 0
        self.jerk = 0
        self.action_limit_penalty = 1


class TrafficSignal(object):
    def __init__(self, min_location=250, max_location=350):

        self.phase_length = {True: 30, False: 90}
        self.cycle_length = sum(self.phase_length.values())

        self.location = 300
        # self.location = min_location + np.random.rand() * (max_location - min_location)
        self.timetable = np.ones(shape=[self.cycle_length]) * -1

        offset = 40
        # offset = np.random.randint(0, self.cycle_length)

        for i in range(self.cycle_length):
            cur_idx = (i+offset) % self.cycle_length
            self.timetable[cur_idx] = 1 if i < self.phase_length[True] else 0
            # print(cur_idx)

        assert(-1 not in self.timetable)
        assert(sum(self.timetable) == self.phase_length[True])
        # self.timetable[offset:(offset+self.phase_length[True])] = 1
        # self.timetable = self.timetable[offset:(offset + self.cycle_length)]

        # running_offset = offset
        # cursignal = True
        # while running_offset < self.cycle_length:
        #     self.timetable[running_offset:(running_offset+self.phase_length[cursignal])] = cursignal
        #     running_offset += self.phase_length[cursignal]
        #     cursignal = not cursignal

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

    def is_green(self, timestep):
        return timestep % self.cycle_length >= self.get_greentime(timestep % self.cycle_length)[0]


class AdvSpdEnv(gym.Env):
    def __init__(self, dt=0.1, track_length=500.0, acc_max=5, acc_min=-5, speed_max=50.0/3.6, dec_th=-3, stop_th=2, reward_coef=[1, 10, 1, 0.01, 0, 1, 1], timelimit=2400):

        # num_observations = 2
        self.dt = dt
        self.track_length = track_length
        self.acc_min = acc_min
        self.acc_max = acc_max
        self.speed_max = speed_max
        self.dec_th = dec_th
        self.stop_th = stop_th
        self.reward_coef = reward_coef
        # [-reward_norm_velocity, -reward_shock, -reward_jerk, -reward_power, -penalty_travel_time]
        self.timelimit = timelimit

        self.action_space = spaces.Box(low=acc_min, high=acc_max, shape=[1, ])
        self.reset()

        min_states = np.array([0.0,  # position
                               0.0,  # velocity
                               0.0,  # signal location
                               0.0,  # green time start
                               0.0  # green time end
                               ])
        max_states = np.array([self.track_length*2,  # position
                               speed_max,  # velocity
                               self.track_length*2,   # signal location
                               self.timelimit*2,   # green time start
                               self.timelimit*2  # green time end
                               ])

        self.observation_space = spaces.Box(low=min_states,
                                            high=max_states)

        # self.scenario = scenario
        self.viewer = None
        pass

    def save(self):
        pass

    def load(self):
        pass

    def get_random_action(self):
        return self.acc_min + (np.random.rand()) * (self.acc_max - self.acc_min)

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
                if not self.signal.is_green(int(self.timestep * self.dt)):
                    self.violation = True

        reward = np.array(self._get_reward()).dot(np.array(self.reward_coef))

        episode_over = self.vehicle.position > self.track_length

        return ob, reward, episode_over, {}

    def _get_state(self):
        # if self.vehicle.position < self.signal.location:
        self.state = [self.vehicle.position,
                      self.vehicle.velocity,
                      self.signal.location,
                      self.signal.get_greentime(int(self.timestep*self.dt))[0] / self.dt,
                      self.signal.get_greentime(int(self.timestep*self.dt))[1] / self.dt]
        # else:
        #     self.state = [self.vehicle.position,
        #                   self.vehicle.velocity,
        #                   self.track_length*2,
        #                   self.timelimit*2,
        #                   self.timelimit*2]
        return self.state

    def reset(self):

        self.vehicle = Vehicle()
        self.signal = TrafficSignal()

        self.timestep = 0
        self.violation = False

        self.state = [self.vehicle.position,
                      self.vehicle.velocity,
                      self.signal.location,
                      self.signal.get_greentime(int(self.timestep*self.dt))[0],
                      self.signal.get_greentime(int(self.timestep*self.dt))[1]]

        self.reward = 0
        self.done = False
        self.info = {}

        return self.state

    def render(self, mode='human', info_show=False, close=False):
        screen_width = 1200
        screen_height = 450

        clearance_x = 80
        clearance_y = 10

        zero_x = 0.25 * screen_width
        visible_track_length = 500
        scale_x = screen_width / visible_track_length
        draw_signal_location = False

        if self.viewer is None:
            draw_signal_location = True
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
            self.viewer.history['reward'] = []
            self.viewer.history['reward_power'] = []
            sns.set_style('whitegrid')
            self.fig = plt.Figure((900 / 80, 200 / 80), dpi=80)
            info = rendering.Figure(self.fig,
                                    rel_anchor_x=0,
                                    rel_anchor_y=0,
                                    batch=self.viewer.batch,
                                    group=self.viewer.background)
            info.position = (clearance_x - 40, 225 + clearance_y)
            self.viewer.components['info'] = info

        if self.signal.is_green(int(self.timestep * self.dt)):
            self.viewer.components['signal'].color = (0, 255, 0)
        else:
            self.viewer.components['signal'].color = (255, 0, 0)
        self.viewer.components['signal'].position = (zero_x + scale_x * (self.signal.location - self.vehicle.position+1), 100 + clearance_y)

        # update history
        self.viewer.history['velocity'].append(self.vehicle.velocity * 3.6)
        # self.viewer.history['speed_limit'].append(self.current_speed_limit *3.6)
        self.viewer.history['position'].append(self.vehicle.position)
        self.viewer.history['acceleration'].append(self.vehicle.acceleration)
        self.viewer.history['reward'].append(np.array(self._get_reward()).dot(np.array(self.reward_coef)))
        self.viewer.history['reward_power'].append(self._get_reward()[3])

        if info_show:
            # info figures

            # self.viewer.components['info'].visible = False

            self.viewer.components['info'].visible = True
            self.fig.clf()
            ax = self.fig.add_subplot(151)
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

            ax2 = self.fig.add_subplot(152)
            ax2.plot(self.viewer.history['position'],
                     self.viewer.history['acceleration'],
                     lw=2,
                     color='k')
            ax2.set_xlabel('Position in m')
            ax2.set_ylabel('Acceleration in m/s²')
            ax2.set_xlim(
                (0.0, max(500, self.vehicle.position + (500 - self.vehicle.position) % 500)))
            ax2.set_ylim((self.acc_min, self.acc_max))

            ax3 = self.fig.add_subplot(153)
            ax3.plot([x*self.dt for x in range(len(self.viewer.history['position']))],
                     self.viewer.history['position'],
                     lw=2,
                     color='k')
            # if draw_signal_location:
            xlim_max = self.timelimit
            for i in range(xlim_max):
                if self.signal.is_green(i):
                    signal_color = 'g'
                else:
                    signal_color = 'r'

                ax3.plot([x*self.dt for x in range(int((i)/self.dt), int((i + 1)//self.dt)+1)],
                         [self.signal.location] * len(range(int(i/self.dt), int((i + 1)//self.dt)+1)),
                         color=signal_color
                         )
            green_search_xlim = int(max(100, len(self.viewer.history['position'])) * self.dt) + 1

            ax3.set_xlabel('Time in s')
            ax3.set_ylabel('Position in m')
            ax3.set_xlim((0.0, green_search_xlim))
            ax3.set_ylim((0, self.track_length))

            ax4 = self.fig.add_subplot(154)
            ax4.plot([x*self.dt for x in range(len(self.viewer.history['reward']))],
                     self.viewer.history['reward'],
                     lw=2,
                     color='k')
            xlim_max = int(max(100, len(self.viewer.history['position'])) * self.dt) + 1

            ax4.set_xlabel('Time in s')
            ax4.set_ylabel('reward')
            ax4.set_xlim((0.0, xlim_max))
            ax4.set_ylim((-5.0, 5.0))

            ax4 = self.fig.add_subplot(155)
            ax4.plot([x*self.dt for x in range(len(self.viewer.history['reward_power']))],
                     self.viewer.history['reward_power'],
                     lw=2,
                     color='k')
            xlim_max = int(max(100, len(self.viewer.history['position'])) * self.dt) + 1

            ax4.set_xlabel('Time in s')
            ax4.set_ylabel('reward')
            ax4.set_xlim((0.0, xlim_max))
            ax4.set_ylim((-1.0, 1.0))

        if self.violation:
            self.viewer.components['car'].color = (255, 0, 0)
        else:
            self.viewer.components['car'].color = (255, 255, 255)

        # self.fig.tight_layout()
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
        applied_action = action

        max_acc = self.calculate_max_acceleration()
        if max_acc < action:
            applied_action = max_acc

        if self.vehicle.velocity + applied_action * self.dt < 0:
            applied_action = - self.vehicle.velocity / self.dt

        self.vehicle.jerk = (applied_action - self.vehicle.acceleration) / self.dt
        self.vehicle.acceleration = applied_action

        assert(np.round(self.vehicle.velocity + applied_action * self.dt, -5) >= 0)
        self.vehicle.velocity = self.vehicle.velocity + applied_action * self.dt
        self.vehicle.position = self.vehicle.position + self.vehicle.velocity * self.dt
        # self.vehicle.action_limit_penalty = abs(applied_action / action)

    def _get_reward(self):
        max_speed = self.speed_max
        reward_norm_velocity = np.abs((self.vehicle.velocity) - max_speed)
        reward_norm_velocity /= max_speed

        reward_jerk = self.vehicle.jerk
        jerk_max = (self.acc_max - self.acc_min) / self.dt
        reward_jerk /= jerk_max

        reward_shock = 1 if self.vehicle.velocity > max_speed else 0
        penalty_signal_violation = 1 if self.violation else 0
        # penalty_action_limit = self.vehicle.action_limit_penalty if self.vehicle.action_limit_penalty != 1 else 0
        # penalty_moving_backward = 1000 if self.vehicle.velocity < 0 else 0
        penalty_travel_time = 1

        reward_remaining_distance = (self.track_length - self.vehicle.position) / self.track_length

        # reward_finishing = 1000 if self.vehicle.position > 490 else 0
        # reward_power = self.energy_consumption() * self.dt / 75 * 0.5

        power = -self.energy_consumption()
        reward_power = self.vehicle.velocity / power if (power > 0 and self.vehicle.velocity >= 0) else 0

        return [-reward_norm_velocity, -reward_shock, -reward_jerk, -reward_power, -penalty_travel_time, -penalty_signal_violation, -reward_remaining_distance]
        # return -reward_norm_velocity - \
        #     reward_shock - \
        #     reward_jerk - \
        #     reward_power - \
        #     penalty_travel_time
        # return reward_norm_velocity

    def calculate_max_acceleration(self):

        dec_th = self.dec_th
        max_acc = self.acc_max
        if not self.signal.is_green(int(self.timestep * self.dt)):

            if self.vehicle.position < self.signal.location:
                mild_stopping_distance = -(self.vehicle.velocity + self.acc_max * self.dt) ** 2 / (2 * (dec_th))
                distance_to_signal = self.signal.location - self.stop_th - self.vehicle.position

                if distance_to_signal < mild_stopping_distance:
                    max_acc = -(self.vehicle.velocity + self.acc_max * self.dt) ** 2 / (2 * (distance_to_signal))

        return max_acc

    def energy_consumption(self, gain=0.001):
        """Calculate power consumption of a vehicle.

        Assumes vehicle is an average sized vehicle.
        The power calculated here is the lower bound of the actual power consumed
        by a vehicle.
        """
        power = 0

        M = 1200  # mass of average sized vehicle (kg)
        g = 9.81  # gravitational acceleration (m/s^2)
        Cr = 0.005  # rolling resistance coefficient
        Ca = 0.3  # aerodynamic drag coefficient
        rho = 1.225  # air density (kg/m^3)
        A = 2.6  # vehicle cross sectional area (m^2)
        # for veh_id in env.k.vehicle.get_ids():
        speed = self.vehicle.velocity
        accel = self.vehicle.acceleration
        power += M * speed * accel + M * g * Cr * speed + 0.5 * rho * A * Ca * speed ** 3
        return - power * gain  # kilo Watts (KW)
