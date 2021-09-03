import gym
import numpy as np
import os
import io
import glob
import math
import time
from gym import spaces
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont


class Vehicle(object):
    def __init__(self, initial_speed=40/3.6, timelimit=7500, num_states=6):
        # init_max_speed = 50 / 3.6
        self.position = 0
        # self.velocity = np.random.rand() * init_max_speed
        self.velocity = initial_speed
        self.acceleration = 0
        self.jerk = 0
        self.action_limit_penalty = 1
        self.actiongap = 0

        # 1: pos / 2: vel / 3: acc / 4: timestep / 5: reward / 6: max speed at the time in the section
        self.veh_info = np.zeros((timelimit, num_states))

    def update(self, acc, timestep, dt):
        self.jerk = abs(acc - self.acceleration) / dt
        self.acceleration = acc

        # assert(np.round(self.velocity + acc * dt, -5) >= 0)
        self.velocity = self.velocity + acc * dt
        self.position = self.position + self.velocity * dt
        # print(f'Position = {np.round(self.position, 4)} || Velocity = {np.round(self.velocity, 4)} || Acc = {np.round(self.acceleration, 4)} || Time = {timestep/10}')

        assert(self.velocity >= 0)
        assert(self.position >= 0)

        # self.veh_info.append([self.position, self.velocity, self.acceleration, self.timestep])
        self.veh_info[timestep] = [self.position, self.velocity, self.acceleration, timestep, 0, 0]


class SectionMaxSpeed(object):
    def __init__(self, track_length=500, unit_length=100, min_speed=30, max_speed=50):
        assert(min_speed <= max_speed)
        assert(unit_length*2 <= track_length)

        self.track_length = track_length
        self.unit_length = unit_length
        self.min_speed = min_speed/3.6
        self.max_speed = max_speed/3.6

        self.num_section = int(self.track_length / self.unit_length)
        assert(self.num_section > 0)
        self.section_max_speed = self.min_speed + np.random.random(size=self.num_section+1) * (self.max_speed - self.min_speed)
        # self.section_max_speed[0] = 50/3.6
        # self.section_max_speed[1] = 30/3.6
        # self.section_max_speed[2] = 50/3.6
        # self.section_max_speed[3] = 30/3.6
        # self.section_max_speed[4] = 50/3.6
        self.sms_list = [[0, self.section_max_speed]]
        # np.zeros((int(self.timelimit/self.action_dt/10), 2))

    def get_cur_max_speed(self, x):
        return self.section_max_speed[int(x/self.unit_length)]

    def get_next_max_speed(self, x):
        i = int(x/self.unit_length)
        if i+1 < self.num_section-1:
            return self.section_max_speed[i+1]
        else:
            return self.max_speed

    def get_distance_to_next_section(self, x):
        i = int(x/self.unit_length)
        return (i+1) * self.unit_length - x


class TrafficSignal(object):
    def __init__(self, min_location=250, max_location=350, green_time=30, red_time=90, location=300, offset=40, offset_rand=False):

        self.phase_length = {True: green_time, False: red_time}
        self.cycle_length = sum(self.phase_length.values())

        self.location = location
        # self.location = min_location + np.random.rand() * (max_location - min_location)
        self.timetable = np.ones(shape=[self.cycle_length]) * -1

        self.offset = offset
        if offset_rand:
            self.offset = np.random.randint(0, self.cycle_length)

        for i in range(self.cycle_length):
            cur_idx = (i+self.offset) % self.cycle_length
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


class AdvSpdEnvRoadMulti(gym.Env):
    # png_list = []

    def __init__(self, num_signal, num_action_unit, dt=0.1, action_dt=5, track_length=1500.0, acc_max=2, acc_min=-3,
                 speed_max=50.0/3.6, dec_th=-3, stop_th=2, reward_coef=[1, 10, 1, 0.01, 0, 1, 1, 1],
                 timelimit=7500, unit_length=100, unit_speed=10, stochastic=False, min_location=250, max_location=350):

        # num_observations = 2

        self.num_signal = num_signal
        self.num_action_unit = num_action_unit

        self.dt = dt
        self.action_dt = action_dt
        self.num_action_updates = int(self.action_dt / self.dt)
        self.track_length = track_length
        self.acc_min = acc_min
        self.acc_max = acc_max
        self.speed_max = speed_max
        self.dec_th = dec_th
        self.stop_th = stop_th
        self.reward_coef = reward_coef
        self.stochastic = stochastic
        # [-reward_norm_velocity, -reward_shock, -reward_jerk, -reward_power, -penalty_travel_time]
        self.timelimit = timelimit
        self.unit_length = unit_length
        self.unit_speed = unit_speed
        self.min_location = min_location
        self.max_location = max_location

        # self.action_space = spaces.Tuple(([spaces.Discrete(int(speed_max * 3.6 / unit_speed) + 1) for i in range(int(track_length/unit_length)+1)]))
        self.action_space = spaces.MultiDiscrete([int(speed_max*2 * 3.6 / unit_speed)] * self.num_action_unit)

        min_states = np.array([0.0,  # position
                               0.0,  # velocity
                               0.0,  # cur_max_speed
                               0.0,  # next_max_speed
                               0.0,  # distance to next section
                               0.0,  # signal location
                               0.0,  # green time start
                               0.0  # green time end
                               ])
        max_states = np.array([self.track_length*2,  # position
                               self.speed_max*2,  # velocity
                               self.speed_max*2,  # cur_max_speed
                               self.speed_max*2,  # next_max_speed
                               self.unit_length,  # distance to next section
                               self.track_length*2,   # signal location
                               self.timelimit*2,   # green time start
                               self.timelimit*2  # green time end
                               ])

        self.observation_space = spaces.Box(low=min_states,
                                            high=max_states)
        self.reset()

        # self.png_list = []
        # self.scenario = scenario
        self.viewer = None
        pass

    def save(self):
        pass

    def load(self):
        pass

    def get_random_action(self):
        return self.action_space.sample()

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
        # prev_ob = self.state

        reward_list = self._take_action(action)

        # print(self.timestep/10)

        ob = self._get_state()

        reward = reward_list.sum()

        self.reward_at_time[int(self.timestep/self.action_dt/10)-1] = [self.timestep/10, reward]

        episode_over = (self.vehicle.position > self.track_length) or (self.timestep > self.timelimit)

        if episode_over == True:
            self.reward_at_time = self.reward_at_time[self.reward_at_time[:, 0] != 0]
        return ob, reward, episode_over, {'vehicle_ob': self.vehicle.veh_info[-1]}

    def _get_signal(self):
        # location = 1e10
        sig = None

        for sig in self.signal:
            if sig.location > self.vehicle.position:
                location = sig.location
                break
        if sig is None:
            sig = TrafficSignal(location=1e10, green_time=119, red_time=1)
        return sig

    def _get_state(self):
        sig = self._get_signal()

        self.state = [self.vehicle.position,
                      self.vehicle.velocity,
                      self.section.get_cur_max_speed(self.vehicle.position),
                      self.section.get_next_max_speed(self.vehicle.position),
                      self.section.get_distance_to_next_section(self.vehicle.position),
                      sig.location,
                      sig.get_greentime(int(self.timestep*self.dt))[0] / self.dt - self.timestep,
                      sig.get_greentime(int(self.timestep*self.dt))[1] / self.dt - self.timestep
                      ]

        return self.state

    def reset(self):

        # if self.stochastic:
        #     self.vehicle = Vehicle(initial_speed=np.random.rand() * self.speed_max/3.6)
        #     self.signal = TrafficSignal(location=self.min_location + np.random.rand() * (self.max_location - self.min_location),
        #                                 offset_rand=True)
        # else:
        self.vehicle = Vehicle(timelimit=self.timelimit)
        self.signal = [TrafficSignal(location=300), TrafficSignal(location=600), TrafficSignal(location=900)]

        self.section = SectionMaxSpeed(self.track_length, self.unit_length)
        self.section_input = SectionMaxSpeed(self.track_length, self.unit_length)
        # self.section_input.section_max_speed

        self.timestep = 0
        self.violation = False

        self.state = self._get_state()

        self.reward = 0
        self.done = False
        self.info = {}
        self.png_list = []
        self.reward_at_time = np.zeros((int(self.timelimit/self.action_dt/10), 2))

        return self.state

    def render(self, mode='human', info_show=False, close=False, visible=True):
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
            finish.position = (zero_x + scale_x * self.track_length,
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
            for i in range(self.section.num_section):
                from_x = i * self.section.unit_length
                to_x = (i+1) * self.section.unit_length
                ax.plot([from_x, to_x],
                        [self.section.get_cur_max_speed(from_x)*3.6, self.section.get_cur_max_speed(from_x)*3.6],
                        lw=1.5,
                        color='r')

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

            self.viewer.checkfinish = True

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
        return self.viewer.render(return_rgb_array=mode == 'rgb_array', visible=visible)

    # def get_veh_acceleration(self, position, velocity):
    #     acceleration = self.acc_max
    #     max_speed = self.section.get_cur_max_speed(position)

    #     if max_speed >= velocity:
    #         # acceleration
    #         acceleration = min((max_speed - velocity)/self.dt, self.acc_max)
    #     else:
    #         # deceleration
    #         acceleration = max((max_speed - velocity)/self.dt, self.acc_min)

    #     # signal stopping
    #     sig_max_acc = self.calculate_max_acceleration()
    #     acceleration = min(sig_max_acc, acceleration)

    #     assert(acceleration >= self.acc_min)
    #     assert(acceleration <= self.acc_max)
    #     return acceleration

    def _take_action(self, action):
        applied_action = (action + 1) * self.unit_speed

        cur_idx = int(self.vehicle.position/self.unit_length) + 1
        num_action_in = (min(cur_idx+self.num_action_unit, len(self.section.section_max_speed)-1)) - cur_idx
        self.section.section_max_speed[cur_idx:(
            min(cur_idx+self.num_action_unit, len(self.section.section_max_speed)-1))] = applied_action[:num_action_in] / 3.6

        reward_list = np.zeros(self.num_action_updates)
        i = 0
        for _ in range(self.num_action_updates):
            self.timestep += 1
            # print(f"{self.vehicle.position=} || {self.vehicle.velocity=}")
            # acceleration = self.get_veh_acceleration(self.vehicle.position, self.vehicle.velocity)
            acceleration = self.get_veh_acc_idm(self.vehicle.position, self.vehicle.velocity)

            assert(acceleration >= self.acc_min)
            assert(acceleration <= self.acc_max)

            if self.vehicle.velocity + acceleration * self.dt < 0:
                acceleration = - self.vehicle.velocity / self.dt

            prev_position = self.vehicle.position
            # print("acceleration =", np.round(acceleration, 4))
            self.vehicle.update(acceleration, self.timestep, self.dt)

            cur_position = self.vehicle.position
            sig = self._get_signal()
            if prev_position <= sig.location:
                if cur_position > sig.location:
                    if not sig.is_green(int(self.timestep * self.dt)):
                        self.violation = True

            reward = self._get_reward()
            reward_with_coef = np.array(reward).dot(np.array(self.reward_coef))
            reward_list[i] = reward_with_coef
            i += 1
            self.vehicle.veh_info[self.timestep][4] = reward_with_coef

            self.section.sms_list.append([self.timestep/10, self.section.section_max_speed])

            self.vehicle.veh_info[self.timestep][5] = self.section.section_max_speed[math.floor(self.vehicle.position/self.unit_length)]

            if self.vehicle.position > self.track_length:
                break

            if self.timestep > self.timelimit:
                break

        assert(self.vehicle.velocity >= 0)
        assert(self.vehicle.position >= 0)

        return reward_list

    def _get_reward(self):
        max_speed = self.section.get_cur_max_speed(self.vehicle.position)
        reward_norm_velocity = np.abs((self.vehicle.velocity) - max_speed)
        reward_norm_velocity /= self.speed_max

        reward_jerk = np.abs(self.vehicle.jerk)
        jerk_max = (self.acc_max - self.acc_min) / self.dt
        reward_jerk /= jerk_max

        reward_shock = 0
        if self.vehicle.velocity > self.section.get_cur_max_speed(self.vehicle.position):
            reward_shock += 1
        if self.vehicle.velocity > self.speed_max:
            reward_shock += 1

        penalty_signal_violation = 1 if self.violation else 0
        # penalty_action_limit = self.vehicle.action_limit_penalty if self.vehicle.action_limit_penalty != 1 else 0
        # penalty_moving_backward = 1000 if self.vehicle.velocity < 0 else 0
        penalty_travel_time = 1

        reward_remaining_distance = (self.track_length - self.vehicle.position) / self.track_length

        reward_action_gap = self.vehicle.actiongap / (self.acc_max - self.acc_min)
        # reward_finishing = 1000 if self.vehicle.position > 490 else 0
        # reward_power = self.energy_consumption() * self.dt / 75 * 0.5
        power = -self.energy_consumption()
        reward_power = max(power, 0) / 100

        # power = -self.energy_consumption()
        # reward_power = self.vehicle.velocity / power if (power > 0 and self.vehicle.velocity >= 0) else 0

        return [-reward_norm_velocity, -reward_shock, -reward_jerk, -reward_power, -penalty_travel_time, -penalty_signal_violation, -reward_remaining_distance, -reward_action_gap]
        # return -reward_norm_velocity - \
        #     reward_shock - \
        #     reward_jerk - \
        #     reward_power - \
        #     penalty_travel_time
        # return reward_norm_velocity

    def get_veh_acc_idm(self, position, velocity):

        # signal on = 가상의 Vehicle
        # signal off leader position = inf
        import math
        des_speed = self.section.get_cur_max_speed(position)
        delta = 4
        a = self.acc_max
        b = self.acc_min
        s_0 = 1
        des_timeheadway = 1
        leader_position = 1e10
        sig = self._get_signal()

        if position <= sig.location:
            if not sig.is_green(int(self.timestep*self.dt)):
                leader_position = sig.location

        relative_speed = (velocity-0)
        spacing = leader_position - position

        des_distance = s_0 + velocity * des_timeheadway + velocity * relative_speed / (2 * math.sqrt(abs(a * b)))
        acceleration = a * (1 - (velocity/des_speed)**delta - (des_distance/spacing)**2)
        acceleration = max(self.acc_min, min(self.acc_max, acceleration))
        assert(acceleration >= self.acc_min)
        assert(acceleration <= self.acc_max)
        return acceleration

        # return self.acc_max

    # def calculate_max_acceleration(self):
    #     dec_th = self.dec_th
    #     max_acc = self.acc_max

    #     spd_max_acc = (self.speed_max - self.vehicle.velocity)/self.dt
    #     max_acc = spd_max_acc if max_acc > spd_max_acc else max_acc
    #     # print("max_acc = ", np.round(max_acc, 4))
    #     # print("section max speed =", np.round(self.section.get_cur_max_speed(self.vehicle.position), 4))
    #     if not self.signal.is_green(int(self.timestep * self.dt)):
    #         if self.vehicle.position < self.signal.location:
    #             mild_stopping_distance = -(self.vehicle.velocity + self.acc_max * self.dt) ** 2 / (2 * (dec_th))
    #             distance_to_signal = self.signal.location - self.stop_th - self.vehicle.position
    #             # print(f'mild stopping distance = {np.round(mild_stopping_distance, 4)} || distance to signal = {np.round(distance_to_signal, 4)}')
    #             if distance_to_signal < mild_stopping_distance:
    #                 if distance_to_signal == 0:
    #                     max_acc = 0
    #                 else:
    #                     # print("warning!")
    #                     max_acc = -(self.vehicle.velocity) ** 2 / (2 * (distance_to_signal))

    #                 # print("max_acc = ", np.round(max_acc, 4))
    #             # else:
    #             #     if self.vehicle.velocity <= 2:
    #             #         max_acc = -(self.vehicle.velocity) ** 2 / (2 * (distance_to_signal))

    #     assert(max_acc >= self.acc_min)
    #     return max_acc

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

    def info_graph(self, veh_info, check_finish=False):
        # t1 = time.time()
        pos = veh_info[:, 0]
        vel = veh_info[:, 1]*3.6
        acc = veh_info[:, 2]
        step = veh_info[:, 3]/10
        # reward = veh_info[:, 4]
        reward = self.reward_at_time[self.reward_at_time[:, 0] <= step[-1]]
        maxspeed = veh_info[:, 5]*3.6

        # print(reward)
        print("time:", step[-1])

        # info figures
        plt.rc('font', size=15)
        plt.rc('axes', titlesize=22)
        plt.rc('axes', labelsize=15)
        plt.rc('xtick', labelsize=15)
        plt.rc('ytick', labelsize=15)

        if check_finish == True:
            fig = plt.figure(figsize=(15, 20))
            fig.clf()
            ax1 = fig.add_subplot(411)
            ax2 = fig.add_subplot(412)
            ax3 = fig.add_subplot(413)
            ax4 = fig.add_subplot(414)
        else:
            fig = plt.figure(figsize=(15, 10))
            fig.clf()
            ax1 = fig.add_subplot(221)
            ax2 = fig.add_subplot(222)
            ax3 = fig.add_subplot(223)
            ax4 = fig.add_subplot(224)

        # pos-vel
        section_max_speed = self.section.sms_list[int(step[-1]*10)][1]
        print(section_max_speed)
        unit_length = self.unit_length
        cur_idx = int(self.vehicle.position/self.unit_length)
        if check_finish == False:
            # for i in range(len(section_max_speed)):
            for i in range(cur_idx, min(cur_idx+self.num_action_unit+1, len(self.section.section_max_speed)-1)):
                ax1.plot(np.linspace(i*unit_length, (i+1)*unit_length, unit_length*10),
                         [section_max_speed[i]*3.6]*(unit_length*10), lw=2, color='r')

        ax1.plot(pos, vel, lw=2, color='k')
        ax1.plot(pos, maxspeed, lw=1.5, color='b', alpha=0.3)

        ax1.set_title('x-v graph')
        ax1.set_xlabel('Position in m')
        ax1.set_ylabel('Velocity in km/h')
        ax1.set_xlim((0.0, self.track_length))
        ax1.set_ylim((0.0, 110))

        # pos-acc
        ax2.plot(pos, acc, lw=2, color='k')
        ax2.set_title('x-a graph')
        ax2.set_xlabel('Position in m')
        ax2.set_ylabel('Acceleration in m/s²')
        ax2.set_xlim((0.0, self.track_length))
        ax2.set_ylim((self.acc_min-1, self.acc_max+1))

        # x-t with signal phase
        ax3.plot([x*self.dt for x in range(len(pos))], pos, lw=2, color='k')
        green = self.signal[0].phase_length[True]
        red = self.signal[0].phase_length[False]
        cycle = green+red
        for j in range(self.num_signal):
            for i in range(int(self.timelimit/10/cycle)+1):
                ax3.plot(np.linspace(cycle*i-(cycle-self.signal[j].offset), cycle*i-(cycle-self.signal[j].offset)+green, green*10),
                        [self.signal[j].location]*(green*10), lw=2, color='g')
                ax3.plot(np.linspace(cycle*i-(cycle-self.signal[j].offset)+green, cycle*i-(cycle-self.signal[j].offset)+cycle, red*10),
                        [self.signal[j].location]*(red*10), lw=2, color='r')
        ax3.set_title('x-t graph')
        ax3.set_xlabel('Time in s')
        ax3.set_ylabel('Position in m')
        # if step[-1] == 0:
        #     ax3.set_xlim((0.0, 10))
        # else:
        #     ax3.set_xlim((0.0, math.ceil(step[-1]/100)*10))
        ax3.set_xlim((0.0, math.ceil(self.timestep/10)))
        ax3.set_ylim((0, self.track_length))

        # t-reward
        # ax4.plot(step, reward, lw=2, color='k')
        # ax4.plot([-10, 240], [0, 0], lw=1, color='k')
        ax4.plot(reward[:, 0], reward[:, 1], lw=2, color='k', alpha=0.5)
        ax4.scatter(reward[:, 0], reward[:, 1], color='k')
        # xlim_max = int(max(100, len(pos)) * self.dt) + 1
        ax4.set_title('reward-t graph')
        ax4.set_xlabel('Time in s')
        ax4.set_ylabel('Reward')
        # if step[-1] == 0:
        #     ax4.set_xlim((0.0, 10))
        # else:
        #     ax4.set_xlim((0.0, math.ceil(step[-1]/100)*10))
        ax4.set_xlim((0.0, math.ceil(self.timestep/10)))
        reward_min = np.round(np.min(self.reward_at_time[:, 1]), 0)-2
        ax4.set_ylim((reward_min, 3.0))

        plt.subplots_adjust(hspace=0.35)

        # print("make fig: {}".format(time.time()-t1))

        if check_finish == True:
            plt.savefig('./simulate_gif/info_graph.png')

        return fig

    def car_moving(self, veh_info, startorfinish=False, combine=False):
        # t2 = time.time()
        pos = (np.round(veh_info[:, 0], 0)).astype(int)
        step = veh_info[:, 3]
        # print("pos:", pos)
        # print("step:", step)

        car_filename = "./util/assets/track/img/car_80x40.png"
        signal_filename = "./util/assets/track/img/sign_60x94.png"
        start_finish_filename = "./util/assets/track/img/start_finish_30x100.png"

        car = Image.open(car_filename)
        # signal = Image.open(signal_filename)
        signal = []
        for i in range(self.num_signal):
            signal.append(Image.open(signal_filename))
        start = Image.open(start_finish_filename)
        finish = Image.open(start_finish_filename)

        if combine == True:
            canvas = (1500, 1500)
        else:
            canvas = (1500, 500)

        clearence = (0, 200)
        zero_x = 150
        scale_x = 10
        signal_position = 300

        start_position = (zero_x - int(scale_x * (pos[-1])), canvas[1] - clearence[1])
        finish_position = (zero_x + int(scale_x * (self.track_length - pos[-1])), canvas[1] - clearence[1])
        # signal_position = (zero_x + int(scale_x * (self.signal.location - pos[-1])), canvas[1] - clearence[1] - 50)
        signal_position = []
        for i in range(self.num_signal):
            signal_position.append((zero_x + int(scale_x * (self.signal[i].location - pos[-1])), canvas[1] - clearence[1] - 50))
        car_position = (zero_x - 80, canvas[1] - clearence[1] + 30)

        try:
            font = ImageFont.truetype('arial.ttf', 25)
            timefont = ImageFont.truetype('arial.ttf', 40)
        except:
            font = ImageFont.load_default()
            timefont = ImageFont.load_default()

        background = Image.new('RGB', canvas, (255, 255, 255))
        draw = ImageDraw.Draw(background)
        for i in range(int(self.track_length//50+1)):
            draw.text((zero_x - int(scale_x * (pos[-1])) + int(scale_x*50*i), canvas[1] - 90), "{}m".format(50*i), (0, 0, 0), font)
        draw.line((0, canvas[1]-100, 1500, canvas[1]-100), (0, 0, 0), width=5)
        background.paste(start, start_position)
        background.paste(finish, finish_position)
        # signal_draw = ImageDraw.Draw(signal)
        signal_draw = []
        for i in range(self.num_signal):
            signal_draw.append(ImageDraw.Draw(signal[i]))
        
            if self.signal[i].is_green(int(step[-1] * self.dt)):
                signal_draw[i].ellipse((0, 0, 60, 60,), (0, 255, 0))  # green signal
            else:
                signal_draw[i].ellipse((0, 0, 60, 60,), (255, 0, 0))  # red signal
            
            background.paste(signal[i], signal_position[i], signal[i])
        # background.paste(signal, signal_position, signal)
        background.paste(car, car_position, car)

        # print("make car-moving: {}".format(time.time()-t2))

        # self.info_graph(ob_list)
        # graph = Image.open('./simulate_gif/graph_{}.png'.format(time[-1]))

        if combine == True:
            # t3 = time.time()
            graph = fig2img(self.info_graph(veh_info))
            plt.close()
            background.paste(graph, (0, 50))
            # print("convert: {}".format(time.time()-t3))

        draw.text((10, 10), "Time Step: {}s".format(step[-1]/10), (0, 0, 0), timefont)

        if startorfinish == True:
            for i in range(80):
                self.png_list.append(background)

        else:
            self.png_list.append(background)

    def make_gif(self, path="./simulate_gif/simulation.gif"):
        self.png_list[0].save(path, save_all=True, append_images=self.png_list[1:], optimize=False, duration=30, loop=1)


def fig2img(fig):
    # t4 = time.time()
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)

    # print("fig2img: {}".format(time.time()-t4))
    return img
