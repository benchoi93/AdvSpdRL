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
    def __init__(self):
        max_speed = 50 / 3.6
        self.position = 0
        # self.velocity = np.random.rand() * max_speed
        self.velocity = 40/3.6
        self.acceleration = 0
        self.jerk = 0
        self.action_limit_penalty = 1
        self.actiongap = 0


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
        self.section_max_speed[0] = 50/3.6
        self.section_max_speed[1] = 30/3.6
        self.section_max_speed[2] = 50/3.6
        self.section_max_speed[3] = 30/3.6
        self.section_max_speed[4] = 50/3.6

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
    def __init__(self, min_location=250, max_location=350):

        self.phase_length = {True: 30, False: 90}
        self.cycle_length = sum(self.phase_length.values())

        self.location = 300
        # self.location = min_location + np.random.rand() * (max_location - min_location)
        self.timetable = np.ones(shape=[self.cycle_length]) * -1

        self.offset = 40
        # offset = np.random.randint(0, self.cycle_length)

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


class AdvSpdEnv(gym.Env):
    png_list = []

    def __init__(self, dt=0.1, track_length=500.0, acc_max=5, acc_min=-5, speed_max=100.0/3.6, dec_th=-3, stop_th=2, reward_coef=[1, 10, 1, 0.01, 0, 1, 1, 1], timelimit=2400, unit_length=100):
        png_list = []

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
        self.unit_length = unit_length

        self.action_space = spaces.Box(low=acc_min, high=acc_max, shape=[1, ])
        self.reset()

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
                               self.speed_max,  # velocity
                               self.speed_max,  # cur_max_speed
                               self.speed_max,  # next_max_speed
                               self.unit_length,  # distance to next section
                               self.track_length*2,   # signal location
                               self.timelimit*2,   # green time start
                               self.timelimit*2  # green time end
                               ])

        self.observation_space = spaces.Box(low=min_states,
                                            high=max_states)

        # self.png_list = []
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
                      self.section.get_cur_max_speed(self.vehicle.position),
                      self.section.get_next_max_speed(self.vehicle.position),
                      self.section.get_distance_to_next_section(self.vehicle.position),
                      self.signal.location,
                      self.signal.get_greentime(int(self.timestep*self.dt))[0] / self.dt - self.timestep,
                      self.signal.get_greentime(int(self.timestep*self.dt))[1] / self.dt - self.timestep
                      ]
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
        self.section = SectionMaxSpeed(self.track_length, self.unit_length)

        self.timestep = 0
        self.violation = False

        self.state = self._get_state()

        self.reward = 0
        self.done = False
        self.info = {}

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

    def _take_action(self, action):
        applied_action = action
        self.vehicle.actiongap = action

        max_acc = self.calculate_max_acceleration()
        if max_acc < action:
            applied_action = max_acc

        if self.vehicle.velocity + applied_action * self.dt < 0:
            applied_action = - self.vehicle.velocity / self.dt

        self.vehicle.jerk = abs(applied_action - self.vehicle.acceleration) / self.dt
        self.vehicle.acceleration = applied_action

        self.vehicle.actiongap -= applied_action

        assert(np.round(self.vehicle.velocity + applied_action * self.dt, -5) >= 0)
        self.vehicle.velocity = self.vehicle.velocity + applied_action * self.dt
        self.vehicle.position = self.vehicle.position + self.vehicle.velocity * self.dt
        # self.vehicle.action_limit_penalty = abs(applied_action / action)

    def _get_reward(self):
        max_speed = self.section.get_cur_max_speed(self.vehicle.position)
        reward_norm_velocity = np.abs((self.vehicle.velocity) - max_speed)
        reward_norm_velocity /= max_speed

        reward_jerk = np.abs(self.vehicle.jerk)
        jerk_max = (self.acc_max - self.acc_min) / self.dt
        reward_jerk /= jerk_max

        reward_shock = 1 if self.vehicle.velocity > self.section.get_cur_max_speed(self.vehicle.position) else 0
        penalty_signal_violation = 1 if self.violation else 0
        # penalty_action_limit = self.vehicle.action_limit_penalty if self.vehicle.action_limit_penalty != 1 else 0
        # penalty_moving_backward = 1000 if self.vehicle.velocity < 0 else 0
        penalty_travel_time = 1

        reward_remaining_distance = (self.track_length - self.vehicle.position) / self.track_length

        reward_action_gap = self.vehicle.actiongap / (self.acc_max - self.acc_min)
        # reward_finishing = 1000 if self.vehicle.position > 490 else 0
        # reward_power = self.energy_consumption() * self.dt / 75 * 0.5

        power = -self.energy_consumption()
        reward_power = self.vehicle.velocity / power if (power > 0 and self.vehicle.velocity >= 0) else 0

        return [-reward_norm_velocity, -reward_shock, -reward_jerk, -reward_power, -penalty_travel_time, -penalty_signal_violation, -reward_remaining_distance, -reward_action_gap]
        # return -reward_norm_velocity - \
        #     reward_shock - \
        #     reward_jerk - \
        #     reward_power - \
        #     penalty_travel_time
        # return reward_norm_velocity

    def calculate_max_acceleration(self):

        dec_th = self.dec_th
        max_acc = self.acc_max

        spd_max_acc = (self.speed_max - self.vehicle.velocity)/self.dt
        max_acc = spd_max_acc if max_acc > spd_max_acc else max_acc

        if not self.signal.is_green(int(self.timestep * self.dt)):

            if self.vehicle.position < self.signal.location:
                mild_stopping_distance = -(self.vehicle.velocity + self.acc_max * self.dt) ** 2 / (2 * (dec_th))
                distance_to_signal = self.signal.location - self.stop_th - self.vehicle.position
                
                if distance_to_signal < mild_stopping_distance:
                    if distance_to_signal == 0:
                        max_acc = 0
                    else:
                        max_acc = -(self.vehicle.velocity ) ** 2 / (2 * (distance_to_signal))

        assert(max_acc >= self.acc_min)
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

    def info_graph(self, ob_list, check_finish=0):
        t1 = time.time()

        pos = [ob[0] for ob in ob_list]
        vel = [ob[1] for ob in ob_list]
        acc = [ob[2] for ob in ob_list]
        step = [ob[3] for ob in ob_list]
        reward = [ob[4] for ob in ob_list]
        print(step[-1]/10)
        # info figures
        plt.rc('font', size=15)
        plt.rc('axes', titlesize=22)
        plt.rc('axes', labelsize=15)
        plt.rc('xtick', labelsize=15)
        plt.rc('ytick', labelsize=15)

        fig = plt.figure(figsize=(15, 10)) # 여기서 에러... -> Fail to create pixmap with Tk_GetPixmap in TkImgPhotoInstanceSetSize
        fig.clf()
        
        # pos-vel
        ax1 = fig.add_subplot(221)
        ax1.plot(pos, vel, lw=2, color='k')
        section_max_speed = self.section.section_max_speed
        unit_length = self.unit_length
        for i in range(len(section_max_speed)):
            ax1.plot(np.linspace(i*unit_length, (i+1)*unit_length, unit_length*10),
                     [section_max_speed[i]]*(unit_length*10), lw=2, color='r')
        ax1.set_title('x-v graph')
        ax1.set_xlabel('Position in m')
        ax1.set_ylabel('Velocity in km/h')
        ax1.set_xlim((0.0, self.track_length))
        ax1.set_ylim((0.0, 50))

        # pos-acc
        ax2 = fig.add_subplot(222)
        ax2.plot(pos, acc, lw=2, color='k')
        ax2.set_title('x-a graph')
        ax2.set_xlabel('Position in m')
        ax2.set_ylabel('Acceleration in m/s²')
        ax2.set_xlim((0.0, self.track_length))
        ax2.set_ylim((self.acc_min, self.acc_max))

        # x-t with signal phase
        ax3 = fig.add_subplot(223)
        ax3.plot([x*self.dt for x in range(len(pos))], pos, lw=2, color='k')
        
        # xlim_max = self.timelimit
        # for i in range(xlim_max):
        #     if self.signal.is_green(i):
        #         signal_color = 'g'
        #     else:
        #         signal_color = 'r'

        #     ax3.plot([x*self.dt for x in range(int((i)/self.dt), int((i + 1)//self.dt)+1)],
        #              [self.signal.location] * len(range(int(i/self.dt), int((i + 1)//self.dt)+1)),
        #              color=signal_color
        #              )
        green = self.signal.phase_length[True]
        red = self.signal.phase_length[False]
        cycle = green+red
        for i in range(3):
            ax3.plot(np.linspace(cycle*i-(cycle-self.signal.offset), cycle*i-(cycle-self.signal.offset)+green, green*10), 
                                 [self.signal.location]*(green*10), lw=2, color='g')
            ax3.plot(np.linspace(cycle*i-(cycle-self.signal.offset)+green, cycle*i-(cycle-self.signal.offset)+cycle, red*10), 
                                 [self.signal.location]*(red*10), lw=2, color='r')
        ax3.set_title('x-t graph')
        ax3.set_xlabel('Time in s')
        ax3.set_ylabel('Position in m')
        ax3.set_xlim((0.0, math.ceil(step[-1]/100)*10))
        ax3.set_ylim((0, self.track_length))

        # t-reward
        ax4 = fig.add_subplot(224)
        ax4.plot([x*self.dt for x in range(len(reward))], reward, lw=2, color='k')
        # xlim_max = int(max(100, len(pos)) * self.dt) + 1
        ax4.set_title('reward-t graph')
        ax4.set_xlabel('Time in s')
        ax4.set_ylabel('Reward')
        ax4.set_xlim((0.0, math.ceil(step[-1]/100)*10))
        ax4.set_ylim((-3.0, 3.0))
        plt.subplots_adjust(hspace=0.35)

        print("make fig: {}".format(time.time()-t1))

        if check_finish == 1:
            plt.savefig('./simulate_gif/info_graph.png')

        return fig
        
    def car_moving(self, ob_list, startorfinish=0, combine=False):
        # t2 = time.time()
        pos = [int(np.round(ob[0], 0)) for ob in ob_list]
        step = [ob[3] for ob in ob_list]

        car_filename = "./util/assets/track/img/car_80x40.png"
        signal_filename = "./util/assets/track/img/sign_60x94.png"
        start_finish_filename = "./util/assets/track/img/start_finish_30x100.png"

        car = Image.open(car_filename)
        signal = Image.open(signal_filename)
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
        signal_position = (zero_x + int(scale_x * (self.signal.location - pos[-1])), canvas[1] - clearence[1] - 50)
        car_position = (zero_x - 80, canvas[1] - clearence[1] + 30)

        try:
            font = ImageFont.truetype('arial.ttf', 25)
            timefont = ImageFont.truetype('arial.ttf', 40)
        except:
            font = ImageFont.load_default()

        background = Image.new('RGB', canvas, (255, 255, 255))
        draw = ImageDraw.Draw(background)
        for i in range(int(self.track_length//50+1)):
            draw.text((zero_x - int(scale_x * (pos[-1])) + int(scale_x*50*i), canvas[1] - 90), "{}m".format(50*i), (0, 0, 0), font)
        draw.line((0, canvas[1]-100, 1500, canvas[1]-100), (0, 0, 0), width=5)
        background.paste(start, start_position)
        background.paste(finish, finish_position)
        signal_draw = ImageDraw.Draw(signal)
        if self.signal.is_green(int(self.timestep * self.dt)):
            signal_draw.ellipse((0, 0, 60, 60,), (0, 255, 0))  # green signal
        else:
            signal_draw.ellipse((0,0,60,60,), (255, 0, 0))  # red signal

        background.paste(signal, signal_position, signal)
        background.paste(car, car_position, car)

        # print("make car-moving: {}".format(time.time()-t2))

        # self.info_graph(ob_list)
        # graph = Image.open('./simulate_gif/graph_{}.png'.format(time[-1]))

        if combine == True:
            t3 = time.time()
            graph = fig2img(self.info_graph(ob_list))
            plt.close()
            background.paste(graph, (0, 50))
            print("convert: {}".format(time.time()-t3))
        
        draw.text((10, 10), "Time Step: {}s".format(step[-1]/10), (0,0,0), timefont)

        if startorfinish == 1:
            for i in range(100):
                self.png_list.append(background)
                    
        else:
            self.png_list.append(background)
        
    def make_gif(self, path="./simulate_gif/simulation.gif"):
        self.png_list[0].save(path, save_all=True, append_images=self.png_list[1:], optimize=False, duration=20, loop=1)


def fig2img(fig):
    t4 = time.time()
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)

    print("fig2img: {}".format(time.time()-t4))
    return img