import math
from typing import DefaultDict
import numpy as np
import pandas as pd
import os
from rl_env.adv_spd_env_road_multi import Vehicle, TrafficSignal, AdvSpdEnvRoadMulti


class SectionMaxSpeed(object):
    def __init__(self, offset_info, track_length, min_speed=30, max_speed=50, set_local_speed_limit=False):
        self.track_length = track_length
        self.min_speed = min_speed/3.6
        self.max_speed = max_speed/3.6

        # offset_info = offset_dict
        temp = [offset_info[x]['offset'] for x in offset_info]
        running_length = 0
        offset_starts = []
        offset_ends = []
        max_speed = []
        for i in range(len(temp)):
            for offset in temp[i].values():
                offset_starts.append(np.round(offset['start']+running_length, 3))
                offset_ends.append(np.round(offset['end']+running_length, 3))
                max_speed.append(offset['maxSpeed'])
            running_length += offset['end']

        self.offset_starts = offset_starts
        self.offset_ends = offset_ends

        self.num_section = len(offset_starts)
        self.section_max_speed = np.ones(shape=(self.num_section+1,)) * self.max_speed
        self.sms_list = [[0, self.section_max_speed]]

        if set_local_speed_limit:
            for i in range(self.num_section):
                self.section_max_speed[i] = max_speed[i]/3.6

    def get_cur_idx(self, x):
        for i, x_i in enumerate(self.offset_starts):
            if x_i > x:
                break
        return i-1, x_i

    def get_cur_max_speed(self, x):
        i, _ = self.get_cur_idx(x)
        return self.section_max_speed[i]

    def get_next_max_speed(self, x):
        i, _ = self.get_cur_idx(x)
        return self.section_max_speed[i+1]

    def get_distance_to_next_section(self, x):
        _, x_i = self.get_cur_idx(x)
        return x_i - x


class AdvSpdEnvRoadMulti_SRC(AdvSpdEnvRoadMulti):
    def __init__(self, src, num_signal=5, num_action_unit=3, dt=0.1, action_dt=5, track_length=1500.0, acc_max=2, acc_min=-3,
                 speed_max=50.0/3.6, speed_min=0, dec_th=-3, stop_th=2, reward_coef=[1, 10, 1, 0.01, 0, 1, 1, 1],
                 timelimit=7500, unit_length=100, unit_speed=10, stochastic=False, min_location=250, max_location=350):

        super(AdvSpdEnvRoadMulti_SRC, self).__init__(num_signal, num_action_unit, dt, action_dt, track_length, acc_max, acc_min,
                                                     speed_max, speed_min, dec_th, stop_th, reward_coef, timelimit, unit_length, unit_speed, stochastic, min_location, max_location)

        route_src = pd.read_csv(src)
        route_src = route_src.loc[:83]  # Limit to 2 signals in Sejong

        offset_dict = dict()
        signal_dict = dict()

        # for i in range(route_src.shape[0]):
        running_distance = 0
        signal_idx = 0
        for i in range(len(route_src)):
            if not route_src.iloc[i]['linkSeq'] in offset_dict.keys():
                offset_dict[route_src.iloc[i]['linkSeq']] = {"linkID": route_src.iloc[i]['linkID'],
                                                             "offset": {}}
            offset_dict[route_src.iloc[i]['linkSeq']]['offset'][route_src.iloc[i]['offsetSeq']] = {"start": route_src.iloc[i]['offsetStart']/100,
                                                                                                   "end": route_src.iloc[i]['offsetEnd']/100,
                                                                                                   "maxSpeed": float(route_src.iloc[i]['MaxSpeed'])}
            running_distance += route_src.iloc[i]['offsetEnd']/100-route_src.iloc[i]['offsetStart']/100
            if route_src.iloc[i]['existSignal']:
                signal_dict[signal_idx] = {"location": running_distance,
                                           "signalGreen": int(route_src.iloc[i]['signalGreenPhaseLength']),
                                           "signalRed": int(route_src.iloc[i]['siganlRedPhaseLength']),
                                           "signalOffset": int(route_src.iloc[i]['signalOffset']),
                                           "signalGroup": int(route_src.iloc[i]['SignalGroup']),
                                           "signalNumber": int(route_src.iloc[i]['SignalNumber']),
                                           "signalName": str(route_src.iloc[i]['SignalName'])
                                           }
                signal_idx += 1

        self.offset_dict = offset_dict
        self.signal_dict = signal_dict
        self.track_length = running_distance
        self.num_signal = len(signal_dict)

        self.reset()

    def reset(self):
        self.vehicle = Vehicle(timelimit=self.timelimit)

        rand_offset = np.random.randint(0, min([self.signal_dict[x]['signalGreen']+self.signal_dict[x]['signalRed'] for x in self.signal_dict]))

        self.signal = []
        for signal_i in self.signal_dict.values():
            new_signal = TrafficSignal(location=signal_i['location'],
                                       red_time=signal_i['signalRed'],
                                       green_time=signal_i['signalGreen'],
                                       offset=signal_i['signalOffset'] + rand_offset)
            self.signal.append(new_signal)

        self.section = SectionMaxSpeed(self.offset_dict, self.track_length, max_speed=self.speed_max*3.6)
        self.section_input = SectionMaxSpeed(self.offset_dict, self.track_length, max_speed=self.speed_max*3.6, set_local_speed_limit=True)
        # self.section_input.section_max_speed

        self.timestep = 0
        self.violation = False

        self.state = self._get_state()

        self.reward = 0
        self.done = False
        self.info = {}
        self.png_list = []
        self.reward_per_unitlen = []
        # self.reward_at_time = np.zeros((int(self.timelimit/self.action_dt/10), 2))

        return self.state

    def _take_action(self, action):
        # applied_action = (action + 1) * self.unit_speed / 3.6
        applied_action = (action) * self.unit_speed + self.speed_min*3.6

        cur_idx, _ = self.section.get_cur_idx(self.vehicle.position)
        if cur_idx > 0:
            prev_speed = min(self.section.section_max_speed[cur_idx-1], self.section_input.section_max_speed[cur_idx-1])
            if np.abs(applied_action - prev_speed) > (self.unit_speed/3.6*2):
                if applied_action > prev_speed + self.unit_speed/3.6*2:
                    applied_action = prev_speed + self.unit_speed/3.6*2
                elif applied_action < prev_speed - self.unit_speed/3.6*2:
                    applied_action = prev_speed - self.unit_speed/3.6*2
                else:
                    applied_action = applied_action

                # applied_action = max(prev_speed - self.unit_speed/3.6*2, min(prev_speed + self.unit_speed/3.6*2, applied_action))

        self.section.section_max_speed[cur_idx] = applied_action

        cur_idx_old = int(cur_idx)
        reward_list = []

        while cur_idx == cur_idx_old:
            self.timestep += 1

            acceleration = self.get_veh_acc_idm(self.vehicle.position, self.vehicle.velocity)

            assert(acceleration >= self.acc_min)
            assert(acceleration <= self.acc_max)

            if self.vehicle.velocity + acceleration * self.dt < 0:
                acceleration = - self.vehicle.velocity / self.dt

            prev_position = self.vehicle.position
            sig = self._get_signal()

            self.vehicle.update(acceleration, self.timestep, self.dt)
            cur_position = self.vehicle.position

            pass_sig_reward = 0
            if sig is not None:
                if sig.location > prev_position:
                    if sig.location < cur_position:
                        pass_sig_reward = self.vehicle.velocity / self.speed_max

            reward = self._get_reward()
            reward[5] += pass_sig_reward

            # max_speed = self.section.get_cur_max_speed(self.vehicle.position)
            # reward_norm_velocity1 = np.abs((self.vehicle.velocity) - max_speed)
            # reward_norm_velocity1 /= self.speed_max

            # reward[0] +=

            reward_with_coef = np.array(reward).dot(np.array(self.reward_coef))
            reward_list.append(reward)

            cur_idx, _ = self.section.get_cur_idx(self.vehicle.position)
            try:
                self.vehicle.veh_info[self.timestep][4] = reward_with_coef
                # self.section.sms_list.append([self.timestep/10, self.section.section_max_speed])
                self.section.sms_list.append(
                    [self.timestep/10, np.min(np.stack([self.section.section_max_speed, self.section_input.section_max_speed]), 0)])
                self.vehicle.veh_info[self.timestep][5] = min(self.section.section_max_speed[math.floor(
                    self.vehicle.position/self.unit_length)], self.section_input.section_max_speed[math.floor(self.vehicle.position/self.unit_length)])
            except:
                # print(self.timestep)
                break

            if self.vehicle.position > self.track_length:
                break

            if self.timestep > self.timelimit:
                break

        assert(self.vehicle.velocity >= 0)
        assert(self.vehicle.position >= 0)

        return reward_list
