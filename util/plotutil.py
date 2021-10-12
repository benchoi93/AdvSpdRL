import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from matplotlib.gridspec import GridSpec
import gym
import numpy as np
import os
import io
import glob
import math
import time
from itertools import chain
from tqdm import tqdm
from gym import spaces


def info_graph(env_list, veh_info_list, check_finish=False, path='./simulate_gif/info_graph.png'):
    pos = [veh[:, 0] for veh in veh_info_list]
    vel = [veh[:, 1]*3.6 for veh in veh_info_list]
    acc = [veh[:, 2] for veh in veh_info_list]
    step = [veh[:, 3]/10 for veh in veh_info_list]
    reward = [env_list[i].reward_at_time[env_list[i].reward_at_time[:, 0] <= step[i][-1]] for i in range(len(env_list))]
    maxspeed = [veh[:, 5]*3.6 for veh in veh_info_list]

    timestep = np.max([env.timestep for env in env_list])

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

    color_list = ['k', 'darkorange', 'indigo', 'b']

    for i in range(len(env_list)):
        env = env_list[i]

        # pos-vel
        section_max_speed = env.section.sms_list[int(step[i][-1]*10)][1]
        unit_length = env.unit_length
        cur_idx = int(env.vehicle.position/env.unit_length)
        if check_finish == False:
            # for i in range(len(section_max_speed)):
            for j in range(cur_idx, min(cur_idx+env.num_action_unit+1, len(env.section.section_max_speed)-1)):
                ax1.plot(np.linspace(j*unit_length, (j+1)*unit_length, unit_length*10),
                        [section_max_speed[j]*3.6]*(unit_length*10), lw=2, color='r')
        ax1.plot(pos[i], vel[i], lw=2, color=color_list[i])
        ax1.plot(pos[i], maxspeed[i], lw=1.5, color=color_list[i], alpha=0.5)
        ax1.set_title('x-v graph')
        ax1.set_xlabel('Position in m')
        ax1.set_ylabel('Velocity in km/h')
        ax1.set_xlim((0.0, env.track_length))
        ax1.set_ylim((0.0, 110))

        # pos-acc
        ax2.plot(pos[i], acc[i], lw=2, color=color_list[i])
        ax2.set_title('x-a graph')
        ax2.set_xlabel('Position in m')
        ax2.set_ylabel('Acceleration in m/s²')
        ax2.set_xlim((0.0, env.track_length))
        ax2.set_ylim((env.acc_min-1, env.acc_max+1))

        # x-t with signal phase
        ax3.plot([x*env.dt for x in range(len(pos[i]))], pos[i], lw=2, color=color_list[i])
        green = env.signal[0].phase_length[True]
        red = env.signal[0].phase_length[False]
        cycle = green+red
        for j in range(env.num_signal):
            for k in range(int(env.timelimit/10/cycle)+1):
                ax3.plot(np.linspace(cycle*k-(cycle-env.signal[j].offset), cycle*k-(cycle-env.signal[j].offset)+green, green*10),
                        [env.signal[j].location]*(green*10), lw=2, color='g')
                ax3.plot(np.linspace(cycle*k-(cycle-env.signal[j].offset)+green, cycle*k-(cycle-env.signal[j].offset)+cycle, red*10),
                        [env.signal[j].location]*(red*10), lw=2, color='r')
        ax3.set_title('x-t graph')
        ax3.set_xlabel('Time in s')
        ax3.set_ylabel('Position in m')
        ax3.set_xlim((0.0, math.ceil(timestep/10)))
        ax3.set_ylim((0, env.track_length))

    # t-reward
        ax4.scatter(reward[i][:, 0], reward[i][:, 1], color=color_list[i], s=15)
        ax4.plot(reward[i][:, 0], reward[i][:, 1], lw=2, color=color_list[i], alpha=0.5)
        ax4.set_title('reward-t graph')
        ax4.set_xlabel('Time in s')
        ax4.set_ylabel('Reward')
        ax4.set_xlim((0.0, math.ceil(timestep/10)))
        reward_min = np.min([np.round(np.min(env.reward_at_time[:, 1]), 0) for env in env_list])-2
        ax4.set_ylim((reward_min, 3.0))

    plt.subplots_adjust(hspace=0.35)


    if check_finish == True:
        plt.savefig(path)

    return fig

def info_graph_separate(env_list, veh_info_list, path='./simulate_gif/info_graph.png'):
    pos = [veh[:, 0] for veh in veh_info_list]
    vel = [veh[:, 1]*3.6 for veh in veh_info_list]
    acc = [veh[:, 2] for veh in veh_info_list]
    step = [veh[:, 3]/10 for veh in veh_info_list]
    reward = [env_list[i].reward_at_time[env_list[i].reward_at_time[:, 0] <= step[i][-1]] for i in range(len(env_list))]
    maxspeed = [veh[:, 5]*3.6 for veh in veh_info_list]

    # info figures
    plt.rc('font', size=15)
    plt.rc('axes', titlesize=22)
    plt.rc('axes', labelsize=15)
    plt.rc('xtick', labelsize=15)
    plt.rc('ytick', labelsize=15)

    # if check_finish == True:
    fig = plt.figure(figsize=(15*len(env_list), 20))
    fig.clf()
    fig_list = []
    for i in range(len(env_list)):
        ax1 = fig.add_subplot(4, len(env_list), (i+1))
        ax2 = fig.add_subplot(4, len(env_list), (i+1)+len(env_list)*1)
        ax3 = fig.add_subplot(4, len(env_list), (i+1)+len(env_list)*2)
        ax4 = fig.add_subplot(4, len(env_list), (i+1)+len(env_list)*3)
        fig_list.append([ax1, ax2, ax3, ax4])

    # else:
    #     fig = plt.figure(figsize=(15, 10))
    #     fig.clf()
    #     ax1 = fig.add_subplot(221)
    #     ax2 = fig.add_subplot(222)
    #     ax3 = fig.add_subplot(223)
    #     ax4 = fig.add_subplot(224)

    color_list = ['k', 'darkorange', 'indigo', 'b']

    for i in range(len(env_list)):
        env = env_list[i]

        ax1 = fig_list[i][0]
        ax2 = fig_list[i][1]
        ax3 = fig_list[i][2]
        ax4 = fig_list[i][3]

        # pos-vel
        # section_max_speed = env.section.sms_list[int(step[i][-1]*10)][1]
        # unit_length = env.unit_length
        # cur_idx = int(env.vehicle.position/env.unit_length)
        # if check_finish == False:
        #     # for i in range(len(section_max_speed)):
        #     for j in range(cur_idx, min(cur_idx+env.num_action_unit+1, len(env.section.section_max_speed)-1)):
        #         ax1.plot(np.linspace(j*unit_length, (j+1)*unit_length, unit_length*10),
        #                 [section_max_speed[j]*3.6]*(unit_length*10), lw=2, color='r')
        ax1.plot(pos[i], vel[i], lw=2, color=color_list[i])
        ax1.plot(pos[i], maxspeed[i], lw=1.5, color=color_list[i], alpha=0.5)
        ax1.set_title('x-v graph')
        ax1.set_xlabel('Position in m')
        ax1.set_ylabel('Velocity in km/h')
        ax1.set_xlim((0.0, env.track_length))
        ax1.set_ylim((0.0, 110))

        # pos-acc   
        ax2.plot(pos[i], acc[i], lw=2, color=color_list[i])
        ax2.set_title('x-a graph')
        ax2.set_xlabel('Position in m')
        ax2.set_ylabel('Acceleration in m/s²')
        ax2.set_xlim((0.0, env.track_length))
        ax2.set_ylim((env.acc_min-1, env.acc_max+1))

        # x-t with signal phase
        ax3.plot([x*env.dt for x in range(len(pos[i]))], pos[i], lw=2, color=color_list[i])
        green = env.signal[0].phase_length[True]
        red = env.signal[0].phase_length[False]
        cycle = green+red
        for j in range(env.num_signal):
            for k in range(int(env.timelimit/10/cycle)+1):
                ax3.plot(np.linspace(cycle*k-(cycle-env.signal[j].offset), cycle*k-(cycle-env.signal[j].offset)+green, green*10),
                        [env.signal[j].location]*(green*10), lw=2, color='g')
                ax3.plot(np.linspace(cycle*k-(cycle-env.signal[j].offset)+green, cycle*k-(cycle-env.signal[j].offset)+cycle, red*10),
                        [env.signal[j].location]*(red*10), lw=2, color='r')
        ax3.set_title('x-t graph')
        ax3.set_xlabel('Time in s')
        ax3.set_ylabel('Position in m')
        ax3.set_xlim((0.0, math.ceil(env.timestep/10)))
        ax3.set_ylim((0, env.track_length))

        # t-reward
        ax4.scatter(reward[i][:, 0], reward[i][:, 1], color=color_list[i], s=15)
        ax4.plot(reward[i][:, 0], reward[i][:, 1], lw=2, color=color_list[i], alpha=0.5)
        ax4.set_title('reward-t graph')
        ax4.set_xlabel('Time in s')
        ax4.set_ylabel('Reward')
        ax4.set_xlim((0.0, math.ceil(env.timestep/10)))
        reward_min = np.min([np.round(np.min(env.reward_at_time[:, 1]), 0) for env in env_list])-2
        ax4.set_ylim((reward_min, 3.0))

    plt.subplots_adjust(hspace=0.35)

    plt.savefig(path)

    return fig

def info_graph_detail(env_list, veh_info_list, separate = True, path='./simulate_gif/info_graph.png'):
    pos = [veh[:, 0] for veh in veh_info_list]
    vel = [veh[:, 1]*3.6 for veh in veh_info_list]
    acc = [veh[:, 2] for veh in veh_info_list]
    step = [veh[:, 3]/10 for veh in veh_info_list]
    maxspeed = [veh[:, 5]*3.6 for veh in veh_info_list]
    
    reward = [env.reward_per_unitlen for env in env_list]
    reward_coef = np.array(env_list[0].reward_coef)
    reward_idx = np.where(reward_coef != 0)[0]
    reward_coef = reward_coef[reward_coef != 0]
    reward_name = ["reward_norm_velocity",
                   "reward_shock",
                   "reward_jerk", 
                   "reward_power", 
                   "penalty_travel_time", 
                   "penalty_signal_violation", 
                   "reward_remaining_distance", 
                   "reward_action_gap"]
    reward_name = list(map(lambda x:reward_name[x], reward_idx))
    num_reward = len(reward_coef)

    timestep = np.max([env.timestep for env in env_list])

    # info figures
    plt.rc('font', size=15)
    plt.rc('axes', titlesize=22)
    plt.rc('axes', labelsize=15)
    plt.rc('xtick', labelsize=15)
    plt.rc('ytick', labelsize=15)

    if separate == True:
        fig = plt.figure(figsize=(15*(2+np.ceil(num_reward/3)), 20*len(env_list)))
        fig.clf()
        
        width_ratios = [3,3]
        for i in range(int(np.ceil(num_reward/3))):
            width_ratios.append(2)
        gs = GridSpec(nrows=3*len(env_list), ncols=int(2+np.ceil(num_reward/3)), width_ratios=width_ratios)
        
        ax_list = []
        for i in range(len(env_list)):
            temp = []
            temp.append(fig.add_subplot(gs[0+3*i:3+3*i, 0]))   # x-t
            temp.append(fig.add_subplot(gs[0+3*i, 1]))         # x-v
            temp.append(fig.add_subplot(gs[1+3*i, 1]))         # x-a
            temp.append(fig.add_subplot(gs[2+3*i, 1]))         # r-t
            for j in range(num_reward):
                if j < 3:
                    temp.append(fig.add_subplot(gs[j%3+3*i, 2]))
                elif 3 <= j < 6:
                    temp.append(fig.add_subplot(gs[j%3+3*i, 3]))
                elif 6 <= j < 8:
                    temp.append(fig.add_subplot(gs[j%3+3*i, 4]))

            ax_list.append(temp)

        color_list = ['k', 'darkorange', 'indigo', 'b']

        for i in range(len(env_list)):
            env = env_list[i]

            # x-t with signal phase
            ax_list[i][0].plot([x*env.dt for x in range(len(pos[i]))], pos[i], lw=2, color=color_list[i])
            green = env.signal[0].phase_length[True]
            red = env.signal[0].phase_length[False]
            cycle = green+red
            for j in range(env.num_signal):
                for k in range(int(env.timelimit/10/cycle)+1):
                    ax_list[i][0].plot(np.linspace(cycle*k-(cycle-env.signal[j].offset), cycle*k-(cycle-env.signal[j].offset)+green, green*10),
                            [env.signal[j].location]*(green*10), lw=2, color='g')
                    ax_list[i][0].plot(np.linspace(cycle*k-(cycle-env.signal[j].offset)+green, cycle*k-(cycle-env.signal[j].offset)+cycle, red*10),
                            [env.signal[j].location]*(red*10), lw=2, color='r')
            ax_list[i][0].set_title('x-t graph')
            ax_list[i][0].set_xlabel('Time in s')
            ax_list[i][0].set_ylabel('Position in m')
            ax_list[i][0].set_xlim((0.0, math.ceil(timestep/10)))
            ax_list[i][0].set_ylim((0, env.track_length))

            # pos-vel
            ax_list[i][1].plot(pos[i], vel[i], lw=2, color=color_list[i])
            ax_list[i][1].plot(pos[i], maxspeed[i], lw=1.5, color=color_list[i], alpha=0.5)
            ax_list[i][1].set_title('x-v graph')
            ax_list[i][1].set_xlabel('Position in m')
            ax_list[i][1].set_ylabel('Velocity in km/h')
            ax_list[i][1].set_xlim((0.0, env.track_length))
            ax_list[i][1].set_ylim((0.0, 110))

            # pos-acc
            ax_list[i][2].plot(pos[i], acc[i], lw=1, color=color_list[i])
            ax_list[i][2].set_title('x-a graph')
            ax_list[i][2].set_xlabel('Position in m')
            ax_list[i][2].set_ylabel('Acceleration in m/s²')
            ax_list[i][2].set_xlim((0.0, env.track_length))
            ax_list[i][2].set_ylim((env.acc_min-1, env.acc_max+1))
            
            reward_idx = np.append(reward_idx, -1)
            reward_list = [list(map(lambda x:x[1][j], reward[i])) for j in reward_idx]

            # t-total_reward
            t = list(map(lambda x:x[0], reward[i]))
            ax_list[i][3].scatter(t, reward_list[-1], color=color_list[i], s=12)
            ax_list[i][3].plot(t, reward_list[-1], lw=2, color=color_list[i], alpha=0.5)
            ax_list[i][3].set_title('reward-t graph')
            ax_list[i][3].set_xlabel('Time in s')
            ax_list[i][3].set_ylabel('Reward')
            ax_list[i][3].set_xlim((0.0, math.ceil(timestep/10)))

            for j in range(num_reward):
                ax_list[i][4+j].scatter(t, [reward_list[j][i]*reward_coef[j] for i in range(len(reward_list[j]))], color=color_list[i], s=12)
                ax_list[i][4+j].plot(t,[reward_list[j][i]*reward_coef[j] for i in range(len(reward_list[j]))], lw=2, color=color_list[i], alpha=0.5)
                ax_list[i][4+j].set_title('{}-t graph (Coef = {})'.format(reward_name[j], reward_coef[j]))
                ax_list[i][4+j].set_xlabel('Time in s')
                ax_list[i][4+j].set_ylabel('{}'.format(reward_name[j]))
                ax_list[i][4+j].set_xlim((0.0, math.ceil(timestep/10)))
        
    else:
        fig = plt.figure(figsize=(15*(2+np.ceil(num_reward/3)), 20))
        fig.clf()
        
        width_ratios = [3,3]
        for i in range(int(np.ceil(num_reward/3))):
            width_ratios.append(2)
        gs = GridSpec(nrows=3, ncols=int(2+np.ceil(num_reward/3)), width_ratios=width_ratios)
        
        ax_list = []
        ax_list.append(fig.add_subplot(gs[0:3, 0]))   # x-t
        ax_list.append(fig.add_subplot(gs[0, 1]))         # x-v
        ax_list.append(fig.add_subplot(gs[1, 1]))         # x-a
        ax_list.append(fig.add_subplot(gs[2, 1]))         # r-t
        for j in range(num_reward):
            if j < 3:
                ax_list.append(fig.add_subplot(gs[j%3, 2]))
            elif 3 <= j < 6:
                ax_list.append(fig.add_subplot(gs[j%3, 3]))
            elif 6 <= j < 8:
                ax_list.append(fig.add_subplot(gs[j%3, 4]))

        color_list = ['k', 'darkorange', 'indigo', 'b']

        for i in range(len(env_list)):
            env = env_list[i]

            # x-t with signal phase
            ax_list[0].plot([x*env.dt for x in range(len(pos[i]))], pos[i], lw=2, color=color_list[i])
            green = env.signal[0].phase_length[True]
            red = env.signal[0].phase_length[False]
            cycle = green+red
            for j in range(env.num_signal):
                for k in range(int(env.timelimit/10/cycle)+1):
                    ax_list[0].plot(np.linspace(cycle*k-(cycle-env.signal[j].offset), cycle*k-(cycle-env.signal[j].offset)+green, green*10),
                            [env.signal[j].location]*(green*10), lw=2, color='g')
                    ax_list[0].plot(np.linspace(cycle*k-(cycle-env.signal[j].offset)+green, cycle*k-(cycle-env.signal[j].offset)+cycle, red*10),
                            [env.signal[j].location]*(red*10), lw=2, color='r')
            ax_list[0].set_title('x-t graph')
            ax_list[0].set_xlabel('Time in s')
            ax_list[0].set_ylabel('Position in m')
            ax_list[0].set_xlim((0.0, math.ceil(timestep/10)))
            ax_list[0].set_ylim((0, env.track_length))

            # pos-vel
            ax_list[1].plot(pos[i], vel[i], lw=2, color=color_list[i])
            ax_list[1].plot(pos[i], maxspeed[i], lw=1.5, color=color_list[i], alpha=0.5)
            ax_list[1].set_title('x-v graph')
            ax_list[1].set_xlabel('Position in m')
            ax_list[1].set_ylabel('Velocity in km/h')
            ax_list[1].set_xlim((0.0, env.track_length))
            ax_list[1].set_ylim((0.0, 110))

            # pos-acc
            ax_list[2].plot(pos[i], acc[i], lw=1, color=color_list[i])
            ax_list[2].set_title('x-a graph')
            ax_list[2].set_xlabel('Position in m')
            ax_list[2].set_ylabel('Acceleration in m/s²')
            ax_list[2].set_xlim((0.0, env.track_length))
            ax_list[2].set_ylim((env.acc_min-1, env.acc_max+1))
            
            reward_idx = np.append(reward_idx, -1)
            reward_list = [list(map(lambda x:x[1][j], reward[i])) for j in reward_idx]

            # t-total_reward
            t = list(map(lambda x:x[0], reward[i]))
            ax_list[3].scatter(t, reward_list[-1], color=color_list[i], s=12)
            ax_list[3].plot(t, reward_list[-1], lw=2, color=color_list[i], alpha=0.5)
            ax_list[3].set_title('reward-t graph')
            ax_list[3].set_xlabel('Time in s')
            ax_list[3].set_ylabel('Reward')
            ax_list[3].set_xlim((0.0, math.ceil(timestep/10)))

            for j in range(num_reward):
                ax_list[4+j].scatter(t, [reward_list[j][i]*reward_coef[j] for i in range(len(reward_list[j]))], color=color_list[i], s=12)
                ax_list[4+j].plot(t,[reward_list[j][i]*reward_coef[j] for i in range(len(reward_list[j]))], lw=2, color=color_list[i], alpha=0.5)
                ax_list[4+j].set_title('{}-t graph (Coef = {})'.format(reward_name[j], reward_coef[j]))
                ax_list[4+j].set_xlabel('Time in s')
                ax_list[4+j].set_ylabel('{}'.format(reward_name[j]))
                ax_list[4+j].set_xlim((0.0, math.ceil(timestep/10)))

        plt.subplots_adjust(hspace=0.5)

    plt.savefig(path, bbox_inches="tight")

    return fig

def car_moving(env, veh_info, startorfinish=False, combine=False):
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
    for i in range(env.num_signal):
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
    finish_position = (zero_x + int(scale_x * (env.track_length - pos[-1])), canvas[1] - clearence[1])
    # signal_position = (zero_x + int(scale_x * (env.signal.location - pos[-1])), canvas[1] - clearence[1] - 50)
    signal_position = []
    for i in range(env.num_signal):
        signal_position.append((zero_x + int(scale_x * (env.signal[i].location - pos[-1])), canvas[1] - clearence[1] - 50))
    car_position = (zero_x - 80, canvas[1] - clearence[1] + 30)

    try:
        font = ImageFont.truetype('arial.ttf', 25)
        timefont = ImageFont.truetype('arial.ttf', 40)
    except:
        font = ImageFont.load_default()
        timefont = ImageFont.load_default()

    background = Image.new('RGB', canvas, (255, 255, 255))
    draw = ImageDraw.Draw(background)
    for i in range(int(env.track_length//50+1)):
        draw.text((zero_x - int(scale_x * (pos[-1])) + int(scale_x*50*i), canvas[1] - 90), "{}m".format(50*i), (0, 0, 0), font)
    draw.line((0, canvas[1]-100, 1500, canvas[1]-100), (0, 0, 0), width=5)
    background.paste(start, start_position)
    background.paste(finish, finish_position)
    # signal_draw = ImageDraw.Draw(signal)
    signal_draw = []
    for i in range(env.num_signal):
        signal_draw.append(ImageDraw.Draw(signal[i]))

        if env.signal[i].is_green(int(step[-1] * env.dt)):
            signal_draw[i].ellipse((0, 0, 60, 60,), (0, 255, 0))  # green signal
        else:
            signal_draw[i].ellipse((0, 0, 60, 60,), (255, 0, 0))  # red signal

        background.paste(signal[i], signal_position[i], signal[i])
    # background.paste(signal, signal_position, signal)
    background.paste(car, car_position, car)

    # print("make car-moving: {}".format(time.time()-t2))

    # env.info_graph(ob_list)
    # graph = Image.open('./simulate_gif/graph_{}.png'.format(time[-1]))

    if combine == True:
        # t3 = time.time()
        graph = fig2img(info_graph([env], [veh_info]))
        plt.close()
        background.paste(graph, (0, 50))
        # print("convert: {}".format(time.time()-t3))

    draw.text((10, 10), "Time Step: {}s".format(step[-1]/10), (0, 0, 0), timefont)

    if startorfinish == True:
        dup = 80
    else:
        dup = 1

    return [background] * dup

def make_gif(self, env_num=0):
    # env = pickle.load(open(f"simulate_gif/scn_{i}_plot_info.pkl", "rb"))
    self.png_list = []
    start = car_moving(self, self.vehicle.veh_info[:1], startorfinish=True, combine=True)
    mid = [car_moving(self, self.vehicle.veh_info[:i+2], startorfinish=False, combine=True) for i in tqdm(range(self.timestep))]
    end = car_moving(self, self.vehicle.veh_info[:self.timestep+1], startorfinish=True, combine=True)

    self.png_list = start + list(chain(*mid)) + end

    self.png_list[0].save(f"simulate_gif/scn_{0}_carmoving_env{env_num}.gif", save_all=True, append_images=self.png_list[1:], optimize=False, duration=30, loop=1)

def fig2img(fig):
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)

    return img