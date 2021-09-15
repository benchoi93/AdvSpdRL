import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import gym
import numpy as np
import os
import io
import glob
import math
import time
from gym import spaces


def info_graph(env_list, veh_info_list, check_finish=False, path='./simulate_gif/info_graph.png'):
    # t1 = time.time()
    pos = [veh[:, 0] for veh in veh_info_list]
    vel = [veh[:, 1]*3.6 for veh in veh_info_list]
    acc = [veh[:, 2] for veh in veh_info_list]
    step = [veh[:, 3]/10 for veh in veh_info_list]
    reward = [env_list[i].reward_at_time[env_list[i].reward_at_time[:, 0] <= step[i][-1]] for i in range(len(env_list))]
    maxspeed = [veh[:, 5]*3.6 for veh in veh_info_list]

    print(np.shape(pos[0]))
    print(np.shape(pos[1]))
    # print(pos[1][-2])

    # pos = veh_info_list[:, :, 0]
    # vel = veh_info_list[:, :, 1]*3.6
    # acc = veh_info_list[:, :, 2]
    # step = veh_info_list[:, :, 3]/10
    # # reward = veh_info_list[:, 4]
    # reward = env.reward_at_time[env.reward_at_time[:, 0] <= step[-1]]
    # maxspeed = veh_info_list[:, :, 5]*3.6

    # print(reward)
    # print("time:", step[-1])

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

    # pos-vel
    for i in range(len(env_list)):
        env = env_list[i]
        section_max_speed = env.section.sms_list[int(step[i][-1]*10)][1]
        # print(section_max_speed)
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
    for i in range(len(env_list)):
        env = env_list[i]
        ax2.plot(pos[i], acc[i], lw=2, color=color_list[i])
        ax2.set_title('x-a graph')
        ax2.set_xlabel('Position in m')
        ax2.set_ylabel('Acceleration in m/s²')
        ax2.set_xlim((0.0, env.track_length))
        ax2.set_ylim((env.acc_min-1, env.acc_max+1))

    # x-t with signal phase
    for i in range(len(env_list)):
        env = env_list[i]
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
        # if step[-1] == 0:
        #     ax3.set_xlim((0.0, 10))
        # else:
        #     ax3.set_xlim((0.0, math.ceil(step[-1]/100)*10))
        ax3.set_xlim((0.0, math.ceil(env.timestep/10)))
        ax3.set_ylim((0, env.track_length))

    # t-reward
    # ax4.plot(step, reward, lw=2, color='k')
    # ax4.plot([-10, 240], [0, 0], lw=1, color='k')
    for i in range(len(env_list)):
        env = env_list[i]
        ax4.scatter(reward[i][:, 0], reward[i][:, 1], color=color_list[i], s=15)
        ax4.plot(reward[i][:, 0], reward[i][:, 1], lw=2, color=color_list[i], alpha=0.5)
        # xlim_max = int(max(100, len(pos)) * env.dt) + 1
        ax4.set_title('reward-t graph')
        ax4.set_xlabel('Time in s')
        ax4.set_ylabel('Reward')
        # if step[-1] == 0:
        #     ax4.set_xlim((0.0, 10))
        # else:
        #     ax4.set_xlim((0.0, math.ceil(step[-1]/100)*10))
        ax4.set_xlim((0.0, math.ceil(env.timestep/10)))
        reward_min = np.min([np.round(np.min(env.reward_at_time[:, 1]), 0) for env in env_list])-2
        ax4.set_ylim((reward_min, 3.0))

    plt.subplots_adjust(hspace=0.35)

    # print("make fig: {}".format(time.time()-t1))

    if check_finish == True:
        plt.savefig(path)

    return fig


def car_moving(env, veh_info, startorfinish=False, combine=False):
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
        graph = fig2img(info_graph(env, veh_info))
        plt.close()
        background.paste(graph, (0, 50))
        # print("convert: {}".format(time.time()-t3))

    draw.text((10, 10), "Time Step: {}s".format(step[-1]/10), (0, 0, 0), timefont)

    if startorfinish == True:
        # for i in range(80):
        #     env.png_list.append(background)
        dup = 80

    else:
        # env.png_list.append(background)
        dup = 1

    return [background] * dup


def fig2img(fig):
    # t4 = time.time()
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)

    # print("fig2img: {}".format(time.time()-t4))
    return img
