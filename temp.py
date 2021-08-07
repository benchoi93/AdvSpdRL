import matplotlib
# matplotlib.use('Agg')

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import matplotlib.animation as mpani
import numpy as np

def simulation(ob_list):
    dfd

def info_graph(ob_list):
    pos = [ob[0] for ob in ob_list]
    vel = [ob[1] for ob in ob_list]
    acc = [ob[2] for ob in ob_list]
    time = [ob[3] for ob in ob_list]
    reward = [ob[4] for ob in ob_list]

    fig = plt.figure()
    ax1 = fig.add_subplot(511)
    ax2 = fig.add_subplot(512)
    ax3 = fig.add_subplot(513)
    ax4 = fig.add_subplot(514)
    ax5 = fig.add_subplot(515)

    




