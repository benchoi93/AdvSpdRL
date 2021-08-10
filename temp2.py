from PIL import Image, ImageDraw
import matplotlib
# matplotlib.use('Agg')

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import matplotlib.animation as mpani
import numpy as np
import time
ob_list = [[1,1,1,1,1]]
t1 = time.time()
        # info figures
plt.rc('font', size=15)
plt.rc('axes', titlesize=22)
plt.rc('axes', labelsize=15)
plt.rc('xtick', labelsize=15)
plt.rc('ytick', labelsize=15)
fig = plt.figure(figsize=(15, 10))
fig.clf()
pos = [ob[0] for ob in ob_list]
vel = [ob[1] for ob in ob_list]
acc = [ob[2] for ob in ob_list]
step = [ob[3] for ob in ob_list]
reward = [ob[4] for ob in ob_list]

# pos-vel
ax1 = fig.add_subplot(221)
ax1.plot(pos, vel, lw=2, color='k')
ax1.set_title('x-v graph')
ax1.set_xlabel('Position in m')
ax1.set_ylabel('Velocity in km/h')
ax1.set_xlim((0.0, 500))
ax1.set_ylim((0.0, 50))

# pos-acc
ax2 = fig.add_subplot(222)
ax2.plot(pos, acc, lw=2, color='k')
ax2.set_title('x-a graph')
ax2.set_xlabel('Position in m')
ax2.set_ylabel('Acceleration in m/s²')
ax2.set_xlim((0.0, 500))
ax2.set_ylim((-3, 3))
# ax2.set_ylim((np.min(acc), np.max(acc)))

# x-t with signal phase
ax3 = fig.add_subplot(223)
ax3.plot([1],[1], lw=2, color='k')

ax3.set_title('x-t graph')
ax3.set_xlabel('Time in s')
ax3.set_ylabel('Position in m')
ax3.set_xlim((0.0, 500))
ax3.set_ylim((0, 500))

# t-reward
ax4 = fig.add_subplot(224)
ax4.plot([1],[1], lw=2, color='k')

ax4.set_title('x-t graph')
ax4.set_xlabel('Time in s')
ax4.set_ylabel('Position in m')
ax4.set_xlim((0.0, 500))
ax4.set_ylim((0, 500))

# fig.tight_layout()
plt.subplots_adjust(hspace=0.35)

plt.savefig('./simulate_gif/info_graph.png')
