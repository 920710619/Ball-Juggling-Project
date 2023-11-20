import numpy as np
import matplotlib.pyplot as plt

reward_nums = np.loadtxt('./Buffer/reward_nums.dat')   # load the reward data

x_num = 501    # dimension of reward figure

# Smooth the reward figure
window_size = 9
smooth_nums = np.convolve(reward_nums, np.ones(window_size)/window_size, mode='valid')

plt.title('DDPG Reward Figure')
plt.xlabel('Episode')
plt.ylabel('Reward')

plt.plot(np.arange(x_num), reward_nums, color='red', alpha=0.1)
plt.plot(np.arange(x_num - window_size + 1), smooth_nums, color='red', alpha=1)

plt.grid(color='gray', linestyle='-', linewidth=1, alpha=0.2)
#plt.savefig('./Buffer/DDPG_reward_figure.png', dpi = 500)

plt.show()
