import numpy as np
import matplotlib.pyplot as plt

all_rewards = np.load('../PPO/all_rewards.npy')
len_x = np.arange(len(all_rewards))
plt.plot(len_x,all_rewards)
plt.xlabel('updates')
plt.ylabel('rewards')
plt.show()