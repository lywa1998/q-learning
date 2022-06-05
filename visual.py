import numpy as np
import matplotlib.pyplot as plt

data = np.zeros((500,))
data1 = np.load(f"./save/qlearning.npy")
# for ii in range(500):
#     data[ii] = np.mean(data1[max(ii-10, 0):ii])
data2 = np.load("./save/sarsa.npy")
plt.plot(data1, label='Q-Learning')
plt.plot(data2, label='Sarsa')
# plt.ylim(-100, 0)
plt.title("Reward")
plt.legend()
plt.show()